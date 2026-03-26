import math
from collections import defaultdict


def tile_allocation(
    workload,
    macro_num=1,
    core_num=1,
    array_col=1,
    array_row=1,
    cim_depth=1,
    precision=1,
):

    base = math.ceil(
        (workload["C"] * workload.get("R", 1) * workload.get("S", 1) / array_row)
        * (workload["M"] * precision / array_col / cim_depth)
        / macro_num
        / core_num
    )
    float_base = (
        (workload["C"] * workload.get("R", 1) * workload.get("S", 1) / array_row)
        * (workload["M"] * precision / array_col / cim_depth)
        / macro_num
        / core_num
    )
    # Ensure the base requirement is at least 1 (avoid 0)

    return max(base, 1)


def greedy_tile_allocation_group(
    layers, all_layers, workload, compute, total_tile, tile_allocation_min
):
    """
    Greedy tile allocation for a group of layers.

    Args:
    - layers: List of layer names in the current group
    - all_layers: Sequential list of all layers (used for indexing)
    - workload: List of workload objects for all layers
    - compute: List of computation volumes for all layers
    - total_tile: Total available tiles for the current group (e.g., 32)
    - tile_allocation_min: List of minimum tile allocations for all layers

    Returns:
    - tile_allocs: Final tile allocation list for each layer in the current group
    """

    # Find the indices of the layers within the group
    idx_list = [all_layers.index(layer) for layer in layers]

    # Get the minimum tile allocation for each layer
    min_tiles = [tile_allocation_min[idx] for idx in idx_list]
    total_min = sum(min_tiles)

    if total_min > total_tile:
        raise ValueError(
            f"Minimum tile requirement {total_min} for the group exceeds the maximum limit {total_tile}, please check tile_allocation"
        )

    # Calculate the computation volume for each layer in the group
    group_compute = [compute[idx] for idx in idx_list]
    total_compute = sum(group_compute)

    # Allocate remaining tiles
    remaining_tile = total_tile - total_min
    extra_tiles = [0 for _ in layers]

    if remaining_tile > 0 and total_compute > 0:
        for i in range(len(layers)):
            # Proportionally allocate extra tiles
            ratio = group_compute[i] / total_compute
            extra_tiles[i] = int(ratio * remaining_tile)

        # There might be leftover tiles (due to rounding)
        leftover = remaining_tile - sum(extra_tiles)
        if leftover > 0:
            # Greedily allocate remaining tiles to the layers with the highest computation volume
            sorted_idx = sorted(
                range(len(layers)), key=lambda i: group_compute[i], reverse=True
            )
            for i in sorted_idx:
                extra_tiles[i] += 1
                leftover -= 1
                if leftover == 0:
                    break

    # Final tile allocation = minimum tiles + greedy allocation
    tile_allocs = [min_tiles[i] + extra_tiles[i] for i in range(len(layers))]
    return tile_allocs


def proportional_tile_allocation_group(
    layers, all_layers, compute, total_tile, tile_allocation_min
):
    # Get the indices of layers within the group
    idx_list = [all_layers.index(layer) for layer in layers]
    group_compute = [compute[idx] for idx in idx_list]
    min_tiles = [tile_allocation_min[idx] for idx in idx_list]

    total_compute = sum(group_compute)

    # First proportional allocation, satisfying the compute ratio initially
    raw_alloc = [
        total_tile * (group_compute[i] / total_compute)
        for i in range(len(group_compute))
    ]

    # Forcefully satisfy the min_tile constraint
    alloc = [max(math.floor(raw_alloc[i]), min_tiles[i]) for i in range(len(raw_alloc))]

    # Adjust so the sum equals total_tile
    allocated = sum(alloc)
    diff = total_tile - allocated

    # If there is still a remainder, allocate it based on the compute ratio
    if diff > 0:
        fractional = [raw_alloc[i] - alloc[i] for i in range(len(raw_alloc))]
        sorted_indices = sorted(
            range(len(fractional)), key=lambda i: (-fractional[i], i)
        )
        for i in range(diff):
            alloc[sorted_indices[i % len(sorted_indices)]] += 1
    elif diff < 0:
        # Subtract any excess from where it is most over-allocated
        excess = [alloc[i] - raw_alloc[i] for i in range(len(alloc))]
        sorted_indices = sorted(range(len(excess)), key=lambda i: (-excess[i], i))
        for i in range(-diff):
            for j in sorted_indices:
                if alloc[j] > min_tiles[j]:
                    alloc[j] -= 1
                    break

    # Error function
    def compute_error(alloc_vec):
        total_alloc = sum(alloc_vec)
        return sum(
            abs((alloc_vec[i] / total_alloc) - (group_compute[i] / total_compute))
            for i in range(len(alloc_vec))
        )

    # Pairwise optimization (try to preserve proportions)
    max_attempts = 100
    attempt = 0
    while attempt < max_attempts:
        odd_indices = [
            i
            for i in range(len(alloc))
            if alloc[i] % 2 == 1 and alloc[i] > min_tiles[i]
        ]
        if len(odd_indices) < 2:
            break

        best_error = compute_error(alloc)
        best_pair = None
        for i in range(len(odd_indices)):
            for j in range(i + 1, len(odd_indices)):
                alloc_try = alloc.copy()
                alloc_try[odd_indices[i]] -= 1
                alloc_try[odd_indices[j]] += 1
                error = compute_error(alloc_try)
                if error < best_error:
                    best_error = error
                    best_pair = (odd_indices[i], odd_indices[j])

        if best_pair:
            alloc[best_pair[0]] -= 1
            alloc[best_pair[1]] += 1
            attempt += 1
        else:
            break

    # Return format remains unchanged
    return alloc


def greedy_tile_allocation(
    group_layers,
    compute_workloads,
    min_tiles,
    max_total_tiles,
    head_num=1,
    transformer=False,
    multi_batch=False,
):
    """
    Greedy allocation function ensuring minimum requirements are met and allocating proportionally to computation volume.

    :param group_layers: List of layer identifiers within the group
    :param compute_workloads: Computation workload for each layer (corresponding to group_layers)
    :param min_tiles: Minimum required tiles for each layer (corresponding to group_layers)
    :param max_total_tiles: Maximum total number of tiles
    :param head_num: Special parameter used for total tile calculation in Transformer models
    :param transformer: Whether it is a Transformer model, determines the total tile calculation formula
    :return: Allocation scheme dictionary {layer_name: allocated_tiles}
    """
    # Create a mapping from layer name to computation volume and minimum tiles
    compute_dict = {
        layer: compute for layer, compute in zip(group_layers, compute_workloads)
    }
    min_tile_dict = {layer: min_t for layer, min_t in zip(group_layers, min_tiles)}

    # 1. Initialize allocation: satisfy minimum requirements
    alloc = min_tile_dict.copy()

    # 2. Calculate the current total number of tiles (using different formulas depending on whether it's a Transformer)
    def calculate_total_tiles(alloc_dict):
        """Calculate total tiles"""
        if transformer:
            # Transformer model: Total tiles = Total normal tiles + (head_num-1) * (tiles for A + tiles for Z0)
            total_normal = sum(alloc_dict.values())
            tile_A = alloc_dict.get("A", 0)
            tile_Z0 = alloc_dict.get("Z0", 0)
            if multi_batch:
                return total_normal + 2 * (head_num - 1) * (tile_A + tile_Z0)
            else:
                return total_normal + (head_num - 1) * (tile_A + tile_Z0)
        else:
            # Non-Transformer model: Total tiles = Sum of all tiles
            return sum(alloc_dict.values())

    current_total = calculate_total_tiles(alloc)

    # 3. If the initial allocation exceeds the limit, raise an error
    """
    if current_total > max_total_tiles:
        raise ValueError("Cannot meet the minimum requirement, initial allocation exceeds the maximum tile limit")
    """

    # 4. Remaining tiles to allocate
    remaining_tiles = max_total_tiles - current_total

    # 5. Calculate total computation volume
    total_compute = sum(compute_dict.values())
    if total_compute <= 0:
        return alloc  # Computation volume for all layers is 0, return directly

    # 6. Calculate the extra tiles each layer should get (proportional to computation)
    # First calculate proportional allocation (float)
    proportional_extra = {}
    for layer in group_layers:
        proportional_extra[layer] = (
            remaining_tiles * compute_dict[layer] / total_compute
        )

    # 7. Initial integer allocation (floor)
    extra_alloc = {
        layer: math.floor(proportional_extra[layer]) for layer in group_layers
    }

    # 8. Allocate the remainder (distributed proportionally to computation)
    allocated_extra = sum(extra_alloc.values())
    remaining_extra = remaining_tiles - allocated_extra

    # Sort by the fractional part size (ensure distribution by computation proportion)
    fractional_parts = [
        (layer, proportional_extra[layer] - extra_alloc[layer])
        for layer in group_layers
    ]
    # Sort by fractional part descending (higher computation takes priority)
    fractional_parts.sort(key=lambda x: -x[1])

    # Allocate the remainder (strictly by computation proportion)
    for i in range(int(remaining_extra)):
        # Find the layer with the largest fractional part currently (highest computation ratio)
        layer = fractional_parts[0][0]

        # Increase the allocation for this layer
        extra_alloc[layer] += 1

        # Update the fractional part for this layer
        fractional_parts[0] = (layer, fractional_parts[0][1] - 1)

        # Resort to maintain the computation proportion
        fractional_parts.sort(key=lambda x: -x[1])

    # 9. Apply extra allocation
    for layer in group_layers:
        alloc[layer] += extra_alloc[layer]

    # 10. Verify total tile count
    final_total = calculate_total_tiles(alloc)
    if final_total > max_total_tiles:
        # If exceeded, adjustments are needed
        over = final_total - max_total_tiles

        # Try to reduce from the layer that is most over-allocated
        while over > 0:
            # Find the layers with the most over-allocated portion beyond their proportion
            excesses = [
                (layer, alloc[layer] - min_tile_dict[layer] - proportional_extra[layer])
                for layer in group_layers
                if alloc[layer] > min_tile_dict[layer]
            ]

            if not excesses:
                break  # Cannot reduce any further

            # Sort by over-allocation ratio
            excesses.sort(key=lambda x: (-x[1], x[0]))
            layer_to_reduce = excesses[0][0]

            # Reduce the allocation for this layer
            alloc[layer_to_reduce] -= 1

            # Calculate the impact on total tiles after reduction
            if transformer:
                # Transformer model: Special layers have extra impact
                if layer_to_reduce in ["A", "Z0"]:
                    over -= head_num  # Reducing 1 tile of a special layer is equivalent to reducing head_num total tiles
                else:
                    over -= 1
            else:
                # Non-Transformer model: Reducing 1 tile of any layer reduces 1 total tile
                over -= 1

    # 11. Final validation
    if calculate_total_tiles(alloc) > max_total_tiles:
        raise ValueError(
            "Cannot find an allocation scheme that satisfies the constraints"
        )

    return alloc


def compute_time_statistics_cnn(
    layers_cal, layers_weight, weight_is_duration=True, debug=False
):
    """
    Robust sweep-line for CNN-style inputs.
    - layers_cal: nested structure (e.g. [(s,e), ...]  or  [[(s,e)], [(s,e)], ...])
    - layers_weight: nested structure, commonly (start, duration) pairs wrapped in lists
    - weight_is_duration: if True, interpret weight pairs as (start, duration)
    Returns: cal_only, weight_only, overlap
    """
    import math

    def extract_intervals(obj, is_weight=False):
        intervals = []

        def visit(x):
            # numeric pair (tuple/list) e.g. (s,e) or (s,duration)
            if (
                isinstance(x, (tuple, list))
                and len(x) == 2
                and all(isinstance(v, (int, float)) for v in x)
            ):
                s = float(x[0])
                second = float(x[1])
                if is_weight and weight_is_duration:
                    e = s + second
                else:
                    e = second
                if math.isfinite(s) and math.isfinite(e) and e > s:
                    intervals.append((s, e))
                return
            # if iterable (list/tuple) but not numeric pair, recurse
            if isinstance(x, (list, tuple)):
                for elem in x:
                    visit(elem)
                return
            # else ignore
            return

        visit(obj)
        return intervals

    cal_intervals = extract_intervals(layers_cal, is_weight=False)
    weight_intervals = extract_intervals(layers_weight, is_weight=True)

    if debug:
        print("cal_intervals (count={}):".format(len(cal_intervals)), cal_intervals)
        print(
            "weight_intervals (count={}):".format(len(weight_intervals)),
            weight_intervals,
        )

    if not cal_intervals and not weight_intervals:
        return 0.0, 0.0, 0.0

    # sweep-line events
    events = []
    for s, e in cal_intervals:
        events.append((s, "cal", +1))
        events.append((e, "cal", -1))
    for s, e in weight_intervals:
        events.append((s, "weight", +1))
        events.append((e, "weight", -1))

    # sort: time asc, starts (+1) before ends (-1)
    events.sort(key=lambda x: (x[0], -x[2]))

    cal_active = weight_active = 0
    prev_t = None
    cal_only = weight_only = overlap = 0.0

    for t, kind, delta in events:
        if prev_t is not None and t > prev_t:
            dt = t - prev_t
            if cal_active > 0 and weight_active > 0:
                overlap += dt
            elif cal_active > 0:
                cal_only += dt
            elif weight_active > 0:
                weight_only += dt
        if kind == "cal":
            cal_active += delta
        else:
            weight_active += delta
        prev_t = t

    return cal_only, weight_only, overlap


def compute_time_statistics(layers_cal, layers_weight):
    events = []  # (time, type, +1/-1)

    # cal interval: already in (start, end) format
    for block in layers_cal:
        for batch in block:
            for item in batch:
                if not item or len(item) != 2:
                    continue
                s, e = item
                events.append((s, "cal", +1))
                events.append((e, "cal", -1))

    # weight interval: in (start, duration) format, needs to be converted to (start, start+duration)
    for block in layers_weight:
        for batch in block:
            for item in batch:
                if not item or len(item) != 2:
                    continue
                s, dur = item
                e = s + dur
                events.append((s, "weight", +1))
                events.append((e, "weight", -1))

    # Sort by time
    events.sort(key=lambda x: (x[0], -x[2]))

    cal_active = 0
    weight_active = 0
    prev_time = None
    cal_only = weight_only = overlap = 0.0

    for time, kind, delta in events:
        if prev_time is not None and time > prev_time:
            duration = time - prev_time
            if cal_active > 0 and weight_active > 0:
                overlap += duration
            elif cal_active > 0:
                cal_only += duration
            elif weight_active > 0:
                weight_only += duration

        if kind == "cal":
            cal_active += delta
        else:
            weight_active += delta
        prev_time = time

    return cal_only, weight_only, overlap


def calculate_bubble_percent(layers_cal, layers_bubble):
    """
    Directly receive raw computation time and bubble time parameters to calculate the bubble time percentage for each layer.

    :param layers_cal: List of computation time intervals, each element is a (start, end) tuple
    :param layers_bubble: List of bubble time intervals, each element is a dict (key is start time, value is duration)
    :return: List of bubble time percentages for each layer
    """
    percent = []
    # Iterate through the data of each layer
    for cal_time, bubble_dict in zip(layers_cal, layers_bubble):
        # Calculate the total computation time of the current layer (end - start)
        cal_duration = cal_time[1] - cal_time[0]

        # Handle cases where bubble time is empty
        if not bubble_dict:
            percent.append(0)
            continue

        # Calculate the total bubble time of the current layer (sum all values in the dict)
        total_bubble = sum(bubble_dict.values())

        # Calculate percentage and append to results
        percent.append((total_bubble / cal_duration) * 100)

    return percent


def capacity_aware_transformer_grouping(
    weights, layer_names, arch_size, ops_per_block=None
):
    """
    [Hierarchical Capacity-Aware Grouping]
    Implements a 3-level top-down grouping strategy based on Transformer semantics:
    - Level 1 (Block Level): If the entire Block fits, group all operators together.
    - Level 2 (Semantic Level): Split into Attention (Q, K, V, A, Z0) and FFN (Z1, FFN1, FFN2).
    - Level 3 (Fine-grained Level): Fallback to greedy 1D split strictly following dataflow order.
    """
    import logging

    num_ops = len(weights)
    total_weight = sum(weights)

    logging.info(
        f"🚀 [Hierarchical Grouping] Starting multi-level capacity-aware grouping..."
    )
    logging.info(f"   - Hardware single-stage capacity limit: {arch_size:,.0f} bits")
    logging.info(f"   - Single Block total workload: {total_weight:,.0f} bits")

    # ==========================================
    # 🔥 CORE FIX: Enforce Strict Topological Dataflow Order
    # ==========================================
    standard_order = ["Q", "K", "V", "A", "Z0", "Z1", "FFN1", "FFN2"]

    # Create a mapping from operator name to its original index in layer_names
    name_to_idx = {name.split(".")[0]: i for i, name in enumerate(layer_names)}

    # Reorder indices strictly following the execution sequence
    ordered_indices = [
        name_to_idx[name] for name in standard_order if name in name_to_idx
    ]

    # Safety fallback: append any unknown layers to the end
    for i, name in enumerate(layer_names):
        if name.split(".")[0] not in standard_order:
            ordered_indices.append(i)

    # ==========================================
    # Level 1: Whole Block Level Parallelism
    # ==========================================
    if total_weight <= arch_size:
        fit_blocks = arch_size // total_weight
        logging.info(f"   ✅ [Level 1] Hardware capacity is sufficient.")
        logging.info(
            f"   ✅ Applying [Whole-Block] strategy: Grouping all {num_ops} operators into a single Pipeline Stage."
        )
        return [ordered_indices]  # Return sorted by execution order

    # ==========================================
    # Level 2: Semantic Sub-Block Level (Attention vs FFN)
    # ==========================================
    logging.info(
        f"   ⚠️ [Level 1 Failed] Single Block exceeds capacity. Attempting Semantic Sub-Block split..."
    )

    attn_ops = ["Q", "K", "V", "A", "Z0"]
    ffn_ops = ["Z1", "FFN1", "FFN2"]

    attn_indices = [name_to_idx[op] for op in attn_ops if op in name_to_idx]
    ffn_indices = [name_to_idx[op] for op in ffn_ops if op in name_to_idx]

    # Check if the layers match the standard 8-operator Transformer template
    if (
        len(attn_indices) + len(ffn_indices) == num_ops
        and len(attn_indices) > 0
        and len(ffn_indices) > 0
    ):
        attn_weight = sum(weights[i] for i in attn_indices)
        ffn_weight = sum(weights[i] for i in ffn_indices)

        if attn_weight <= arch_size and ffn_weight <= arch_size:
            logging.info(f"   ✅ [Level 2] Semantic split successful!")
            logging.info(
                f"      - Stage 1 [Attention (Q,K,V,A,Z0)]: {attn_weight:,.0f} bits"
            )
            logging.info(f"      - Stage 2 [FFN (Z1,FFN1,FFN2)]: {ffn_weight:,.0f} bits")
            return [attn_indices, ffn_indices]
        else:
            logging.info(
                f"   ⚠️ [Level 2 Failed] Sub-modules (Attention: {attn_weight:,.0f}, FFN: {ffn_weight:,.0f}) still exceed limits."
            )
    else:
        logging.info(
            f"   ⚠️ [Level 2 Skipped] Operator names do not match standard Attention/FFN structure."
        )

    # ==========================================
    # Level 3: Fine-Grained Greedy 1D Split (Strictly Ordered)
    # ==========================================
    logging.info(
        f"   🔥 [Level 3] Falling back to fine-grained greedy 1D split (Following Dataflow)..."
    )
    groups = []
    current_group = []
    current_weight_sum = 0

    # Traverse strictly based on the execution order (ordered_indices)
    for op_idx in ordered_indices:
        w = weights[op_idx]
        op_name = layer_names[op_idx].split(".")[0]

        # Isolated huge operator
        if w > arch_size:
            if current_group:
                groups.append(current_group)
                current_group = []
                current_weight_sum = 0
            groups.append([op_idx])
            logging.info(
                f"      🚨 [Isolation] Operator [{op_name}] ({w:,.0f} bits) exceeds capacity! Grouped independently."
            )
            continue

        # Greedy split based on ordered sequence
        if current_weight_sum + w > arch_size:
            groups.append(current_group)
            names = [layer_names[i].split(".")[0] for i in current_group]
            logging.info(
                f"      ✅ [Stage {len(groups)}] {names} | Accumulated: {current_weight_sum:,.0f} bits"
            )
            current_group = [op_idx]
            current_weight_sum = w
        else:
            current_group.append(op_idx)
            current_weight_sum += w

    # Flush the remaining operators
    if current_group:
        groups.append(current_group)
        names = [layer_names[i].split(".")[0] for i in current_group]
        logging.info(
            f"      ✅ [Stage {len(groups)}] {names} | Accumulated: {current_weight_sum:,.0f} bits"
        )

    logging.info(
        f"🎉 Grouping complete! Fallback generated {len(groups)} Pipeline Stages.\n"
    )
    return groups
