import math
from collections import defaultdict


class TileAllocator:
    def __init__(
        self,
        layers,
        compute_workloads,
        min_tiles_per_layer,
        max_tiles_per_group=16,
        min_layers_per_group=1,
        max_layers_per_group=float("inf"),
        workload_shape=[],
    ):
        """
        :param layers: Ordered layer names.
        :param compute_workloads: Per-layer compute workload values.
        :param min_tiles_per_layer: Minimum required tiles per layer.
        :param max_tiles_per_group: Tile budget for each group.
        :param min_layers_per_group: Minimum number of layers per group.
        :param max_layers_per_group: Maximum number of layers per group.
        """
        self.layers = layers
        self.compute_workloads = compute_workloads
        self.min_tiles = min_tiles_per_layer
        self.max_tiles = max_tiles_per_group
        self.min_layers = min_layers_per_group
        self.max_layers = max_layers_per_group
        self.n_layers = len(layers)
        self.all_allocations = []
        self.workload_shape = workload_shape
        self.legal_tiles_map = {}

    def _get_layer_index(self, layer_name):
        return self.layers.index(layer_name)

    def _pick_spatial_extent(self, layer_name):
        """
        Select a representative spatial extent from workload shape.
        Priority: P > Q > M > C.
        """
        if not self.workload_shape:
            return None
        idx = self._get_layer_index(layer_name)
        shape = self.workload_shape[idx] if idx < len(self.workload_shape) else {}
        if not isinstance(shape, dict):
            return None
        for key in ("P", "Q", "M", "C"):
            val = shape.get(key)
            if isinstance(val, int) and val > 0:
                return val
        return None

    def _divisors_in_range(self, n, lo, hi):
        if n is None or n <= 0:
            return []
        result = set()
        root = int(math.sqrt(n))
        for d in range(1, root + 1):
            if n % d != 0:
                continue
            q = n // d
            if lo <= d <= hi:
                result.add(d)
            if lo <= q <= hi:
                result.add(q)
        return sorted(result)

    def get_layer_legal_tiles(
        self,
        layer_name,
        min_tile=None,
        max_tile=None,
        include_divisors=True,
        include_ceil_equivalence=False,
        include_powers_of_two=True,
    ):
        """
        Build legal tile candidates for one layer.
        - divisors: shape-aligned tile counts (strong prior).
        - ceil equivalence reps: first tile for each ceil(extent/tile) bucket.
        - powers of two: hardware-friendly anchors.
        """
        idx = self._get_layer_index(layer_name)
        lo = int(min_tile if min_tile is not None else self.min_tiles[idx])
        hi = int(max_tile if max_tile is not None else self.max_tiles)
        lo = max(1, lo)
        hi = max(lo, hi)

        candidates = {lo, hi}
        extent = self._pick_spatial_extent(layer_name)

        if include_divisors and extent is not None:
            candidates.update(self._divisors_in_range(extent, lo, hi))

        if include_ceil_equivalence and extent is not None:
            seen = set()
            for t in range(lo, hi + 1):
                eff = math.ceil(extent / t)
                if eff not in seen:
                    seen.add(eff)
                    candidates.add(t)

        if include_powers_of_two:
            p = 1
            while p < lo:
                p <<= 1
            while p <= hi:
                candidates.add(p)
                p <<= 1

        return sorted(candidates)

    def build_legal_tiles_map(
        self,
        max_tile=None,
        include_divisors=True,
        include_ceil_equivalence=False,
        include_powers_of_two=True,
    ):
        """
        Build legal tile sets for all layers and cache into self.legal_tiles_map.
        """
        tile_cap = int(max_tile if max_tile is not None else self.max_tiles)
        legal = {}
        for i, layer in enumerate(self.layers):
            legal[layer] = self.get_layer_legal_tiles(
                layer_name=layer,
                min_tile=self.min_tiles[i],
                max_tile=tile_cap,
                include_divisors=include_divisors,
                include_ceil_equivalence=include_ceil_equivalence,
                include_powers_of_two=include_powers_of_two,
            )
        self.legal_tiles_map = legal
        return legal

    def summarize_legal_tiles(self, legal_tiles_map=None, max_tile=None):
        """
        Return per-layer compression stats for quick inspection.
        """
        tile_cap = int(max_tile if max_tile is not None else self.max_tiles)
        legal = (
            legal_tiles_map
            if legal_tiles_map is not None
            else (
                self.legal_tiles_map
                if self.legal_tiles_map
                else self.build_legal_tiles_map(max_tile=tile_cap)
            )
        )
        summary = {}
        for i, layer in enumerate(self.layers):
            lo = max(1, int(self.min_tiles[i]))
            full = max(0, tile_cap - lo + 1)
            kept = len(legal.get(layer, []))
            ratio = (kept / full) if full > 0 else 1.0
            summary[layer] = {
                "min_tile": lo,
                "max_tile": tile_cap,
                "full_count": full,
                "legal_count": kept,
                "compression_ratio": ratio,
                "legal_tiles": legal.get(layer, []),
            }
        return summary

    def proportional_tile_allocation_group(
        self,
        group_layers,
        compute_workloads,
        min_tiles,
        max_total_tiles,
        head_num=1,
        transformer=False,
        multi_batch=False,
    ):
        """
        Proportional tile allocation within a group:
        - starts from per-layer minimum tile requirements,
        - distributes extra tiles by compute ratio,
        - keeps total assigned tiles within max_total_tiles.
        """
        import math

        projected_alloc = {}

        if len(group_layers) == 1:
            projected_alloc[group_layers[0]] = max_total_tiles
            return projected_alloc

        compute_dict = {
            layer: compute for layer, compute in zip(group_layers, compute_workloads)
        }
        min_tile_dict = {layer: min_t for layer, min_t in zip(group_layers, min_tiles)}

        legal_tile_dict = {}
        for layer in group_layers:
            legal_tile_dict[layer] = self.get_layer_legal_tiles(
                layer_name=layer,
                min_tile=min_tile_dict[layer],
                max_tile=max_total_tiles,
                include_divisors=True,
                include_ceil_equivalence=False,
                include_powers_of_two=True,
            )


        def calculate_total_tiles(alloc_dict):
            if transformer:
                total_normal = sum(alloc_dict.values())
                tile_A = alloc_dict.get("A", 0)
                tile_Z0 = alloc_dict.get("Z0", 0)
                if multi_batch:
                    return total_normal + 2 * (head_num - 1) * (tile_A + tile_Z0)
                else:
                    return total_normal + (head_num - 1) * (tile_A + tile_Z0)
            else:
                return sum(alloc_dict.values())


        total_compute = sum(compute_dict.values())
        if total_compute <= 0:
            for layer in group_layers:
                projected_alloc[layer] = min_tile_dict[layer]
            return projected_alloc

        proportional_extra = {
            layer: (max_total_tiles - sum(min_tile_dict.values()))
            * compute_dict[layer]
            / total_compute
            for layer in group_layers
        }
        alloc = {
            layer: min_tile_dict[layer] + math.floor(proportional_extra[layer])
            for layer in group_layers
        }

        for layer in group_layers:
            legal_set = legal_tile_dict[layer]
            candidates = [t for t in legal_set if t <= alloc[layer]]
            alloc[layer] = (
                max(candidates + [min_tile_dict[layer]])
                if candidates
                else min_tile_dict[layer]
            )

        leftover = max_total_tiles - sum(alloc.values())

        while leftover > 0:
            sorted_layers = sorted(group_layers, key=lambda x: alloc[x])
            distributed = False
            for layer in sorted_layers:
                if leftover <= 0:
                    break
                legal_set = legal_tile_dict[layer]
                bigger = [t for t in legal_set if t > alloc[layer]]
                if bigger:
                    next_val = min(bigger)
                    gap = next_val - alloc[layer]
                    if gap <= leftover:
                        alloc[layer] = next_val
                        leftover -= gap
                        distributed = True
            if not distributed:
                for layer in sorted_layers:
                    if leftover <= 0:
                        break
                    alloc[layer] += 1
                    leftover -= 1

        projected_alloc = alloc

        if calculate_total_tiles(projected_alloc) > max_total_tiles:
            raise ValueError("Group tile allocation exceeds max_total_tiles")

        return projected_alloc

    def explore_allocations(self, max_compute_ratio=4, min_compute_ratio=0.3):
        """
        Enumerate feasible group allocations with DFS pruning.
        :param max_compute_ratio: Upper ratio bound against ideal group compute.
        :param min_compute_ratio: Lower ratio bound against ideal group compute.
        """
        self.all_allocations = []

        total_compute = sum(self.compute_workloads)
        min_required_groups = max(1, math.ceil(sum(self.min_tiles) / self.max_tiles))
        self.ideal_compute = total_compute / min_required_groups

        self.max_compute_threshold = self.ideal_compute * max_compute_ratio
        self.min_compute_threshold = self.ideal_compute * min_compute_ratio

        self._dfs(0, self.max_tiles, [], [])
        return self.all_allocations

    def _dfs(self, layer_idx, remaining_tiles, current_group, allocation):
        """
        Depth-first search over grouping/allocation states.
        :param layer_idx: Current layer index.
        :param remaining_tiles: Remaining tile budget in the current group.
        :param current_group: Layers currently in the active group.
        :param allocation: Partial allocation result built so far.
        """
        if (
            not allocation
            and current_group
            and len(current_group) < 3
            and layer_idx < self.n_layers
        ):
            if not (
                len(current_group) < self.max_layers
                and self.min_tiles[layer_idx] <= remaining_tiles
            ):
                return

        if layer_idx >= self.n_layers:
            if current_group:
                group_size = len(current_group)
                if group_size >= self.min_layers and group_size <= self.max_layers:
                    if not allocation and group_size < 4:
                        return

                    try:
                        compute = []
                        min_tiles = []
                        for i in range(group_size):
                            compute.append(
                                self.compute_workloads[
                                    self.layers.index(current_group[i])
                                ]
                            )
                            min_tiles.append(
                                self.min_tiles[self.layers.index(current_group[i])]
                            )
                            # min_tiles = [self.min_tiles[self.layers.index(layer)] for layer in current_group]
                        group_allocation = self.proportional_tile_allocation_group(
                            current_group, compute, self.min_tiles, self.max_tiles
                        )
                        allocation.append((current_group.copy(), group_allocation))
                        self.all_allocations.append(allocation.copy())
                        allocation.pop()
                    except ValueError:
                        pass
            return

        layer_name = self.layers[layer_idx]
        min_req = self.min_tiles[layer_idx]

        can_add_to_current = (
            len(current_group) < self.max_layers and min_req <= remaining_tiles
        )

        if can_add_to_current:
            current_group.append(layer_name)

            self._dfs(
                layer_idx + 1,
                remaining_tiles - min_req,
                current_group,
                allocation,
            )

            current_group.pop()

        if current_group:
            group_size = len(current_group)
            if group_size >= self.min_layers and group_size <= self.max_layers:
                if not allocation and group_size < 4:
                    return

                try:
                    compute = []
                    min_tiles = []
                    for i in range(group_size):
                        compute.append(
                            self.compute_workloads[self.layers.index(current_group[i])]
                        )
                        min_tiles.append(
                            self.min_tiles[self.layers.index(current_group[i])]
                        )
                    group_allocation = self.proportional_tile_allocation_group(
                        current_group, compute, min_tiles, self.max_tiles
                    )
                    allocation.append((current_group.copy(), group_allocation))

                    new_group = [layer_name]

                    self._dfs(
                        layer_idx + 1, self.max_tiles - min_req, new_group, allocation
                    )

                    allocation.pop()
                except ValueError:
                    pass

        else:
            new_group = [layer_name]

            self._dfs(layer_idx + 1, self.max_tiles - min_req, new_group, allocation)

    def _dfs(self, layer_idx, remaining_tiles, current_group, allocation):
        if layer_idx >= self.n_layers:
            if current_group:
                group_size = len(current_group)
                if self.min_layers <= group_size <= self.max_layers:
                    try:
                        compute = [
                            self.compute_workloads[self.layers.index(l)]
                            for l in current_group
                        ]
                        min_tiles = [
                            self.min_tiles[self.layers.index(l)] for l in current_group
                        ]
                        group_allocation = self.proportional_tile_allocation_group(
                            current_group, compute, min_tiles, self.max_tiles
                        )
                        allocation.append((current_group.copy(), group_allocation))

                        self.all_allocations.append(allocation.copy())

                        allocation.pop()
                    except ValueError:
                        pass
            return

        layer_name = self.layers[layer_idx]
        min_req = self.min_tiles[layer_idx]
        next_compute = self.compute_workloads[layer_idx]

        if current_group:
            current_compute = sum(
                [self.compute_workloads[self.layers.index(l)] for l in current_group]
            )
            current_min_sum = sum(
                [self.min_tiles[self.layers.index(l)] for l in current_group]
            )
        else:
            current_compute = 0
            current_min_sum = 0

        group_size = len(current_group)

        physically_can_add = (group_size < self.max_layers) and (
            current_min_sum + min_req <= self.max_tiles
        )

        can_add_to_current = physically_can_add
        can_create_new_group = False

        if current_group:
            if group_size >= self.min_layers:
                can_create_new_group = True

            if physically_can_add and (
                current_compute + next_compute > self.max_compute_threshold
            ):
                can_add_to_current = False

            if (
                can_create_new_group
                and physically_can_add
                and (current_compute < self.min_compute_threshold)
            ):
                can_create_new_group = False
        else:
            can_add_to_current = True
            can_create_new_group = False

        if can_add_to_current:
            current_group.append(layer_name)
            self._dfs(
                layer_idx + 1, remaining_tiles - min_req, current_group, allocation
            )
            current_group.pop()

        if can_create_new_group:
            try:
                compute = [
                    self.compute_workloads[self.layers.index(l)] for l in current_group
                ]
                min_tiles = [
                    self.min_tiles[self.layers.index(l)] for l in current_group
                ]
                group_allocation = self.proportional_tile_allocation_group(
                    current_group, compute, min_tiles, self.max_tiles
                )

                allocation.append((current_group.copy(), group_allocation))

                new_group = [layer_name]
                self._dfs(
                    layer_idx + 1, self.max_tiles - min_req, new_group, allocation
                )

                allocation.pop()
            except ValueError:
                pass

    def print_allocations(self, allocations):
        """Print a readable summary of candidate allocations."""
        print(f"Total allocations: {len(allocations)}")
        for i, allocation in enumerate(allocations):
            print(f"\nAllocation #{i+1}:")
            total_groups = len(allocation)
            total_tiles_used = 0

            for group_idx, (group_layers, layer_allocations) in enumerate(allocation):
                group_size = len(group_layers)
                group_tiles = sum(layer_allocations.values())
                total_tiles_used += group_tiles
                print(
                    f"  Group {group_idx+1}/{total_groups} (layers: {group_size}, tiles: {group_tiles}/{self.max_tiles}):"
                )

                for layer in group_layers:
                    min_req = self.min_tiles[self.layers.index(layer)]
                    alloc = layer_allocations[layer]
                    compute = self.compute_workloads[self.layers.index(layer)]
                    print(
                        f"    Layer {layer}: tiles {alloc} (min {min_req}, compute {compute})"
                    )

            print(
                f"  Total tiles used: {total_tiles_used}/{total_groups * self.max_tiles}"
            )

    def find_best_allocation(self, metric="min_groups"):
        """Print a readable summary of candidate allocations."""
        if not self.all_allocations:
            self.explore_allocations()

        if metric == "min_groups":
            return min(self.all_allocations, key=len)
        elif metric == "max_utilization":
            best_allocation = None
            best_utilization = -1

            for allocation in self.all_allocations:
                total_tiles_used = 0
                total_tiles_available = len(allocation) * self.max_tiles

                for _, layer_allocations in allocation:
                    total_tiles_used += sum(layer_allocations.values())

                utilization = total_tiles_used / total_tiles_available

                if utilization > best_utilization:
                    best_utilization = utilization
                    best_allocation = allocation

            return best_allocation
        elif metric == "min_max_tiles":
            best_allocation = None
            min_max_tiles = float("inf")

            for allocation in self.all_allocations:
                max_tiles_in_group = 0
                for _, layer_allocations in allocation:
                    max_in_group = max(layer_allocations.values())
                    if max_in_group > max_tiles_in_group:
                        max_tiles_in_group = max_in_group

                if max_tiles_in_group < min_max_tiles:
                    min_max_tiles = max_tiles_in_group
                    best_allocation = allocation

            return best_allocation
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def analyze_allocations(self):
        """Analyze allocation frequency and layer grouping patterns."""
        if not self.all_allocations:
            self.explore_allocations()

        group_freq = defaultdict(int)
        for allocation in self.all_allocations:
            group_key = tuple(tuple(group) for group, _ in allocation)
            group_freq[group_key] += 1

        most_common = (
            max(group_freq.items(), key=lambda x: x[1]) if group_freq else (None, 0)
        )

        layer_positions = defaultdict(list)
        layer_groups = defaultdict(list)
        for allocation in self.all_allocations:
            for group_idx, (group, _) in enumerate(allocation):
                for layer in group:
                    layer_positions[layer].append(group_idx)
                    layer_groups[layer].append(group)

        avg_positions = {}
        for layer, positions in layer_positions.items():
            avg_positions[layer] = sum(positions) / len(positions)

        most_common_groups = {}
        for layer, groups in layer_groups.items():
            group_counter = defaultdict(int)
            for group in groups:
                group_counter[tuple(group)] += 1
            if group_counter:
                most_common_group = max(group_counter.items(), key=lambda x: x[1])
                most_common_groups[layer] = (most_common_group[0], most_common_group[1])

        return {
            "total_allocations": len(self.all_allocations),
            "most_common_grouping": most_common[0],
            "frequency": most_common[1],
            "avg_layer_positions": avg_positions,
            "most_common_groups": most_common_groups,
        }

