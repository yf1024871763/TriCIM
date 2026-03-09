import math
from collections import defaultdict
def tile_allocation(workload,macro_num=1,core_num=1,array_col=1,array_row=1,cim_depth=1,precision=1):
    
    base = math.ceil(
        (workload['C'] * workload.get('R', 1) * workload.get('S', 1) / array_row) * 
        (workload['M'] * precision / array_col / cim_depth) / macro_num / core_num
    )
    float_base =  (workload['C'] * workload.get('R', 1) * workload.get('S', 1) / array_row) *(workload['M'] * precision / array_col / cim_depth) / macro_num / core_num
    # 确保基础需求至少为1（避免0）
    return max(base, 1)
def greedy_tile_allocation_group(layers, all_layers, workload, compute, total_tile, tile_allocation_min):
    """
    对一组 layer 进行贪心 Tile 分配

    参数：
    - layers: 当前组的层名称列表
    - all_layers: 所有层的顺序列表（用于索引）
    - workload: 所有层的 workload 对象列表
    - compute: 所有层的计算量列表
    - total_tile: 当前组可用总 Tile 数（例如 32）
    - tile_allocation_min: 所有层的最小 Tile 分配列表

    返回：
    - tile_allocs: 当前组每层的最终 Tile 分配列表
    """

    # 找到组内层的索引
    idx_list = [all_layers.index(layer) for layer in layers]

    # 获取每层的最小tile分配
    min_tiles = [tile_allocation_min[idx] for idx in idx_list]
    total_min = sum(min_tiles)

    if total_min > total_tile:
        raise ValueError(f"组内最小Tile需求 {total_min} 超过了最大限制 {total_tile}，请检查tile_allocation")

    # 计算组内每层的计算量
    group_compute = [compute[idx] for idx in idx_list]
    total_compute = sum(group_compute)

    # 分配剩余tile
    remaining_tile = total_tile - total_min
    extra_tiles = [0 for _ in layers]

    if remaining_tile > 0 and total_compute > 0:
        for i in range(len(layers)):
            # 比例分配额外tile
            ratio = group_compute[i] / total_compute
            extra_tiles[i] = int(ratio * remaining_tile)

        # 有可能有tile剩余没分完（由于取整）
        leftover = remaining_tile - sum(extra_tiles)
        if leftover > 0:
            # 贪心分配剩余tile给计算量最大者
            sorted_idx = sorted(range(len(layers)), key=lambda i: group_compute[i], reverse=True)
            for i in sorted_idx:
                extra_tiles[i] += 1
                leftover -= 1
                if leftover == 0:
                    break

    # 最终tile分配 = 最小tile + 贪心分配
    tile_allocs = [min_tiles[i] + extra_tiles[i] for i in range(len(layers))]
    return tile_allocs
'''
def proportional_tile_allocation_group(layers, all_layers, compute, total_tile, tile_allocation_min):
    # 获取索引和计算数据
    idx_list = [all_layers.index(layer) for layer in layers]
    group_compute = [compute[idx] for idx in idx_list]
    min_tiles = [tile_allocation_min[idx] for idx in idx_list]
    
    # 计算总需求和检查最小值
    total_min = sum(min_tiles)

    if total_min > total_tile:
        raise ValueError(f"组内层 {layers} 的最小Tile总需求 {total_min} 超过最大 {total_tile}，请检查下限。")
    
    remaining_tile = total_tile - total_min
    total_compute = sum(group_compute)
    
    # === 一次性比例分配 ===
    # 1. 计算每层应得的浮点数值
    allocations = [min_tiles[i] + remaining_tile * (c / total_compute) 
                  for i, c in enumerate(group_compute)]
    
    # 2. 取整数部分作为基础分配
    base_alloc = [math.floor(a) for a in allocations]
    fractional = [alloc - base for alloc, base in zip(allocations, base_alloc)]
    
    # 3. 分配剩余Tile（基于小数部分）
    allocated = sum(base_alloc)
    remaining_tile = total_tile - allocated
    
    # 按小数部分排序（降序）
    sorted_indices = sorted(range(len(fractional)), key=lambda i: (-fractional[i], i))
    
    # 分配剩余Tile
    for i in range(remaining_tile):
        idx = sorted_indices[i % len(sorted_indices)]  # 循环分配确保公平
        base_alloc[idx] += 1
    
    # === 偶优化（智能调整）===
    # 1. 构建计算量到索引的映射
    compute_groups = defaultdict(list)
    for i, c in enumerate(group_compute):
        compute_groups[c].append(i)
    
    # 2. 首先在计算量相同的层内进行偶优化
    for _, indices in compute_groups.items():
        if len(indices) < 2:
            continue
            
        # 尝试将组内的奇数值调整为偶数
        for i in indices:
            if base_alloc[i] % 2 == 1 and base_alloc[i] > min_tiles[i]:
                # 在组内寻找可以调整的伙伴
                for j in indices:
                    if i != j and base_alloc[j] % 2 == 1 and base_alloc[j] > min_tiles[j]:
                        # 交换调整：i减1，j加1
                        base_alloc[i] -= 1
                        base_alloc[j] += 1
                        break
    
    # 3. 全局调整（跨不同计算量的层）
    max_attempts = 100
    attempt = 0
    while attempt < max_attempts:
        # 找出所有可调整的奇数层（大于最小值）
        odd_indices = [i for i in range(len(base_alloc)) 
                       if base_alloc[i] % 2 == 1 and base_alloc[i] > min_tiles[i]]
        
        if len(odd_indices) < 2:
            break
            
        # 成对调整
        i, j = odd_indices[0], odd_indices[1]
        base_alloc[i] -= 1
        base_alloc[j] += 1
        attempt += 1

    return base_alloc

def proportional_tile_allocation_group(layers, all_layers, compute, total_tile, tile_allocation_min):
    idx_list = [all_layers.index(layer) for layer in layers]
    group_compute = [compute[idx] for idx in idx_list]
    min_tiles = [tile_allocation_min[idx] for idx in idx_list]

    total_min = sum(min_tiles)
    if total_min > total_tile:
        raise ValueError(f"组内层 {layers} 的最小Tile总需求 {total_min} 超过最大 {total_tile}，请检查下限。")
    
    remaining_tile = total_tile - total_min
    total_compute = sum(group_compute)
    
    # 原始分配（浮点数）
    allocations_float = [min_tiles[i] + remaining_tile * (group_compute[i] / total_compute) 
                         for i in range(len(group_compute))]
    base_alloc = [math.floor(a) for a in allocations_float]
    fractional = [allocations_float[i] - base_alloc[i] for i in range(len(base_alloc))]
    
    # 分配剩余tile，尽量按比例最接近
    allocated = sum(base_alloc)
    leftover = total_tile - allocated
    sorted_indices = sorted(range(len(fractional)), key=lambda i: (-fractional[i], i))
    for i in range(leftover):
        base_alloc[sorted_indices[i]] += 1
    
    # 记录当前比例误差
    def compute_error(alloc):
        adjusted = [alloc[i] - min_tiles[i] for i in range(len(alloc))]
        adjusted_sum = sum(adjusted)
        if adjusted_sum == 0:
            return float('inf')
        return sum(abs((adjusted[i] / adjusted_sum) - (group_compute[i] / total_compute))
                   for i in range(len(alloc)))

    # 尝试偶优化（不会破坏分配比例太多）
    max_attempts = 100
    attempt = 0
    while attempt < max_attempts:
        odd_indices = [i for i in range(len(base_alloc)) 
                       if base_alloc[i] % 2 == 1 and base_alloc[i] > min_tiles[i]]
        if len(odd_indices) < 2:
            break

        # 遍历所有奇数组合，尝试交换并选误差最小的
        best_error = compute_error(base_alloc)
        best_pair = None
        for i in range(len(odd_indices)):
            for j in range(i + 1, len(odd_indices)):
                alloc_try = base_alloc.copy()
                alloc_try[odd_indices[i]] -= 1
                alloc_try[odd_indices[j]] += 1
                error = compute_error(alloc_try)
                if error < best_error:
                    best_error = error
                    best_pair = (odd_indices[i], odd_indices[j])
        
        if best_pair:
            base_alloc[best_pair[0]] -= 1
            base_alloc[best_pair[1]] += 1
            attempt += 1
        else:
            break
    
    return base_alloc
'''
def proportional_tile_allocation_group(layers,all_layers, compute,total_tile,tile_allocation_min):
        # 获取组内各层的索引
        idx_list = [all_layers.index(layer) for layer in layers]
        group_compute = [compute[idx] for idx in idx_list]
        min_tiles = [tile_allocation_min[idx] for idx in idx_list]

        total_compute = sum(group_compute)

        # 第一次按比例分配，先满足 compute 比例
        raw_alloc = [total_tile * (group_compute[i] / total_compute) for i in range(len(group_compute))]

        # 强制满足 min_tile 限制
        alloc = [max(math.floor(raw_alloc[i]), min_tiles[i]) for i in range(len(raw_alloc))]

        # 调整使总和为 total_tile
        allocated = sum(alloc)
        diff = total_tile - allocated

        # 若还有剩余，按 compute 比例分配剩余部分
        if diff > 0:
            fractional = [raw_alloc[i] - alloc[i] for i in range(len(raw_alloc))]
            sorted_indices = sorted(range(len(fractional)), key=lambda i: (-fractional[i], i))
            for i in range(diff):
                alloc[sorted_indices[i % len(sorted_indices)]] += 1
        elif diff < 0:
            # 超过的部分从最大超额的地方减去
            excess = [alloc[i] - raw_alloc[i] for i in range(len(alloc))]
            sorted_indices = sorted(range(len(excess)), key=lambda i: (-excess[i], i))
            for i in range(-diff):
                for j in sorted_indices:
                    if alloc[j] > min_tiles[j]:
                        alloc[j] -= 1
                        break

        # 误差函数
        def compute_error(alloc_vec):
            total_alloc = sum(alloc_vec)
            return sum(abs((alloc_vec[i] / total_alloc) - (group_compute[i] / total_compute)) for i in range(len(alloc_vec)))

        # 偶优化（尽量保留比例）
        max_attempts = 100
        attempt = 0
        while attempt < max_attempts:
            odd_indices = [i for i in range(len(alloc)) if alloc[i] % 2 == 1 and alloc[i] > min_tiles[i]]
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

        # 返回格式保持不变
        return alloc
'''
def greedy_tile_allocation(layers, layer_idx, workloads, total_tile=32, 
                                min_tile=1,  # 每个layer的最小分配量
                                macro_num=1, core_num=1, 
                                array_col=1, array_row=1, cim_depth=1, precision=1):
    """
    按比例分配Tile资源，确保每个layer至少获得min_tile个Tile
    """
    # 1. 计算基础需求（确保至少为min_tile）
    base_tiles = []
    for layer in layers:
        idx = layer_idx.index(layer)
        base = tile_allocation(
            workloads[idx], macro_num, core_num, 
            array_col, array_row, cim_depth, precision
        )
        base_tiles.append(max(base, min_tile))  # 强制不小于最小分配量
    
    # 2. 检查总最小需求是否超过总配额
    num_layers = len(layers)
    total_min = num_layers * min_tile
    if total_min > total_tile:
        # 若总最小需求超过配额，按比例缩减最小量（但仍保证每个至少1）
        scale = total_tile / total_min
        base_tiles = [max(math.ceil(base * scale), 1) for base in base_tiles]
    
    # 3. 计算原始比例（基于基础需求）
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    current_gcd = base_tiles[0]
    for val in base_tiles[1:]:
        current_gcd = gcd(current_gcd, val)
    ratios = [val // current_gcd for val in base_tiles]
    ratio_sum = sum(ratios)
    
    # 4. 分配剩余配额（总配额 - 已分配的最小量总和）
    allocated_min = sum(base_tiles)
    remaining = total_tile - allocated_min
    if remaining < 0:
        remaining = 0  # 防止负数
    
    allocations = base_tiles.copy()
    
    # 按比例分配剩余配额
    while remaining > 0:
        # 按比例分配1个Tile（优先分配给比例占比低的）
        current_sum = sum(allocations)
        ratios_current = [alloc / ratios[i] if ratios[i] > 0 else float('inf') 
                         for i, alloc in enumerate(allocations)]
        min_ratio_idx = ratios_current.index(min(ratios_current))
        
        allocations[min_ratio_idx] += 1
        remaining -= 1
    
    return allocations
'''
import math

def greedy_tile_allocation(group_layers, compute_workloads, min_tiles, max_total_tiles, head_num=1, transformer=False,multi_batch=False):
    """
    贪心分配函数，保证满足最小分配要求，并尽量按照计算量比例分配
    
    :param group_layers: 组内的层标识列表
    :param compute_workloads: 每层的计算负载（与group_layers顺序对应）
    :param min_tiles: 每层所需最小Tile数（与group_layers顺序对应）
    :param max_total_tiles: 最大总Tile数
    :param head_num: 特殊参数，用于Transformer模型的总Tile计算
    :param transformer: 是否为Transformer模型，决定总Tile计算公式
    :return: 分配方案字典 {层名: 分配的Tile数}
    """
    # 创建层名到计算量和最小Tile的映射
    compute_dict = {layer: compute for layer, compute in zip(group_layers, compute_workloads)}
    min_tile_dict = {layer: min_t for layer, min_t in zip(group_layers, min_tiles)}
    
    # 1. 初始化分配：满足最小需求
    alloc = min_tile_dict.copy()
    
    # 2. 计算当前总Tile数（根据是否为Transformer使用不同公式）
    def calculate_total_tiles(alloc_dict):
        """计算总Tile数"""
        if transformer:
            # Transformer模型：总Tile = 普通Tile总数 + (head_num-1)*(A的Tile数 + Z0的Tile数)
            total_normal = sum(alloc_dict.values())
            tile_A = alloc_dict.get('A', 0)
            tile_Z0 = alloc_dict.get('Z0', 0)
            if multi_batch:
                return total_normal + 2*(head_num - 1) * (tile_A + tile_Z0)
            else:
                return total_normal + (head_num - 1) * (tile_A + tile_Z0)
        else:
            # 非Transformer模型：总Tile = 所有Tile之和
            return sum(alloc_dict.values())
    
    current_total = calculate_total_tiles(alloc)
    # 3. 如果初始分配已超过限制，报错
    '''
    if current_total > max_total_tiles:
        raise ValueError("无法满足最小需求，初始分配已超过最大Tile限制")
    '''
    # 4. 剩余可分配的Tile数
    remaining_tiles = max_total_tiles - current_total
    
    # 5. 计算总计算量
    total_compute = sum(compute_dict.values())
    if total_compute <= 0:
        return alloc  # 所有层计算量为0，直接返回
    
    # 6. 计算每层应得的额外Tile数（按计算量比例）
    # 先计算比例分配（浮点数）
    proportional_extra = {}
    for layer in group_layers:
        proportional_extra[layer] = (remaining_tiles * compute_dict[layer] / total_compute)
    
    # 7. 初始整数分配（向下取整）
    extra_alloc = {layer: math.floor(proportional_extra[layer]) for layer in group_layers}
    
    # 8. 分配剩余的部分（按计算量比例分布）
    allocated_extra = sum(extra_alloc.values())
    remaining_extra = remaining_tiles - allocated_extra
    
    # 按小数部分大小排序（确保按计算量比例分布）
    fractional_parts = [
        (layer, proportional_extra[layer] - extra_alloc[layer])
        for layer in group_layers
    ]
    # 按小数部分降序排序（计算量高的优先）
    fractional_parts.sort(key=lambda x: -x[1])
    
    # 分配剩余的部分（严格按计算量比例）
    for i in range(int(remaining_extra)):
        # 找到当前小数部分最大的层（计算量比例最高的）
        layer = fractional_parts[0][0]
        
        # 增加该层的分配
        extra_alloc[layer] += 1
        
        # 更新该层的小数部分
        fractional_parts[0] = (layer, fractional_parts[0][1] - 1)
        
        # 重新排序以保持计算量比例
        fractional_parts.sort(key=lambda x: -x[1])
    
    # 9. 应用额外分配
    for layer in group_layers:
        alloc[layer] += extra_alloc[layer]
    
    # 10. 验证总Tile数
    final_total = calculate_total_tiles(alloc)
    if final_total > max_total_tiles:
        # 如果超过，需要调整
        over = final_total - max_total_tiles
        
        # 尝试从超额最多的层中减少
        while over > 0:
            # 找到分配超过比例的部分最多的层
            excesses = [
                (layer, alloc[layer] - min_tile_dict[layer] - proportional_extra[layer])
                for layer in group_layers
                if alloc[layer] > min_tile_dict[layer]
            ]
            
            if not excesses:
                break  # 无法再减少
                
            # 按超额比例排序
            excesses.sort(key=lambda x: (-x[1], x[0]))
            layer_to_reduce = excesses[0][0]
            
            # 减少该层的分配
            alloc[layer_to_reduce] -= 1
            
            # 计算减少后对总Tile的影响
            if transformer:
                # Transformer模型：特殊层有额外影响
                if layer_to_reduce in ['A', 'Z0']:
                    over -= head_num  # 减少1个特殊层Tile相当于减少head_num个总Tile
                else:
                    over -= 1
            else:
                # 非Transformer模型：所有层减少1个Tile就是减少1个总Tile
                over -= 1
    
    # 11. 最终验证'

    if calculate_total_tiles(alloc) > max_total_tiles:
        raise ValueError("无法找到满足约束的分配方案")
    
    return alloc
import math

import math
'''
def greedy_tile_allocation(group_layers, compute_workloads, min_tiles, max_total_tiles, 
                           head_num=1, transformer=False, multi_batch=False):
    """
    改进版：严格保证 calculate_total_tiles(alloc) == max_total_tiles，
    避免 Transformer 下 (A/Z0) 推爆。
    """
    # 1. 基础映射
    compute_dict = {layer: compute for layer, compute in zip(group_layers, compute_workloads)}
    min_tile_dict = {layer: min_t for layer, min_t in zip(group_layers, min_tiles)}
    alloc = min_tile_dict.copy()

    # 2. 总Tile数计算
    def calculate_total_tiles(alloc_dict):
        if transformer:
            total_normal = sum(alloc_dict.values())
            tile_A = alloc_dict.get('A', 0)
            tile_Z0 = alloc_dict.get('Z0', 0)
            if multi_batch:
                return total_normal + 2 * (head_num - 1) * (tile_A + tile_Z0)
            else:
                return total_normal + (head_num - 1) * (tile_A + tile_Z0)
        else:
            return sum(alloc_dict.values())

    current_total = calculate_total_tiles(alloc)
    if current_total > max_total_tiles:
        raise ValueError("无法满足最小需求，初始分配已超过最大Tile限制")

    remaining_tiles = max_total_tiles - current_total
    total_compute = sum(compute_dict.values())
    if total_compute <= 0 or remaining_tiles <= 0:
        return alloc

    # 3. 按比例分配
    proportional_extra = {layer: remaining_tiles * compute_dict[layer] / total_compute
                          for layer in group_layers}
    extra_alloc = {layer: math.floor(proportional_extra[layer]) for layer in group_layers}

    # 4. 先应用整数分配
    for layer in group_layers:
        alloc[layer] += extra_alloc[layer]

    # 5. 动态补齐直到 calculate_total_tiles == max_total_tiles
    while calculate_total_tiles(alloc) < max_total_tiles:
        # 找到最合适的层（优先非 A/Z0，计算量大的）
        candidates = [l for l in group_layers if l not in ['A', 'Z0']]
        if not candidates:
            candidates = group_layers  # 如果只有 A/Z0 也只能用它们

        # 选计算量最大的
        target = max(candidates, key=lambda x: compute_dict[x])
        alloc[target] += 1

        # 安全检查，防止死循环
        if calculate_total_tiles(alloc) > max_total_tiles:
            # 回退一步
            alloc[target] -= 1
            break

    return alloc
'''
def compute_time_statistics_cnn(layers_cal, layers_weight, weight_is_duration=True, debug=False):
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
            if isinstance(x, (tuple, list)) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
                s = float(x[0]); second = float(x[1])
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
        print("weight_intervals (count={}):".format(len(weight_intervals)), weight_intervals)

    if not cal_intervals and not weight_intervals:
        return 0.0, 0.0, 0.0

    # sweep-line events
    events = []
    for s, e in cal_intervals:
        events.append((s, "cal", +1)); events.append((e, "cal", -1))
    for s, e in weight_intervals:
        events.append((s, "weight", +1)); events.append((e, "weight", -1))

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

    # cal 区间：本来就是 (start, end)
    for block in layers_cal:
        for batch in block:
            for item in batch:
                if not item or len(item) != 2:
                    continue
                s, e = item
                events.append((s, "cal", +1))
                events.append((e, "cal", -1))

    # weight 区间：是 (start, duration)，要转成 (start, start+duration)
    for block in layers_weight:
        for batch in block:
            for item in batch:
                if not item or len(item) != 2:
                    continue
                s, dur = item
                e = s + dur
                events.append((s, "weight", +1))
                events.append((e, "weight", -1))

    # 按时间排序
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
    直接接收原始计算时间和bubble时间参数，计算每个层的bubble时间占比
    
    :param layers_cal: 计算时间段列表，每个元素为 (start, end) 元组
    :param layers_bubble: bubble时间段列表，每个元素为字典（键为起始时间，值为持续时间）
    :return: 每个层的bubble时间百分比列表
    """
    percent = []
    # 循环处理每一层的数据
    for cal_time, bubble_dict in zip(layers_cal, layers_bubble):
        # 计算当前层的总计算时间（end - start）
        cal_duration = cal_time[1] - cal_time[0]
        
        # 处理bubble时间为空的情况
        if not bubble_dict:
            percent.append(0)
            continue
        
        # 计算当前层的总bubble时间（求和字典中的所有值）
        total_bubble = sum(bubble_dict.values())
        
        # 计算百分比并添加到结果
        percent.append((total_bubble / cal_duration) * 100)
    
    return percent
def capacity_aware_transformer_grouping(weights, layer_names, arch_size, ops_per_block):
    """
    基于真实算子名称的 Transformer 动态三级容量感知分组
    
    参数:
        weights: list, 每个算子的 workload 权重大小
        layer_names: list, 对应的算子名称列表 (如 process.layers)
        arch_size: float/int, 硬件级可用容量 (Arch Size)
        ops_per_block: int, 每个 Block 包含的算子数 (如你的截图里似乎是 8 个)
    返回:
        groups: list of lists, 返回分组的 index 列表
    """
    
    def get_subblock_type(name):
        """根据算子名称，动态判断它属于 Transformer 的哪个宏模块 (Sub-block)"""
        name_up = name.upper()
        # 如果名字里包含 Q, K, V，归为 QKV 组
        if any(k in name_up for k in ['Q', 'K', 'V']):
            return 'QKV_BRANCH'
        # 如果名字里包含 FFN，归为 FFN 组
        elif 'FFN' in name_up:
            return 'FFN_BRANCH'
        # 其他的 (比如 A, Z0, Z1, Proj, Out) 归为 Attention/Proj 组
        else:
            return 'ATTN_PROJ_BRANCH'

    groups = []
    num_ops = len(weights)
    op_idx = 0
    
    print(f"\n🚀 开始执行基于算子名称的动态容量感知分组...")
    print(f"   - 硬件级单级容量限制: {arch_size:,.0f} bit")
    print(f"   - 待处理算子总数: {num_ops} (共 {num_ops // ops_per_block} 个 Blocks)")

    while op_idx < num_ops:
        # 1. 定位当前算子属于哪个 Block
        block_idx = op_idx // ops_per_block
        block_start = block_idx * ops_per_block
        block_end = min(block_start + ops_per_block, num_ops)
        
        # ==============================================================
        # Level 1: Block-Level Parallelism (尝试把一整个 Block 吞掉)
        # ==============================================================
        block_weights_sum = sum(weights[block_start:block_end])
        
        if op_idx == block_start and block_weights_sum <= arch_size:
            groups.append(list(range(block_start, block_end)))
            print(f"   ✅ [Level 1] Block {block_idx} 容量充裕，整体成组 -> Ops [{block_start}:{block_end-1}]")
            op_idx = block_end
            continue
            
        # ==============================================================
        # Level 2: Sub-Block-Level Parallelism (按算子名称动态分块)
        # ==============================================================
        # 获取当前算子的类型 (比如它是 QKV 还是 FFN)
        current_sb_type = get_subblock_type(layer_names[op_idx])
        
        # 往后寻找所有与当前算子类型相同、且连续的算子
        sb_indices = [op_idx]
        next_idx = op_idx + 1
        
        while next_idx < block_end and get_subblock_type(layer_names[next_idx]) == current_sb_type:
            sb_indices.append(next_idx)
            next_idx += 1
            
        # 计算这个动态提取出来的 Sub-block 的总权重
        sb_weights_sum = sum(weights[i] for i in sb_indices)
        
        if sb_weights_sum <= arch_size:
            groups.append(sb_indices)
            # 打印的时候顺便把算子名字打出来，极其直观！
            sb_names = [layer_names[i].split('.')[0] for i in sb_indices] 
            print(f"   ⚠️ [Level 2] Block {block_idx} 降级为 {current_sb_type} 宏模块成组 -> {sb_names}")
            op_idx = next_idx
            continue
            
        # ==============================================================
        # Level 3: Fine-Grained Operator Parallelism (最差情况，孤立单算子)
        # ==============================================================
        single_weight = weights[op_idx]
        op_name = layer_names[op_idx].split('.')[0]
        
        if single_weight <= arch_size:
            groups.append([op_idx])
            print(f"   🔥 [Level 3] 容量严重受限，细粒度算子孤立成组 -> [{op_name}]")
        else:
            groups.append([op_idx]) 
            print(f"   🚨 [警告] 算子 [{op_name}] 权重 ({single_weight:,.0f}) 超出单级容量上限！后续需触发张量切片 (Tiling)！")
            
        op_idx += 1
        
    print(f"🎉 分组完成！{num_ops} 个离散算子已被智能压缩为 {len(groups)} 个 Pipeline Stages。")
    return groups