import math
from collections import defaultdict

class TileAllocator:
    def __init__(self, layers, compute_workloads, min_tiles_per_layer, max_tiles_per_group=16,
                 min_layers_per_group=1, max_layers_per_group=float('inf'),workload_shape=[]):
        """
        :param layers: 层标识列表
        :param compute_workloads: 每层的计算负载
        :param min_tiles_per_layer: 每层所需最小Tile数列表
        :param max_tiles_per_group: 每组最大可用Tile数
        :param min_layers_per_group: 每组最少层数
        :param max_layers_per_group: 每组最多层数
        """
        self.layers = layers
        self.compute_workloads = compute_workloads
        self.min_tiles = min_tiles_per_layer
        self.max_tiles = max_tiles_per_group
        self.min_layers = min_layers_per_group
        self.max_layers = max_layers_per_group
        self.n_layers = len(layers)
        self.all_allocations = []  # 存储所有可能的分配方案
        self.workload_shape = workload_shape


    def proportional_tile_allocation_group(self, group_layers, compute_workloads, min_tiles, max_total_tiles,
                                        head_num=1, transformer=False, multi_batch=False):
        """
        按组分配Tile：
        - 卷积层按合法tile投影
        - 全连接层直接贪心
        - 尽量让组内总和 = max_total_tiles
        """
        import math
        projected_alloc = {}

        # 单层组直接分配 max_total_tiles
        if len(group_layers) == 1:
            projected_alloc[group_layers[0]] = max_total_tiles
            return projected_alloc

        # 映射
        compute_dict = {layer: compute for layer, compute in zip(group_layers, compute_workloads)}
        min_tile_dict = {layer: min_t for layer, min_t in zip(group_layers, min_tiles)}

        # 卷积层空间信息
        layer_shapes = {}
        for i, layer in enumerate(self.layers):
            shape_info = self.workload_shape[i]
            if "P" in shape_info:
                layer_shapes[layer] = shape_info["P"]

        # 合法tile集合
        def legal_tiles(H):
            return sorted([d for d in range(1, H + 1) if H % d == 0])

        legal_tile_dict = {}
        for layer in group_layers:
            if layer in layer_shapes:
                H = layer_shapes[layer]
                legal_tile_dict[layer] = [t for t in legal_tiles(H) if t >= min_tile_dict[layer]]
            else:
                # 全连接层：合法tile至少min_tile
                legal_tile_dict[layer] = [min_tile_dict[layer]]

        # 总tile计算函数
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

        # 按比例分配最小tile + extra
        total_compute = sum(compute_dict.values())
        if total_compute <= 0:
            for layer in group_layers:
                projected_alloc[layer] = min_tile_dict[layer]
            return projected_alloc

        proportional_extra = {layer: (max_total_tiles - sum(min_tile_dict.values())) * compute_dict[layer] / total_compute
                            for layer in group_layers}
        # 初步分配
        alloc = {layer: min_tile_dict[layer] + math.floor(proportional_extra[layer]) for layer in group_layers}

        # 投影卷积层到合法tile
        for layer in group_layers:
            if layer in layer_shapes:
                legal_set = legal_tile_dict[layer]
                candidates = [t for t in legal_set if t <= alloc[layer]]
                alloc[layer] = max(candidates + [min_tile_dict[layer]]) if candidates else min_tile_dict[layer]

        # 剩余tile
        leftover = max_total_tiles - sum(alloc.values())

        # 升级剩余tile：按当前分配最小层优先，直到耗尽
        while leftover > 0:
            # 按当前分配从小到大排序
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
                # 如果没有合法提升，就按1递增剩余tile
                for layer in sorted_layers:
                    if leftover <= 0:
                        break
                    alloc[layer] += 1
                    leftover -= 1

        # 最终分配
        projected_alloc = alloc

        if calculate_total_tiles(projected_alloc) > max_total_tiles:
            raise ValueError("形态约束下无法满足总tile上限")

        return projected_alloc





    def explore_allocations(self):
        """探索所有可能的分配方案（带层数约束）"""
        # 从第一层开始探索
        self._dfs(0, self.max_tiles, [], [])
        return self.all_allocations
    def explore_allocations(self, max_compute_ratio=4, min_compute_ratio=0.3):
        """
        带计算量感知剪枝的分配探索
        :param max_compute_ratio: 允许的最大计算量超载系数（1.5表示单组最多承担均值的 1.5 倍）
        :param min_compute_ratio: 允许的最小计算量下限（0.3表示太闲了不许切分）
        """
        self.all_allocations = []
        
        # 1. 算出理想状态下的单组计算量
        total_compute = sum(self.compute_workloads)
        # 物理上最少需要的组数
        min_required_groups = max(1, math.ceil(sum(self.min_tiles) / self.max_tiles))
        # 理想均摊计算量
        self.ideal_compute = total_compute / min_required_groups
        
        # 计算量上下界阈值
        self.max_compute_threshold = self.ideal_compute * max_compute_ratio
        self.min_compute_threshold = self.ideal_compute * min_compute_ratio

        # 开始 DFS
        self._dfs(0, self.max_tiles, [], [])
        return self.all_allocations

    def _dfs(self, layer_idx, remaining_tiles, current_group, allocation):
        """
        递归搜索分配方案（带层数约束）
        :param layer_idx: 当前处理的层索引
        :param remaining_tiles: 当前组剩余可用Tile数
        :param current_group: 当前组内的层列表
        :param allocation: 当前完整的分配方案
        """
        # 剪枝操作：如果这是第一组且层数少于4，提前终止
        if not allocation and current_group and len(current_group) < 3 and layer_idx < self.n_layers:
            # 如果当前是第一组且层数少于4，且还有剩余层可以添加，但当前无法添加更多层
            if not (len(current_group) < self.max_layers and self.min_tiles[layer_idx] <= remaining_tiles):
                # 如果无法添加更多层，则剪枝
                return
        
        # 终止条件：所有层已分配
        if layer_idx >= self.n_layers:
            # 如果有当前组，检查层数约束
            if current_group:
                group_size = len(current_group)
                # 检查层数约束：必须满足 min_layers <= group_size <= max_layers
                # 额外剪枝：如果是第一组，必须至少4层
                if group_size >= self.min_layers and group_size <= self.max_layers:
                    # 如果是第一组且少于4层，跳过
                    if not allocation and group_size < 4:
                        return
                    
                    try:
                        compute = []
                        min_tiles=[]
                        for i in range(group_size):
                            compute.append(self.compute_workloads[self.layers.index(current_group[i])])
                            min_tiles.append(self.min_tiles[self.layers.index(current_group[i])])
                        #min_tiles = [self.min_tiles[self.layers.index(layer)] for layer in current_group]
                        group_allocation = self.proportional_tile_allocation_group(current_group, compute,self.min_tiles,self.max_tiles)
                        allocation.append((current_group.copy(), group_allocation))
                        # 保存当前方案（复制以防修改）
                        self.all_allocations.append(allocation.copy())
                        # 回溯：移除当前组
                        allocation.pop()
                    except ValueError:
                        pass  # 跳过无效分组
            return
        
        layer_name = self.layers[layer_idx]
        min_req = self.min_tiles[layer_idx]
        
        # 检查当前组是否可以继续添加层（不超过最大层数限制）
        can_add_to_current = (len(current_group) < self.max_layers and min_req <= remaining_tiles)
        
        # 情况1：尝试将当前层加入现有组
        if can_add_to_current:
            # 添加当前层到组
            current_group.append(layer_name)
            
            # 递归处理下一层
            self._dfs(
                layer_idx + 1, 
                remaining_tiles - min_req,  # 使用最小需求作为保守估计
                current_group, 
                allocation
            )
            
            # 回溯：移除当前层
            current_group.pop()
        
        # 情况2：尝试将当前层放入新组
        # 只有在当前组非空时才需要尝试（避免重复方案）
        if current_group:
            group_size = len(current_group)
            # 检查层数约束：必须满足 min_layers <= group_size <= max_layers
            if group_size >= self.min_layers and group_size <= self.max_layers:
                # 额外剪枝：如果是第一组，必须至少4层
                if not allocation and group_size < 4:
                    # 如果是第一组且少于4层，跳过
                    return
                
                try:
                    # 对当前组应用贪心分配算法
                    compute = []
                    min_tiles=[]
                    for i in range(group_size):
                        compute.append(self.compute_workloads[self.layers.index(current_group[i])])
                        min_tiles.append(self.min_tiles[self.layers.index(current_group[i])])
                    group_allocation = self.proportional_tile_allocation_group(current_group, compute,min_tiles,self.max_tiles)
                    allocation.append((current_group.copy(), group_allocation))
                    
                    # 创建新组并添加当前层
                    new_group = [layer_name]
                    
                    # 递归处理下一层（新组）
                    self._dfs(
                        layer_idx + 1, 
                        self.max_tiles - min_req, 
                        new_group, 
                        allocation
                    )
                    
                    # 回溯：移除当前组
                    allocation.pop()
                except ValueError:
                    pass  # 跳过无效分组
        
        # 情况3：当前组为空时直接创建新组
        else:
            # 创建新组并添加当前层
            new_group = [layer_name]
            
            # 递归处理下一层
            self._dfs(
                layer_idx + 1, 
                self.max_tiles - min_req, 
                new_group, 
                allocation
            )
    def _dfs(self, layer_idx, remaining_tiles, current_group, allocation):
        # 终止条件：所有层已分配
        if layer_idx >= self.n_layers:
            if current_group:
                group_size = len(current_group)
                if self.min_layers <= group_size <= self.max_layers:
                    try:
                        compute = [self.compute_workloads[self.layers.index(l)] for l in current_group]
                        min_tiles = [self.min_tiles[self.layers.index(l)] for l in current_group]
                        group_allocation = self.proportional_tile_allocation_group(current_group, compute, min_tiles, self.max_tiles)
                        allocation.append((current_group.copy(), group_allocation))
                        
                        self.all_allocations.append(allocation.copy())
                        
                        allocation.pop()
                    except ValueError:
                        pass
            return

        # 当前层信息
        layer_name = self.layers[layer_idx]
        min_req = self.min_tiles[layer_idx]
        next_compute = self.compute_workloads[layer_idx]
        
        # 统计当前组状态
        if current_group:
            current_compute = sum([self.compute_workloads[self.layers.index(l)] for l in current_group])
            current_min_sum = sum([self.min_tiles[self.layers.index(l)] for l in current_group])
        else:
            current_compute = 0
            current_min_sum = 0
            
        group_size = len(current_group)
        
        # 最基本的物理墙：下一层能不能塞下
        physically_can_add = (group_size < self.max_layers) and (current_min_sum + min_req <= self.max_tiles)
        
        can_add_to_current = physically_can_add
        can_create_new_group = False
        
        if current_group:
            if group_size >= self.min_layers:
                can_create_new_group = True
                
            # === 【核心剪枝：计算量过载保护】 ===
            # 即使 min_tiles 极小，物理上能塞得下
            # 但是如果塞进去，计算量超过了设定的最大阈值，强行禁止塞入！避免产生“累死”的组。
            if physically_can_add and (current_compute + next_compute > self.max_compute_threshold):
                can_add_to_current = False
                
            # === 【核心剪枝：计算量下限保护 (防极度碎片化)】 ===
            # 如果当前组吃掉的计算量还极小（比如才占了均值的 30%）
            # 并且物理上明明还能接着塞，那就强行不许换新组，逼迫它继续吃层数。
            if can_create_new_group and physically_can_add and (current_compute < self.min_compute_threshold):
                can_create_new_group = False
        else:
            # 当前组为空，无条件加入首层
            can_add_to_current = True
            can_create_new_group = False

        # --- 分支 A：将当前层加入现有组 ---
        if can_add_to_current:
            current_group.append(layer_name)
            self._dfs(layer_idx + 1, remaining_tiles - min_req, current_group, allocation)
            current_group.pop()  # 回溯
            
        # --- 分支 B：结束当前组，将当前层作为新组的开头 ---
        if can_create_new_group:
            try:
                # 结算当前组
                compute = [self.compute_workloads[self.layers.index(l)] for l in current_group]
                min_tiles = [self.min_tiles[self.layers.index(l)] for l in current_group]
                group_allocation = self.proportional_tile_allocation_group(current_group, compute, min_tiles, self.max_tiles)
                
                allocation.append((current_group.copy(), group_allocation))
                
                # 新起一组
                new_group = [layer_name]
                self._dfs(layer_idx + 1, self.max_tiles - min_req, new_group, allocation)
                
                allocation.pop()  # 回溯
            except ValueError:
                pass
    def print_allocations(self, allocations):
        """打印所有分配方案"""
        print(f"总方案数: {len(allocations)}")
        for i, allocation in enumerate(allocations):
            print(f"\n方案 #{i+1}:")
            total_groups = len(allocation)
            total_tiles_used = 0
            
            for group_idx, (group_layers, layer_allocations) in enumerate(allocation):
                group_size = len(group_layers)
                group_tiles = sum(layer_allocations.values())
                total_tiles_used += group_tiles
                print(f"  组 {group_idx+1}/{total_groups} (层数: {group_size}, 总Tile: {group_tiles}/{self.max_tiles}):")
                
                for layer in group_layers:
                    min_req = self.min_tiles[self.layers.index(layer)]
                    alloc = layer_allocations[layer]
                    compute = self.compute_workloads[self.layers.index(layer)]
                    print(f"    层 {layer}: 分配 {alloc} (最小需求 {min_req}, 计算量 {compute})")
            
            print(f"  总Tile使用: {total_tiles_used}/{total_groups * self.max_tiles}")

    def find_best_allocation(self, metric="min_groups"):
        """根据特定指标找到最佳分配方案"""
        if not self.all_allocations:
            self.explore_allocations()
        
        if metric == "min_groups":
            # 找到分组最少的方案
            return min(self.all_allocations, key=len)
        elif metric == "max_utilization":
            # 找到利用率最高的方案
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
            # 找到组内最大Tile需求最小的方案
            best_allocation = None
            min_max_tiles = float('inf')
            
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
            raise ValueError(f"未知指标: {metric}")

    def analyze_allocations(self):
        """分析所有分配方案，找出常见模式"""
        if not self.all_allocations:
            self.explore_allocations()
        
        # 统计层分组频率
        group_freq = defaultdict(int)
        for allocation in self.all_allocations:
            group_key = tuple(tuple(group) for group, _ in allocation)
            group_freq[group_key] += 1
        
        # 找出最常见的分组方式
        most_common = max(group_freq.items(), key=lambda x: x[1]) if group_freq else (None, 0)
        
        # 统计每层在不同组中的位置
        layer_positions = defaultdict(list)
        layer_groups = defaultdict(list)
        for allocation in self.all_allocations:
            for group_idx, (group, _) in enumerate(allocation):
                for layer in group:
                    layer_positions[layer].append(group_idx)
                    layer_groups[layer].append(group)
        
        # 计算每层的平均位置
        avg_positions = {}
        for layer, positions in layer_positions.items():
            avg_positions[layer] = sum(positions) / len(positions)
        
        # 找出每层最常出现的组
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
            "most_common_groups": most_common_groups
        }
# 您的数据
'''
layers = ['00', '01', '02', '03', '04', '05', '06', '07', '08','09', '10', '11', '12', '13', '14', '15', '16', '17','18', '19', '20']
compute_workloads = [118013952, 115605504, 115605504, 115605504, 115605504, 57802752, 115605504, 6422528, 115605504, 115605504, 57802752, 115605504, 6422528, 115605504, 115605504, 57802752, 115605504, 6422528, 115605504, 115605504, 512000]
min_tiles = [4, 4, 2, 4, 2, 2, 4, 1, 2, 4, 3, 4, 1, 4, 4, 6, 12, 1, 15, 14, 3]
max_tiles_per_group = 32

# 创建分配系统（添加层数约束）
allocator = TileAllocator(
    layers=layers,
    compute_workloads=compute_workloads,
    min_tiles_per_layer=min_tiles,
    max_tiles_per_group=max_tiles_per_group,
    min_layers_per_group=2,  # 每组最少4层
    max_layers_per_group=10  # 每组最多10层
)

# 探索所有可能的分配方案
all_allocations = allocator.explore_allocations()

# 分析结果
analysis = allocator.analyze_allocations()


# 打印最常见的分组模式
if analysis['most_common_grouping']:
    print(f"\n最常见分组 (频率: {analysis['frequency']}):")
    for i, group in enumerate(analysis['most_common_grouping']):
        print(f"  组 {i+1}: {', '.join(group)}")

# 打印每层最常出现的组
print("\n每层最常出现的组:")
for layer, (group, freq) in analysis['most_common_groups'].items():
    print(f"  层 {layer}: {freq}次出现在组 {group}")

# 找到并打印最佳分配方案（最小组数）
best_allocation = allocator.find_best_allocation(metric="min_max_tiles")
print("\n最小组数方案:")
allocator.print_allocations([best_allocation])
print(all_allocations[:100])
print(f"总方案数: {analysis['total_allocations']}")
'''