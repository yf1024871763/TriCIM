import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import re
import matplotlib.patches as mpatches
from matplotlib import gridspec
import seaborn as sns
def plot_bubble(save_path, layers_cal, layers_bubble, labels=None, actual_time=0):
    """
    可视化计算时间和bubble时间
    :param layers_cal: List of tuples [(start1, end1), (start2, end2), ...] 每个层的计算时间段
    :param layers_bubble: List of dicts 每个层的bubble时间段
    :param actual_time: 系统总运行时间
    """
    # 确保导入必要的库
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    no_labels = False
    # 数据校验
    assert len(layers_cal) == len(layers_bubble), "计算时间和bubble时间层数必须一致"
    if labels is None:
        labels = [str(i+1) for i in range(len(layers_cal))]
        no_labels = True

    # 预处理数据
    processed_data = []
    for cal_times, bubble_dict in zip(layers_cal, layers_bubble):
        # 转换计算时间
        cal_intervals = [(cal_times[0], cal_times[1] - cal_times[0])]
        
        # 转换bubble时间
        bubble_intervals = sorted(bubble_dict.items())
        
        # 合并到同一层数据
        processed_data.append({
            'cal': cal_intervals,
            'bubble': bubble_intervals
        })
    
    percent = []
    for data in processed_data:
        if data['bubble'] == []:
            percent.append(0)
            continue
        percent.append((sum(y for _, y in data['bubble']) / data['cal'][0][1]) * 100)

    # 创建图表
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # 定义颜色方案 - 使用十六进制代码更可靠
    CAL_COLOR = '#6495ED'  # 明确的蓝色
    BUBBLE_COLOR = 'lightgray'  # 明确的橙色
    BORDER_COLOR = 'black'  # 边框颜色
    
    # 逐层绘制
    for layer_idx, layer_data in enumerate(reversed(processed_data)):
        y_center = layer_idx + 1  # 层在y轴的位置
        
        # 绘制计算时间（带黑边）- 使用color参数而非facecolors
        if layer_data['cal']:
            ax.broken_barh(
                layer_data['cal'],
                (y_center - 0.45, 0.9),
                color=CAL_COLOR,  # 修正为color参数
                edgecolor=BORDER_COLOR,
                linewidth=0.5,
                label='Compute time' if layer_idx == 0 else None
            )
        
        # 绘制Bubble时间（带黑边）
        if layer_data['bubble']:
            ax.broken_barh(
                layer_data['bubble'],
                (y_center - 0.45, 0.9),
                color=BUBBLE_COLOR,  # 修正为color参数
                edgecolor=BORDER_COLOR,
                linewidth=0.2,
                hatch='//',
                label='Bubble time' if layer_idx == 0 else None
            )
    
    # 自动计算时间范围
    all_points = []
    for layer in processed_data:
        all_points += [t for t, _ in layer['cal'] + layer['bubble']]
        all_points += [t + d for t, d in layer['cal'] + layer['bubble']]
    min_t = min(all_points) if all_points else 0
    max_t = max(all_points) if all_points else actual_time
    
    # 坐标轴设置
    ax.set_yticks(np.arange(1, len(processed_data) + 1))
    
    ax.set_yticklabels(reversed(labels), fontsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.set_xlim(min_t * 0.95, max(max_t, actual_time) * 1.05)
    ax.set_xlabel("Timestamp", fontsize=20)
    ax.set_title(f"Computation and bubble time (Time Cost: {actual_time})", fontsize=25)
    if no_labels:
            ax.set_yticks([])  # 清空y轴刻度，不显示任何数字
    # 图例设置 - 确保与实际颜色匹配
    legend_elements = [
        Patch(facecolor=CAL_COLOR, edgecolor=BORDER_COLOR, label='Compute time'),
        Patch(facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR, hatch='//', label='Bubble time')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=20)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)  # 简化目录创建
    plt.savefig(f"{save_path}/Bubble.png", dpi=400, bbox_inches='tight')
    plt.savefig(f"{save_path}/Bubble.svg", dpi=600,format='svg') 
    # 绘制百分比图
    plt.figure(figsize=(20, 9))
    bars = plt.bar(labels, percent, 0.45, color=CAL_COLOR)  # 使用相同的蓝色
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=15)
    plt.ylabel('Percent (%)', fontsize=20)  # 修正为百分号
    plt.title(f'Bubble percentage with pipeline', fontsize=25) 
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=14)  # 添加百分号
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/Bubble_percentage.png", dpi=400, bbox_inches='tight')

    plt.show()
def plot_compute_weight(save_path, layers_cal, layers_weight, labels=None, actual_time=0):
    """
    可视化计算时间和权重写入时间
    :param layers_cal: List of tuples [(start, end), ...] 每个层的计算时间段
    :param layers_weight: List of list of tuples [[(start, duration), ...], ...] 每层的权重写入时间段
    :param labels: 每层的标签
    :param actual_time: 系统总运行时间
    """


    assert len(layers_cal) == len(layers_weight), "计算时间和权重更新时间层数必须一致"
    no_labels = False
    if labels is None:
        labels = [str(i+1) for i in range(len(layers_cal))]
        no_labels = True

    processed_data = []
    for cal_times, weight_intervals in zip(layers_cal, layers_weight):
        cal_intervals = [(cal_times[0], cal_times[1] - cal_times[0])]
        processed_data.append({
            'cal': cal_intervals,
            'weight': weight_intervals
        })

    # 颜色
    CAL_COLOR = '#6495ED'   # 蓝色
    WEIGHT_COLOR = '#66CDAA'  # 薄荷绿
    BORDER_COLOR = 'black'

    fig, ax = plt.subplots(figsize=(20, 10))

    for layer_idx, layer_data in enumerate(reversed(processed_data)):
        y_center = layer_idx + 1

        # 计算时间
        ax.broken_barh(
            layer_data['cal'],
            (y_center - 0.45, 0.9),
            color=CAL_COLOR,
            edgecolor=BORDER_COLOR,
            linewidth=0.5,
            label='Compute time' if layer_idx == 0 else None
        )

        # 权重写入时间
        if layer_data['weight']:
            ax.broken_barh(
                layer_data['weight'],
                (y_center - 0.45, 0.9),
                color=WEIGHT_COLOR,
                edgecolor=BORDER_COLOR,
                linewidth=0.5,
                label='Weight update' if layer_idx == 0 else None
            )

    # 时间范围
    all_points = []
    for layer in processed_data:
        all_points += [t for t, _ in layer['cal'] + layer['weight']]
        all_points += [t + d for t, d in layer['cal'] + layer['weight']]
    min_t = min(all_points) if all_points else 0
    max_t = max(all_points) if all_points else actual_time

    # 坐标轴
    ax.set_yticks(np.arange(1, len(processed_data) + 1))
    ax.set_yticklabels(reversed(labels), fontsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.set_xlim(min_t * 0.95, max(max_t, actual_time) * 1.05)
    ax.set_xlabel("Timestamp", fontsize=20)
    ax.set_title(f"Computation and Weight Update (Time Cost: {actual_time})", fontsize=25)
    if no_labels:
        ax.set_yticks([])

    # 图例
    legend_elements = [
        Patch(facecolor=CAL_COLOR, edgecolor=BORDER_COLOR, label='Compute time'),
        Patch(facecolor=WEIGHT_COLOR, edgecolor=BORDER_COLOR, label='Weight update')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=20)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/Compute_Weight.png", dpi=400, bbox_inches='tight')
def plot_combined_timelines(save_path, layers_cal, layers_bubble, layers_weight, labels=None, actual_time=0):
    """
    同时可视化计算时间、bubble时间和权重更新时间
    :param save_path: 图像保存路径
    :param layers_cal: List of tuples [(start1, end1), (start2, end2), ...] 每个层的计算时间段
    :param layers_bubble: List of dicts 每个层的bubble时间段
    :param layers_weight: List of list of tuples [[(start, duration), ...], ...] 每层的权重写入时间段
    :param labels: 每层的标签
    :param actual_time: 系统总运行时间
    """
    # 确保导入必要的库
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    # 数据校验
    assert len(layers_cal) == len(layers_bubble) == len(layers_weight), \
        "计算时间、bubble时间和权重更新时间的层数必须一致"
    
    no_labels = False
    if labels is None:
        labels = [str(i+1) for i in range(len(layers_cal))]
        no_labels = True

    # 预处理数据
    processed_data = []
    for cal_times, bubble_dict, weight_intervals in zip(layers_cal, layers_bubble, layers_weight):
        # 转换计算时间
        cal_duration = cal_times[1] - cal_times[0]
        cal_intervals = [(cal_times[0], cal_duration)]
        
        # 转换bubble时间
        bubble_intervals = sorted(bubble_dict.items())
        
        # 合并到同一层数据
        processed_data.append({
            'cal': cal_intervals,
            'bubble': bubble_intervals,
            'weight': weight_intervals,
            'cal_duration': cal_duration
        })
    
    # 计算bubble时间占比
    percent = []
    for data in processed_data:
        if data['bubble'] == []:
            percent.append(0)
            continue
        percent.append((sum(y for _, y in data['bubble']) / data['cal_duration']) * 100)

    # 创建时间线图表
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # 定义颜色方案
    CAL_COLOR = '#6495ED'      # 蓝色 - 计算时间
    BUBBLE_COLOR = '#FFA500'   # 橙色 - Bubble时间
    WEIGHT_COLOR = '#D63344'   # 薄荷绿 - 权重更新时间
    BORDER_COLOR = 'black'     # 边框颜色
    
    # 逐层绘制
    for layer_idx, layer_data in enumerate(reversed(processed_data)):
        y_center = layer_idx + 1  # 层在y轴的位置
        
        # 绘制计算时间
        if layer_data['cal']:
            ax.broken_barh(
                layer_data['cal'],
                (y_center - 0.45, 0.9),
                color=CAL_COLOR,
                edgecolor=BORDER_COLOR,
                linewidth=0.2,
                label='Compute time' if layer_idx == 0 else None
            )
        
        # 绘制Bubble时间
        if layer_data['bubble']:
            ax.broken_barh(
                layer_data['bubble'],
                (y_center - 0.45, 0.9),
                color=BUBBLE_COLOR,
                edgecolor=BORDER_COLOR,
                linewidth=0.1,
                hatch='//',
                label='Bubble time' if layer_idx == 0 else None
            )
        
        # 绘制权重更新时间
        if layer_data['weight']:
            ax.broken_barh(
                layer_data['weight'],
                (y_center - 0.45, 0.9),
                color=WEIGHT_COLOR,
                edgecolor=BORDER_COLOR,
                linewidth=0.2,
                label='Weight update' if layer_idx == 0 else None
            )
    
    # 自动计算时间范围
    all_points = []
    for layer in processed_data:
        all_intervals = layer['cal'] + layer['bubble'] + layer['weight']
        all_points += [t for t, _ in all_intervals]
        all_points += [t + d for t, d in all_intervals]
    
    min_t = min(all_points) if all_points else 0
    max_t = max(all_points) if all_points else actual_time
    
    # 坐标轴设置
    ax.set_yticks(np.arange(1, len(processed_data) + 1))
    ax.set_yticklabels(reversed(labels), fontsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.set_xlim(min_t * 0.95, max(max_t, actual_time) * 1.05)
    ax.set_xlabel("Timestamp", fontsize=20)
    ax.set_title(f"Computation, Bubble and Weight Update Times (Time Cost: {actual_time})", fontsize=25)
    
    if no_labels:
        ax.set_yticks([])  # 不显示标签
    
    # 图例设置
    legend_elements = [
        Patch(facecolor=CAL_COLOR, edgecolor=BORDER_COLOR, label='Compute time'),
        Patch(facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR, hatch='//', label='Bubble time'),
        Patch(facecolor=WEIGHT_COLOR, edgecolor=BORDER_COLOR, label='Weight update')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=20)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/Combined_Timelines.png", dpi=400, bbox_inches='tight')
    
def plot_combined_timelines_block_batch(save_path, layers_cal, layers_bubble, layers_weight, layer_names =None,block_labels=None, actual_time=0):
    """
    可视化多个block的计算时间、bubble时间和权重更新时间
    纵坐标是block，每个block内从上到下是不同的layer（按正确顺序）
    
    :param save_path: 图像保存路径
    :param layers_cal: 三维列表 [block][batch][layer] 每个层的计算时间段
    :param layers_bubble: 三维列表 [block][batch][layer] 每个层的bubble时间段
    :param layers_weight: 三维列表 [block][batch][layer] 每层的权重写入时间段
    :param block_labels: 每个block的标签
    :param actual_time: 系统总运行时间
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    # 获取维度信息
    num_blocks = len(layers_cal)
    num_batches = len(layers_cal[0]) if num_blocks > 0 else 0
    num_layers = len(layers_cal[0][0]) if num_batches > 0 else 0
    
    
    # 如果没有提供block标签，创建默认标签
    if block_labels is None:
        block_labels = [f"Block {i}" for i in range(num_blocks)]
    
    # 创建时间线图表
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # 定义颜色方案
    CAL_COLOR = '#6495ED'      # 蓝色 - 计算时间
    BUBBLE_COLOR = 'lightgrey'   # 橙色 - Bubble时间
    WEIGHT_COLOR = '#D63344'   # 薄荷绿 - 权重更新时间
    BORDER_COLOR = 'black'     # 边框颜色
    
    # 定义层名称（按正确顺序）
    if layer_names==None:
        layer_names = ["Q", "K", "V", "A", "Z0", "Z1", "FFN1", "FFN2"]
    
    # 收集所有时间点用于设置x轴范围
    all_points = []
    
    # 计算每个block的高度和每个layer在block内的位置
    block_height = 1.0
    layer_height = block_height / (num_layers)  # 留出一些边距
    
    # 循环所有block
    for block_idx in range(num_blocks):
        y_center = num_blocks - block_idx  # 从顶部开始，block 0在顶部
        
        # 循环当前block的所有layer（按正确顺序）
        for layer_idx in range(num_layers):
            # 计算当前layer在block内的y位置（从上到下）
            layer_y = y_center - (layer_idx + 1) * layer_height
            
            # 循环当前block的所有batch
            for batch_idx in range(num_batches):
                # =========================
                # 计算时间
                # =========================
                if (block_idx < len(layers_cal) and 
                    batch_idx < len(layers_cal[block_idx]) and 
                    layer_idx < len(layers_cal[block_idx][batch_idx])):
                    
                    cal_data = layers_cal[block_idx][batch_idx][layer_idx]
                    if cal_data and isinstance(cal_data, tuple) and len(cal_data) == 2:
                        start, end = cal_data
                        duration = end - start
                        ax.broken_barh(
                            [(start, duration)],
                            (layer_y - layer_height/2, layer_height),
                            facecolor=CAL_COLOR,
                            edgecolor=BORDER_COLOR,
                            linewidth=0.5
                        )
                        all_points.extend([start, end])
                
                # =========================
                # Bubble 时间
                # =========================
                if (block_idx < len(layers_bubble) and 
                    batch_idx < len(layers_bubble[block_idx]) and 
                    layer_idx < len(layers_bubble[block_idx][batch_idx])):
                    
                    bubble_data = layers_bubble[block_idx][batch_idx][layer_idx]
                    
                    if bubble_data:  # 有数据才画
                        if isinstance(bubble_data, tuple) and len(bubble_data) == 2:
                            start, end = bubble_data
                            duration = end - start
                            ax.broken_barh([(start,duration)], (layer_y-layer_height/2, layer_height),
                                            facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR, linewidth=0.2, hatch='//', alpha=0.4)
                            all_points.extend([start,end])
                        elif isinstance(bubble_data, dict):
                            if 'start' in bubble_data and 'end' in bubble_data:
                                start, end = bubble_data['start'], bubble_data['end']
                                duration = end - start
                                ax.broken_barh([(start,duration)], (layer_y-layer_height/2, layer_height),
                                                facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR, linewidth=0.2, hatch='//', alpha=0.4)
                                all_points.extend([start,end])
                            else:
                                for start,duration in bubble_data.items():
                                    ax.broken_barh([(start,duration)], (layer_y-layer_height/2, layer_height),
                                                    facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR, linewidth=0.2, hatch='//', alpha=0.4)
                                    all_points.extend([start, start+duration])
                        
                
                # =========================
                # Weight update
                # =========================
                if (block_idx < len(layers_weight) and 
                    batch_idx < len(layers_weight[block_idx]) and 
                    layer_idx < len(layers_weight[block_idx][batch_idx])):
                    
                    weight_data = layers_weight[block_idx][batch_idx][layer_idx]
                    if weight_data and isinstance(weight_data, tuple) and len(weight_data) == 2:
                        start, duration = weight_data
                        ax.broken_barh(
                            [(start, duration)],
                            (layer_y - layer_height/2, layer_height),
                            facecolor=WEIGHT_COLOR,
                            edgecolor=BORDER_COLOR,
                            linewidth=0.5
                        )
                        all_points.extend([start, start + duration])
            

    
    # 设置y轴标签（block标签）
    ax.set_yticks(range(1, num_blocks + 1))
    ax.set_yticklabels(reversed(block_labels), fontsize=14)
    ax.set_ylabel('Blocks', fontsize=16)
    
    # 设置x轴范围
    if all_points:
        min_t = min(all_points)
        max_t = max(all_points)
        ax.set_xlim(min_t * 0.95, max(max_t, actual_time) * 1.05)
        print(f"时间范围: {min_t} 到 {max_t}")
    else:
        print("警告: 没有找到任何时间数据")
        ax.set_xlim(0, 1)  # 默认范围
    
    # 添加标题和标签
    ax.set_xlabel("Timestamp", fontsize=16)
    ax.set_title(f"Computation, Bubble and Weight Update Times (Time Cost: {actual_time})", fontsize=18)
    
    # 图例
    legend_elements = [
        Patch(facecolor=CAL_COLOR, edgecolor=BORDER_COLOR, label='Compute time'),
        Patch(facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR, hatch='//', label='Bubble time'),
        Patch(facecolor=WEIGHT_COLOR, edgecolor=BORDER_COLOR, label='Weight update')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/Combined_Timelines_Block_Batch.png", dpi=400, bbox_inches='tight')
    plt.savefig(f"{save_path}/Combined_Timelines_Block_Batch.svg", dpi=600)
    plt.close(fig)
    print(f"图像已保存到: {save_path}/Combined_Timelines_Block_Batch.png")
def plot_combined_timelines_block_batch(save_path, layers_cal, layers_bubble, layers_weight, layer_names=None, block_labels=None, actual_time=0):
    """
    可视化多个block的计算时间、bubble时间和权重更新时间
    纵坐标是block，每个block内从上到下是不同的layer（按正确顺序）

    :param save_path: 图像保存路径
    :param layers_cal: 三维列表 [block][batch][layer] 每个层的计算时间段
    :param layers_bubble: 三维列表 [block][batch][layer] 每个层的bubble时间段
    :param layers_weight: 三维列表 [block][batch][layer] 每层的权重写入时间段
    :param block_labels: 每个block的标签
    :param actual_time: 系统总运行时间
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    num_blocks = len(layers_cal)
    num_batches = len(layers_cal[0]) if num_blocks > 0 else 0

    # 如果没有提供block标签，创建默认标签
    if block_labels is None:
        block_labels = [f"Block {i}" for i in range(num_blocks)]

    # 定义颜色方案
    CAL_COLOR = '#6495ED'      # 蓝色 - 计算时间
    BUBBLE_COLOR = 'lightgrey'   # 灰色 - Bubble时间
    WEIGHT_COLOR = '#D63344'   # 红色 - 权重更新时间
    BORDER_COLOR = 'black'     # 边框颜色

    # 定义层名称（默认）
    if layer_names is None:
        default_layer_names = ["Q", "K", "V", "A", "Z0", "Z1", "FFN1", "FFN2"]
    else:
        default_layer_names = layer_names

    # 收集所有时间点用于设置x轴范围
    all_points = []

    # 创建时间线图表
    fig, ax = plt.subplots(figsize=(20, 12))

    # 每个block高度
    block_height = 1.0

    # 循环所有block
    for block_idx in range(num_blocks):
        # 动态识别当前 block 的 layer 数量
        # 找到该 block 内所有 batch 的最大 layer 长度
        num_layers_block = max(len(batch) for batch in layers_cal[block_idx])

        # 动态生成该 block 的 layer 名称（不足用默认名字补齐）
        if layer_names is None:
            cur_layer_names = default_layer_names[:num_layers_block]
        else:
            cur_layer_names = layer_names[:num_layers_block]

        # 计算每层高度
        layer_height = block_height / num_layers_block

        y_center = num_blocks - block_idx  # 从顶部开始，block0在最上

        # 循环当前 block 的所有 layer
        for layer_idx in range(num_layers_block):
            layer_y = y_center - (layer_idx + 1) * layer_height

            # 循环当前 block 的所有 batch
            for batch_idx in range(num_batches):
                # =========================
                # Compute 时间
                # =========================
                if layer_idx < len(layers_cal[block_idx][batch_idx]):
                    cal_data = layers_cal[block_idx][batch_idx][layer_idx]
                    if cal_data and isinstance(cal_data, tuple) and len(cal_data) == 2:
                        start, end = cal_data
                        duration = end - start
                        ax.broken_barh(
                            [(start, duration)],
                            (layer_y - layer_height/2, layer_height),
                            facecolor=CAL_COLOR,
                            edgecolor=BORDER_COLOR,
                            linewidth=0.5
                        )
                        all_points.extend([start, end])

                # =========================
                # Bubble 时间
                # =========================
                if layer_idx < len(layers_bubble[block_idx][batch_idx]):
                    bubble_data = layers_bubble[block_idx][batch_idx][layer_idx]
                    if bubble_data:
                        if isinstance(bubble_data, tuple) and len(bubble_data) == 2:
                            start, end = bubble_data
                            duration = end - start
                            ax.broken_barh([(start, duration)], (layer_y-layer_height/2, layer_height),
                                            facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR, linewidth=0.2, hatch='//', alpha=0.4)
                            all_points.extend([start, end])
                        elif isinstance(bubble_data, dict):
                            if 'start' in bubble_data and 'end' in bubble_data:
                                start, end = bubble_data['start'], bubble_data['end']
                                duration = end - start
                                ax.broken_barh([(start, duration)], (layer_y-layer_height/2, layer_height),
                                                facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR, linewidth=0.2, hatch='//', alpha=0.4)
                                all_points.extend([start, end])
                            else:
                                for start, duration in bubble_data.items():
                                    ax.broken_barh([(start, duration)], (layer_y-layer_height/2, layer_height),
                                                    facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR, linewidth=0.2, hatch='//', alpha=0.4)
                                    all_points.extend([start, start+duration])

                # =========================
                # Weight update
                # =========================
                if layer_idx < len(layers_weight[block_idx][batch_idx]):
                    weight_data = layers_weight[block_idx][batch_idx][layer_idx]
                    if weight_data and isinstance(weight_data, tuple) and len(weight_data) == 2:
                        start, duration = weight_data
                        ax.broken_barh(
                            [(start, duration)],
                            (layer_y - layer_height/2, layer_height),
                            facecolor=WEIGHT_COLOR,
                            edgecolor=BORDER_COLOR,
                            linewidth=0.5
                        )
                        all_points.extend([start, start+duration])

    # 设置y轴标签（block标签）
    ax.set_yticks(range(1, num_blocks + 1))
    ax.set_yticklabels(reversed(block_labels), fontsize=14)
    ax.set_ylabel('Blocks', fontsize=16)

    # 设置x轴范围
    if all_points:
        min_t = min(all_points)
        max_t = max(all_points)
        ax.set_xlim(min_t * 0.95, max(max_t, actual_time) * 1.05)
        print(f"时间范围: {min_t} 到 {max_t}")
    else:
        print("警告: 没有找到任何时间数据")
        ax.set_xlim(0, 1)  # 默认范围

    # 添加标题和标签
    ax.set_xlabel("Timestamp", fontsize=16)
    ax.set_title(f"Computation, Bubble and Weight Update Times (Time Cost: {actual_time})", fontsize=18)

    # 图例
    legend_elements = [
        Patch(facecolor=CAL_COLOR, edgecolor=BORDER_COLOR, label='Compute time'),
        Patch(facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR, hatch='//', label='Bubble time'),
        Patch(facecolor=WEIGHT_COLOR, edgecolor=BORDER_COLOR, label='Weight update')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/Combined_Timelines_Block_Batch.png", dpi=400, bbox_inches='tight')
    plt.savefig(f"{save_path}/Combined_Timelines_Block_Batch.svg", dpi=600)
    plt.close(fig)
    print(f"图像已保存到: {save_path}/Combined_Timelines_Block_Batch.png")
def plot_combined_timelines_block_batch(save_path, layers_cal, layers_bubble, layers_weight, layer_names=None, block_labels=None, actual_time=0):
    """
    可视化多个block的计算时间、bubble时间和权重更新时间
    纵坐标是block，每个block内从上到下是不同的layer（按正确顺序）

    :param save_path: 图像保存路径
    :param layers_cal: 三维列表 [block][batch][layer] 每个层的计算时间段
    :param layers_bubble: 三维列表 [block][batch][layer] 每个层的bubble时间段
    :param layers_weight: 三维列表 [block][batch][layer] 每层的权重写入时间段
    :param block_labels: 每个block的标签
    :param actual_time: 系统总运行时间
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    num_blocks = len(layers_cal)
    num_batches = len(layers_cal[0]) if num_blocks > 0 else 0

    # 如果没有提供block标签，创建默认标签
    if block_labels is None:
        block_labels = [f"Block {i}" for i in range(num_blocks)]

    # 定义颜色方案
    CAL_COLOR = '#6495ED'      # 蓝色 - 计算时间
    BUBBLE_COLOR = 'lightgrey'   # 灰色 - Bubble时间
    WEIGHT_COLOR = '#D63344'   # 红色 - 权重更新时间
    BORDER_COLOR = 'black'     # 边框颜色

    # 定义层名称（默认）
    if layer_names is None:
        default_layer_names = ["Q", "K", "V", "A", "Z0", "Z1", "FFN1", "FFN2"]
    else:
        default_layer_names = layer_names

    # 收集所有时间点用于设置x轴范围
    all_points = []

    # 创建时间线图表
    fig, ax = plt.subplots(figsize=(20, 12))

    # 纵向累计偏移量（保证不同 block 不重叠）
    y_offset = 0
    block_positions = []  # 记录每个 block 的中间位置用于打标签

    # 循环所有block
    for block_idx in range(num_blocks):
        # 动态识别当前 block 的 layer 数量
        num_layers_block = max(len(batch) for batch in layers_cal[block_idx])

        # 动态生成该 block 的 layer 名称（不足用默认名字补齐）
        if layer_names is None:
            cur_layer_names = default_layer_names[:num_layers_block]
        else:
            cur_layer_names = layer_names[:num_layers_block]

        # 每层高度固定
        layer_height = 1

        # 循环当前 block 的所有 layer
        for layer_idx in range(num_layers_block):
            # 计算当前 layer 的纵坐标（从上往下）
            layer_y = -(y_offset + layer_idx)

            # 循环当前 block 的所有 batch
            for batch_idx in range(num_batches):
                # =========================
                # Compute 时间
                # =========================
                if layer_idx < len(layers_cal[block_idx][batch_idx]):
                    cal_data = layers_cal[block_idx][batch_idx][layer_idx]
                    if cal_data and isinstance(cal_data, tuple) and len(cal_data) == 2:
                        start, end = cal_data
                        duration = end - start
                        ax.broken_barh(
                            [(start, duration)],
                            (layer_y - layer_height/2, layer_height),
                            facecolor=CAL_COLOR,
                            edgecolor=BORDER_COLOR,
                            linewidth=0.5
                        )
                        all_points.extend([start, end])

                # =========================
                # Bubble 时间
                # =========================
                if layer_idx < len(layers_bubble[block_idx][batch_idx]):
                    bubble_data = layers_bubble[block_idx][batch_idx][layer_idx]
                    if bubble_data:
                        if isinstance(bubble_data, tuple) and len(bubble_data) == 2:
                            start, end = bubble_data
                            duration = end - start
                            ax.broken_barh([(start, duration)], (layer_y-layer_height/2, layer_height),
                                            facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR, linewidth=0.2, hatch='//', alpha=0.4)
                            all_points.extend([start, end])
                        elif isinstance(bubble_data, dict):
                            if 'start' in bubble_data and 'end' in bubble_data:
                                start, end = bubble_data['start'], bubble_data['end']
                                duration = end - start
                                ax.broken_barh([(start, duration)], (layer_y-layer_height/2, layer_height),
                                                facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR, linewidth=0.2, hatch='//', alpha=0.4)
                                all_points.extend([start, end])
                            else:
                                for start, duration in bubble_data.items():
                                    ax.broken_barh([(start, duration)], (layer_y-layer_height/2, layer_height),
                                                    facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR, linewidth=0.2, hatch='//', alpha=0.4)
                                    all_points.extend([start, start+duration])

                # =========================
                # Weight update
                # =========================
                if layer_idx < len(layers_weight[block_idx][batch_idx]):
                    weight_data = layers_weight[block_idx][batch_idx][layer_idx]
                    if weight_data and isinstance(weight_data, tuple) and len(weight_data) == 2:
                        start, duration = weight_data
                        ax.broken_barh(
                            [(start, duration)],
                            (layer_y - layer_height/2, layer_height),
                            facecolor=WEIGHT_COLOR,
                            edgecolor=BORDER_COLOR,
                            linewidth=0.5
                        )
                        all_points.extend([start, start+duration])

        # 记录当前 block 的中间位置
        block_positions.append(-(y_offset + num_layers_block/2 - 0.5))
        # 累积偏移（加上当前 block 的层数 + 间隔）
        y_offset += num_layers_block

    # 设置y轴标签（block标签）
    ax.set_yticks(block_positions)
    ax.set_yticklabels(block_labels, fontsize=14)
    ax.set_ylabel('Blocks', fontsize=16)

    # 设置x轴范围
    if all_points:
        min_t = min(all_points)
        max_t = max(all_points)
        ax.set_xlim(min_t * 0.95, max(max_t, actual_time) * 1.05)
        print(f"时间范围: {min_t} 到 {max_t}")
    else:
        print("警告: 没有找到任何时间数据")
        ax.set_xlim(0, 1)  # 默认范围

    # 添加标题和标签
    ax.set_xlabel("Timestamp", fontsize=16)
    ax.set_title(f"Computation, Bubble and Weight Update Times (Time Cost: {actual_time})", fontsize=18)

    # 图例
    legend_elements = [
        Patch(facecolor=CAL_COLOR, edgecolor=BORDER_COLOR, label='Compute time'),
        Patch(facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR, hatch='//', label='Bubble time'),
        Patch(facecolor=WEIGHT_COLOR, edgecolor=BORDER_COLOR, label='Weight update')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/Combined_Timelines_Block_Batch.png", dpi=400, bbox_inches='tight')
    plt.savefig(f"{save_path}/Combined_Timelines_Block_Batch.svg", dpi=600)
    plt.close(fig)
    print(f"图像已保存到: {save_path}/Combined_Timelines_Block_Batch.png")
def plot_combined_timelines_block_batch(save_path, layers_cal, layers_bubble, layers_weight, 
                                        layer_names=None, block_labels=None, actual_time=0):
    """
    可视化多个block的计算时间、bubble时间和权重更新时间
    纵坐标是block，每个block内从上到下是不同的layer（按正确顺序）

    :param save_path: 图像保存路径
    :param layers_cal: 三维列表 [block][batch][layer] 
                       每层的计算时间段，可以是 tuple(start,end) 或 list[(start,end),...]
    :param layers_bubble: 三维列表 [block][batch][layer] 
                          每层的bubble时间段，可以是 tuple(start,end) 或 list[(start,end),...]
    :param layers_weight: 三维列表 [block][batch][layer] 
                          每层的权重写入时间段，可以是 tuple(start,duration) 或 list[(start,duration),...]
    :param layer_names: 层名称列表（可选）
    :param block_labels: 每个block的标签（可选）
    :param actual_time: 系统总运行时间（可选）
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    num_blocks = len(layers_cal)
    num_batches = len(layers_cal[0]) if num_blocks > 0 else 0

    if block_labels is None:
        block_labels = [f"Block {i}" for i in range(num_blocks)]

    CAL_COLOR = '#6495ED'      # 蓝色 - 计算时间
    BUBBLE_COLOR = 'lightgrey' # 灰色 - Bubble时间
    WEIGHT_COLOR = '#D63344'   # 红色 - 权重更新时间
    BORDER_COLOR = 'black'

    if layer_names is None:
        default_layer_names = ["Q", "K", "V", "A", "Z0", "Z1", "FFN1", "FFN2"]
    else:
        default_layer_names = layer_names

    all_points = []
    fig, ax = plt.subplots(figsize=(20, 12))

    y_offset = 0
    block_positions = []

    for block_idx in range(num_blocks):
        num_layers_block = max(len(batch) for batch in layers_cal[block_idx])

        if layer_names is None:
            cur_layer_names = default_layer_names[:num_layers_block]
        else:
            cur_layer_names = layer_names[:num_layers_block]

        layer_height = 1

        for layer_idx in range(num_layers_block):
            layer_y = -(y_offset + layer_idx)

            for batch_idx in range(num_batches):
                # =========================
                # Compute 时间
                # =========================
                if layer_idx < len(layers_cal[block_idx][batch_idx]):
                    cal_data = layers_cal[block_idx][batch_idx][layer_idx]
                    if cal_data:
                        if isinstance(cal_data, tuple):
                            cal_data = [cal_data]
                        for start, end in cal_data:
                            duration = end - start
                            ax.broken_barh(
                                [(start, duration)],
                                (layer_y - layer_height/2, layer_height),
                                facecolor=CAL_COLOR,
                                edgecolor=BORDER_COLOR,
                                linewidth=0.1
                            )
                            all_points.extend([start, end])

                # =========================
                # Bubble 时间
                # =========================
                if layer_idx < len(layers_bubble[block_idx][batch_idx]):
                    bubble_data = layers_bubble[block_idx][batch_idx][layer_idx]
                    if bubble_data:
                        if isinstance(bubble_data, tuple):
                            bubble_data = [bubble_data]
                        elif isinstance(bubble_data, dict):
                            # dict: {start: duration, ...}
                            bubble_data = [(s, s+d) for s, d in bubble_data.items()]
                        for start, end in bubble_data:
                            duration = end - start
                            ax.broken_barh(
                                [(start, duration)],
                                (layer_y - layer_height/2, layer_height),
                                facecolor=BUBBLE_COLOR,
                                edgecolor=BORDER_COLOR,
                                linewidth=0.1,
                                hatch='//',
                                alpha=0.4
                            )
                            all_points.extend([start, end])

                # =========================
                # Weight update
                # =========================
                if layer_idx < len(layers_weight[block_idx][batch_idx]):
                    weight_data = layers_weight[block_idx][batch_idx][layer_idx]
                    if weight_data:
                        if isinstance(weight_data, tuple):
                            weight_data = [weight_data]
                        for start, duration in weight_data:
                            ax.broken_barh(
                                [(start, duration)],
                                (layer_y - layer_height/2, layer_height),
                                facecolor=WEIGHT_COLOR,
                                edgecolor=BORDER_COLOR,
                                linewidth=0.1
                            )
                            all_points.extend([start, start+duration])

        block_positions.append(-(y_offset + num_layers_block/2 - 0.5))
        y_offset += num_layers_block

    ax.set_yticks(block_positions)
    ax.set_yticklabels(block_labels, fontsize=14)
    ax.set_ylabel('Blocks', fontsize=16)

    if all_points:
        min_t = min(all_points)
        max_t = max(all_points)
        ax.set_xlim(min_t * 0.95, max(max_t, actual_time) * 1.05)
        print(f"时间范围: {min_t} 到 {max_t}")
    else:
        print("警告: 没有找到任何时间数据")
        ax.set_xlim(0, 1)

    ax.set_xlabel("Timestamp", fontsize=16)
    ax.set_title(f"Computation, Bubble and Weight Update Times (Time Cost: {actual_time})", fontsize=18)

    legend_elements = [
        Patch(facecolor=CAL_COLOR, edgecolor=BORDER_COLOR, label='Compute time'),
        Patch(facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR, hatch='//', label='Bubble time'),
        Patch(facecolor=WEIGHT_COLOR, edgecolor=BORDER_COLOR, label='Weight update')
    ]
    #ax.legend(handles=legend_elements, loc='upper right', fontsize=14)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/Combined_Timelines_Block_Batch.png", dpi=400, bbox_inches='tight')
    plt.savefig(f"{save_path}/Combined_Timelines_Block_Batch.svg", dpi=600)
    plt.close(fig)
    print(f"图像已保存到: {save_path}/Combined_Timelines_Block_Batch.png")

def plot_combined_timelines_batch_layers(save_path, layers_cal, layers_bubble, layers_weight, labels=None, batch_labels=None, actual_time=0):
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    num_layers = len(layers_cal)
    num_batches = len(layers_cal[0]) if num_layers > 0 else 0

    if labels is None:
        labels = [f"Layer {i+1}" for i in range(num_layers)]
    if batch_labels is None:
        batch_labels = [f"Batch {i+1}" for i in range(num_batches)]

    fig, ax = plt.subplots(figsize=(22, 12))

    CAL_COLOR = '#6495ED'
    BUBBLE_COLOR = '#FFA500'
    WEIGHT_COLOR = '#D63344'
    BORDER_COLOR = 'black'

    all_points = []

    layer_height = 0.8  # 每层高度

    for layer_idx in range(num_layers):
        y_center = num_layers - layer_idx  # 从上往下排列 layer

        for batch_idx in range(num_batches):

            # ==== Compute ====
            cal_data = layers_cal[layer_idx][batch_idx]
            if cal_data and isinstance(cal_data, tuple) and len(cal_data) == 2:
                start, end = cal_data
                ax.broken_barh(
                    [(start, end - start)],
                    (y_center - layer_height/2, layer_height),
                    facecolor=CAL_COLOR,
                    edgecolor=BORDER_COLOR,
                    linewidth=0.2,
                    label="Compute time" if (layer_idx==0 and batch_idx==0) else None
                )
                all_points.extend([start, end])

            # ==== Bubble ====
            bubble_data = layers_bubble[layer_idx][batch_idx]
            if bubble_data:
                if isinstance(bubble_data, dict):
                    # 遍历字典的所有时间段
                    for start, duration in bubble_data.items():
                        ax.broken_barh(
                            [(start, duration)],
                            (y_center - layer_height/2, layer_height),
                            facecolor=BUBBLE_COLOR,
                            edgecolor=BORDER_COLOR,
                            hatch='//',
                            linewidth=0.1,
                            alpha=0.8,   # 透明度 0.2
                            label="Bubble time" if (layer_idx==0 and batch_idx==0) else None
                        )
                        all_points.extend([start, start+duration])
                elif isinstance(bubble_data, tuple) and len(bubble_data)==2:
                    start, end = bubble_data
                    ax.broken_barh(
                        [(start, end-start)],
                        (y_center - layer_height/2, layer_height),
                        facecolor=BUBBLE_COLOR,
                        edgecolor=BORDER_COLOR,
                        hatch='//',
                        linewidth=0.1,
                        alpha=0.8,   # 透明度 0.2
                        label="Bubble time" if (layer_idx==0 and batch_idx==0) else None
                    )
                    all_points.extend([start,end])

            # ==== Weight ====
            weight_data = layers_weight[layer_idx][batch_idx]
            if weight_data:
                if isinstance(weight_data, tuple) and len(weight_data)==2:
                    start, duration = weight_data
                    ax.broken_barh(
                        [(start, duration)],
                        (y_center - layer_height/2, layer_height),
                        facecolor=WEIGHT_COLOR,
                        edgecolor=BORDER_COLOR,
                        linewidth=0.2,
                        label="Weight update" if (layer_idx==0 and batch_idx==0) else None
                    )
                    all_points.extend([start, start+duration])
                elif isinstance(weight_data, list):
                    for (start,duration) in weight_data:
                        ax.broken_barh(
                            [(start,duration)],
                            (y_center - layer_height/2, layer_height),
                            facecolor=WEIGHT_COLOR,
                            edgecolor=BORDER_COLOR,
                            linewidth=0.2,
                            label="Weight update" if (layer_idx==0 and batch_idx==0) else None
                        )
                        all_points.extend([start,start+duration])

    # y轴
    ax.set_yticks(range(1, num_layers+1))
    ax.set_yticklabels(reversed(labels), fontsize=14)
    ax.set_ylabel("Layers", fontsize=16)

    # x轴
    if all_points:
        min_t, max_t = min(all_points), max(all_points)
        ax.set_xlim(min_t*0.95, max(max_t, actual_time)*1.05)
    else:
        ax.set_xlim(0, actual_time if actual_time>0 else 1)

    ax.set_xlabel("Timestamp", fontsize=16)
    ax.set_title(f"Computation, Bubble and Weight Update Times (Time Cost: {actual_time})", fontsize=18)

    legend_elements = [
        Patch(facecolor=CAL_COLOR, edgecolor=BORDER_COLOR, label="Compute time"),
        Patch(facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR, hatch='//', label="Bubble time"),
        Patch(facecolor=WEIGHT_COLOR, edgecolor=BORDER_COLOR, label="Weight update"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/Combined_Timelines_BatchLayers.png", dpi=400, bbox_inches="tight")
    plt.savefig(f"{save_path}/Combined_Timelines_BatchLayers.svg", dpi=600)
    plt.close(fig)
    print(f"图像已保存到: {save_path}/Combined_Timelines_BatchLayers.png")
def plot_combined_timelines_two_groups(save_path, 
                                       layers1_cal, layers1_bubble, layers1_weight, block_labels1=None, actual_time1=0,
                                       layers2_cal=None, layers2_bubble=None, layers2_weight=None, block_labels2=None, actual_time2=0,
                                       layer_names=None,
                                       layer_height=0.8,  # 每层矩形高度，可调
                                       margin_between_groups=2.0):  # 两组之间的间距
    """
    在同一个坐标轴里绘制两个 timeline 组（上下拼接，共用一个横坐标轴）
    每层矩形高度可调
    只保留 legend 和 x 轴数值（无 title、无 y 轴标题）
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def plot_group(ax, layers_cal, layers_bubble, layers_weight, block_labels, offset, layer_names):
        num_blocks = len(layers_cal)
        num_batches = len(layers_cal[0]) if num_blocks > 0 else 0
        num_layers = len(layers_cal[0][0]) if num_batches > 0 else 0

        if block_labels is None:
            block_labels = [f"Block {i}" for i in range(num_blocks)]

        CAL_COLOR = '#6495ED'
        BUBBLE_COLOR = 'lightgrey'
        WEIGHT_COLOR = '#D63344'
        BORDER_COLOR = 'black'

        if layer_names is None:
            layer_names = ["Q", "K", "V", "A", "Z0", "Z1", "FFN1", "FFN2"]

        all_points = []

        # block 高度 = 层数 * 每层高度
        block_height = num_layers * layer_height

        for block_idx in range(num_blocks):
            # block 顶部
            y_top = offset + block_height * (num_blocks - block_idx)
            for layer_idx in range(num_layers):
                # 从顶部往下计算 layer 底部位置
                layer_y = y_top - (layer_idx + 1) * layer_height
                for batch_idx in range(num_batches):
                    # -------------------- compute --------------------
                    if block_idx < len(layers_cal) and batch_idx < len(layers_cal[block_idx]) and layer_idx < len(layers_cal[block_idx][batch_idx]):
                        cal_data = layers_cal[block_idx][batch_idx][layer_idx]
                        if cal_data and isinstance(cal_data, tuple) and len(cal_data) == 2:
                            start, end = cal_data
                            duration = end - start
                            ax.broken_barh([(start, duration)], (layer_y, layer_height),
                                           facecolor=CAL_COLOR, edgecolor=BORDER_COLOR, linewidth=0.2)
                            all_points.extend([start, end])
                    # -------------------- bubble --------------------
                    if block_idx < len(layers_bubble) and batch_idx < len(layers_bubble[block_idx]) and layer_idx < len(layers_bubble[block_idx][batch_idx]):
                        bubble_data = layers_bubble[block_idx][batch_idx][layer_idx]
                        if bubble_data:
                            if isinstance(bubble_data, tuple) and len(bubble_data) == 2:
                                start, end = bubble_data
                                duration = end - start
                                ax.broken_barh([(start, duration)], (layer_y, layer_height),
                                               facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR, linewidth=0.2, hatch='//', alpha=0.4)
                                all_points.extend([start,end])
                            elif isinstance(bubble_data, dict):
                                if 'start' in bubble_data and 'end' in bubble_data:
                                    start, end = bubble_data['start'], bubble_data['end']
                                    duration = end - start
                                    ax.broken_barh([(start,duration)], (layer_y, layer_height),
                                                   facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR, linewidth=0.2, hatch='//', alpha=0.4)
                                    all_points.extend([start,end])
                                else:
                                    for start,duration in bubble_data.items():
                                        ax.broken_barh([(start,duration)], (layer_y, layer_height),
                                                       facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR, linewidth=0.2, hatch='//', alpha=0.4)
                                        all_points.extend([start, start+duration])
                    # -------------------- weight --------------------
                    if block_idx < len(layers_weight) and batch_idx < len(layers_weight[block_idx]) and layer_idx < len(layers_weight[block_idx][batch_idx]):
                        weight_data = layers_weight[block_idx][batch_idx][layer_idx]
                        if weight_data and isinstance(weight_data, tuple) and len(weight_data) == 2:
                            start, duration = weight_data
                            ax.broken_barh([(start, duration)], (layer_y, layer_height),
                                           facecolor=WEIGHT_COLOR, edgecolor=BORDER_COLOR, linewidth=0.2)
                            all_points.extend([start, start+duration])

        return num_blocks, all_points, block_height

    # ---------------- 绘制 ----------------
    # 计算总高度用于 figsize
    num_layers1 = len(layers1_cal[0][0]) if layers1_cal else 0
    num_layers2 = len(layers2_cal[0][0]) if layers2_cal else 0
    num_blocks1 = len(layers1_cal) if layers1_cal else 0
    num_blocks2 = len(layers2_cal) if layers2_cal else 0

    height1 = num_blocks1 * num_layers1 * layer_height
    height2 = num_blocks2 * num_layers2 * layer_height

    total_height = height1 + height2 + margin_between_groups
    fig, ax = plt.subplots(figsize=(16, 12))  # 高度适当缩放

    # 绘制第一组
    num_blocks1, points1, block_height1 = plot_group(ax, layers1_cal, layers1_bubble, layers1_weight,
                                                     block_labels1, offset=0, layer_names=layer_names)

    # 绘制第二组（向下偏移）
    offset2 = -(num_blocks1 * block_height1 + margin_between_groups)
    num_blocks2, points2, block_height2 = plot_group(ax, layers2_cal, layers2_bubble, layers2_weight,
                                                     block_labels2, offset=offset2, layer_names=layer_names)

    # 不显示纵坐标
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel("")

    # 只保留横坐标
    ax.set_xlabel("Timestamp", fontsize=18)
    ax.tick_params(axis='x', labelsize=18)

    # 设置 x 轴范围
    all_points = points1 + points2
    if all_points:
        min_t, max_t = min(all_points), max(all_points)
        ax.set_xlim(min_t * 0.95, max(max_t, max(actual_time1, actual_time2)) * 1.05)

    # 图例
    legend_elements = [
        Patch(facecolor='#6495ED', edgecolor='black', label='Compute time'),
        Patch(facecolor='lightgrey', edgecolor='black', hatch='//', label='Bubble time'),
        Patch(facecolor='#D63344', edgecolor='black', label='Weight update')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=18)

    ax.grid(axis='x', alpha=0.6)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/Two_Groups_Timeline.svg", dpi=600, format='svg')
    plt.close(fig)
    print(f"图像已保存到: {save_path}/Two_Groups_Timeline.svg")
def plot_combined_timelines_two_groups(save_path, 
                                       layers1_cal, layers1_bubble, layers1_weight, block_labels1=None, actual_time1=0,
                                       layers2_cal=None, layers2_bubble=None, layers2_weight=None, block_labels2=None, actual_time2=0,
                                       layer_names=None,
                                       layer_height=1.0,  # 每层矩形高度，默认 1，层间紧挨
                                       margin_between_groups=2.0):  # 两组之间的间距
    """
    在同一个坐标轴里绘制两个 timeline 组（上下拼接，共用一个横坐标轴）
    每层矩形高度可调
    兼容每个 block 内部 layer 数不同
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def plot_group(ax, layers_cal, layers_bubble, layers_weight, block_labels, offset, layer_names):
        num_blocks = len(layers_cal)
        all_points = []

        if block_labels is None:
            block_labels = [f"Block {i}" for i in range(num_blocks)]

        CAL_COLOR = '#6495ED'
        BUBBLE_COLOR = 'lightgrey'
        WEIGHT_COLOR = '#D63344'
        BORDER_COLOR = 'black'

        if layer_names is None:
            layer_names = ["Q", "K", "V", "A", "Z0", "Z1", "FFN1", "FFN2"]

        y_offset = offset

        for block_idx in range(num_blocks):
            num_batches = len(layers_cal[block_idx]) if block_idx < len(layers_cal) else 0
            num_layers = len(layers_cal[block_idx][0]) if num_batches > 0 else 0

            for layer_idx in range(num_layers):
                layer_y = y_offset - (layer_idx + 1) * layer_height
                for batch_idx in range(num_batches):
                    # -------------------- compute --------------------
                    if layer_idx < len(layers_cal[block_idx][batch_idx]):
                        cal_data = layers_cal[block_idx][batch_idx][layer_idx]
                        if cal_data and isinstance(cal_data, tuple) and len(cal_data) == 2:
                            start, end = cal_data
                            duration = end - start
                            ax.broken_barh([(start, duration)], (layer_y, layer_height),
                                           facecolor=CAL_COLOR, edgecolor=BORDER_COLOR, linewidth=0.2)
                            all_points.extend([start, end])
                    # -------------------- bubble --------------------
                    if layer_idx < len(layers_bubble[block_idx][batch_idx]):
                        bubble_data = layers_bubble[block_idx][batch_idx][layer_idx]
                        if bubble_data:
                            if isinstance(bubble_data, tuple) and len(bubble_data) == 2:
                                start, end = bubble_data
                                duration = end - start
                                ax.broken_barh([(start, duration)], (layer_y, layer_height),
                                               facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR,
                                               linewidth=0.2, hatch='//', alpha=0.4)
                                all_points.extend([start, end])
                            elif isinstance(bubble_data, dict):
                                if 'start' in bubble_data and 'end' in bubble_data:
                                    start, end = bubble_data['start'], bubble_data['end']
                                    duration = end - start
                                    ax.broken_barh([(start, duration)], (layer_y, layer_height),
                                                   facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR,
                                                   linewidth=0.2, hatch='//', alpha=0.4)
                                    all_points.extend([start, end])
                                else:
                                    for start, duration in bubble_data.items():
                                        ax.broken_barh([(start, duration)], (layer_y, layer_height),
                                                       facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR,
                                                       linewidth=0.2, hatch='//', alpha=0.4)
                                        all_points.extend([start, start + duration])
                    # -------------------- weight --------------------
                    if layer_idx < len(layers_weight[block_idx][batch_idx]):
                        weight_data = layers_weight[block_idx][batch_idx][layer_idx]
                        if weight_data and isinstance(weight_data, tuple) and len(weight_data) == 2:
                            start, duration = weight_data
                            ax.broken_barh([(start, duration)], (layer_y, layer_height),
                                           facecolor=WEIGHT_COLOR, edgecolor=BORDER_COLOR, linewidth=0.2)
                            all_points.extend([start, start + duration])

            # 累加偏移量（按实际层数叠加）
            y_offset = layer_y

        return all_points, y_offset

    # ---------------- 绘制 ----------------
    fig, ax = plt.subplots(figsize=(16, 12))  # 高度适当缩放

    # 绘制第一组
    points1, offset1 = plot_group(ax, layers1_cal, layers1_bubble, layers1_weight,
                                  block_labels1, offset=0, layer_names=layer_names)

    # 绘制第二组（向下偏移）
    points2, offset2 = plot_group(ax, layers2_cal, layers2_bubble, layers2_weight,
                                  block_labels2, offset=offset1 - margin_between_groups, layer_names=layer_names)

    # 不显示纵坐标
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel("")

    # 只保留横坐标
    ax.set_xlabel("Timestamp", fontsize=18)
    ax.tick_params(axis='x', labelsize=18)

    # 设置 x 轴范围
    all_points = points1 + points2
    if all_points:
        min_t, max_t = min(all_points), max(all_points)
        ax.set_xlim(min_t * 0.95, max(max_t, max(actual_time1, actual_time2)) * 1.05)

    # 图例
    legend_elements = [
        Patch(facecolor='#6495ED', edgecolor='black', label='Compute time'),
        Patch(facecolor='lightgrey', edgecolor='black', hatch='//', label='Bubble time'),
        Patch(facecolor='#D63344', edgecolor='black', label='Weight update')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=18)

    ax.grid(axis='x', alpha=0.6)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/Two_Groups_Timeline.svg", dpi=600, format='svg')
    plt.close(fig)
    print(f"图像已保存到: {save_path}/Two_Groups_Timeline.svg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def plot_combined_timelines_two_groups(save_path, 
                                       layers1_cal, layers1_bubble, layers1_weight, block_labels1=None, actual_time1=0,
                                       layers2_cal=None, layers2_bubble=None, layers2_weight=None, block_labels2=None, actual_time2=0,
                                       layer_names=None,
                                       layer_height=1.0,  # 每层矩形高度，默认 1，层间紧挨
                                       margin_between_groups=2.0):  # 两组之间的间距
    """
    在同一个坐标轴里绘制两个 timeline 组（上下拼接，共用一个横坐标轴）
    每层矩形高度可调
    兼容每个 block 内部 layer 数不同 & 支持 batch 多数据
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def plot_group(ax, layers_cal, layers_bubble, layers_weight, block_labels, offset, layer_names):
        num_blocks = len(layers_cal)
        all_points = []

        if block_labels is None:
            block_labels = [f"Block {i}" for i in range(num_blocks)]

        CAL_COLOR = '#6495ED'
        BUBBLE_COLOR = 'lightgrey'
        WEIGHT_COLOR = '#D63344'
        BORDER_COLOR = 'black'

        if layer_names is None:
            layer_names = ["Q", "K", "V", "A", "Z0", "Z1", "FFN1", "FFN2"]

        y_offset = offset

        for block_idx in range(num_blocks):
            num_batches = len(layers_cal[block_idx]) if block_idx < len(layers_cal) else 0
            num_layers = len(layers_cal[block_idx][0]) if num_batches > 0 else 0

            for layer_idx in range(num_layers):
                layer_y = y_offset - (layer_idx + 1) * layer_height
                for batch_idx in range(num_batches):
                    # -------------------- compute --------------------
                    if layer_idx < len(layers_cal[block_idx][batch_idx]):
                        cal_data = layers_cal[block_idx][batch_idx][layer_idx]
                        if cal_data:
                            if isinstance(cal_data, tuple) and len(cal_data) == 2:
                                # 单个 (start, end)
                                start, end = cal_data
                                duration = end - start
                                ax.broken_barh([(start, duration)], (layer_y, layer_height),
                                               facecolor=CAL_COLOR, edgecolor=BORDER_COLOR, linewidth=0.1)
                                all_points.extend([start, end])
                            elif isinstance(cal_data, list):
                                # 多个 (start, end)
                                for start, end in cal_data:
                                    duration = end - start
                                    ax.broken_barh([(start, duration)], (layer_y, layer_height),
                                                   facecolor=CAL_COLOR, edgecolor=BORDER_COLOR, linewidth=0.1)
                                    all_points.extend([start, end])

                    # -------------------- bubble --------------------
                    if layer_idx < len(layers_bubble[block_idx][batch_idx]):
                        bubble_data = layers_bubble[block_idx][batch_idx][layer_idx]
                        if bubble_data:
                            if isinstance(bubble_data, tuple) and len(bubble_data) == 2:
                                start, end = bubble_data
                                duration = end - start
                                ax.broken_barh([(start, duration)], (layer_y, layer_height),
                                               facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR,
                                               linewidth=0.1, hatch='//', alpha=0.4)
                                all_points.extend([start, end])
                            elif isinstance(bubble_data, list):
                                for start, end in bubble_data:
                                    duration = end - start
                                    ax.broken_barh([(start, duration)], (layer_y, layer_height),
                                                   facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR,
                                                   linewidth=0.1, hatch='//', alpha=0.4)
                                    all_points.extend([start, end])
                            elif isinstance(bubble_data, dict):
                                if 'start' in bubble_data and 'end' in bubble_data:
                                    start, end = bubble_data['start'], bubble_data['end']
                                    duration = end - start
                                    ax.broken_barh([(start, duration)], (layer_y, layer_height),
                                                   facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR,
                                                   linewidth=0.1, hatch='//', alpha=0.4)
                                    all_points.extend([start, end])
                                else:
                                    for start, duration in bubble_data.items():
                                        ax.broken_barh([(start, duration)], (layer_y, layer_height),
                                                       facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR,
                                                       linewidth=0.1, hatch='//', alpha=0.4)
                                        all_points.extend([start, start + duration])

                    # -------------------- weight --------------------
                    if layer_idx < len(layers_weight[block_idx][batch_idx]):
                        weight_data = layers_weight[block_idx][batch_idx][layer_idx]
                        if weight_data:
                            if isinstance(weight_data, tuple) and len(weight_data) == 2:
                                start, duration = weight_data
                                ax.broken_barh([(start, duration)], (layer_y, layer_height),
                                               facecolor=WEIGHT_COLOR, edgecolor=BORDER_COLOR, linewidth=0.1)
                                all_points.extend([start, start + duration])
                            elif isinstance(weight_data, list):
                                for start, duration in weight_data:
                                    ax.broken_barh([(start, duration)], (layer_y, layer_height),
                                                   facecolor=WEIGHT_COLOR, edgecolor=BORDER_COLOR, linewidth=0.1)
                                    all_points.extend([start, start + duration])

            # 累加偏移量（按实际层数叠加）
            y_offset = layer_y

        return all_points, y_offset

    # ---------------- 绘制 ----------------
    fig, ax = plt.subplots(figsize=(16, 12))  # 高度适当缩放

    # 绘制第一组
    points1, offset1 = plot_group(ax, layers1_cal, layers1_bubble, layers1_weight,
                                  block_labels1, offset=0, layer_names=layer_names)

    # 绘制第二组（向下偏移）
    points2, offset2 = plot_group(ax, layers2_cal, layers2_bubble, layers2_weight,
                                  block_labels2, offset=offset1 - margin_between_groups, layer_names=layer_names)

    # 不显示纵坐标
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel("")

    # 只保留横坐标
    ax.set_xlabel("Timestamp", fontsize=18)
    ax.tick_params(axis='x', labelsize=18)

    # 设置 x 轴范围
    all_points = points1 + points2
    if all_points:
        min_t, max_t = min(all_points), max(all_points)
        ax.set_xlim(min_t * 0.95, max(max_t, max(actual_time1, actual_time2)) * 1.05)

    # 图例（保持原样）
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#6495ED', edgecolor='black', label='Compute time'),
        Patch(facecolor='lightgrey', edgecolor='black', hatch='//', label='Bubble time'),
        Patch(facecolor='#D63344', edgecolor='black', label='Weight update')
    ]
    #ax.legend(handles=legend_elements, loc='upper right', fontsize=18)

    ax.grid(axis='x', alpha=0.6)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/Two_Groups_Timeline.svg", dpi=600, format='svg')
    plt.close(fig)
    print(f"图像已保存到: {save_path}/Two_Groups_Timeline.svg")


def plot_combined_timelines_three_groups(save_path, 
                                         layers1_cal, layers1_bubble, layers1_weight, block_labels1=None, actual_time1=0,
                                         layers2_cal=None, layers2_bubble=None, layers2_weight=None, block_labels2=None, actual_time2=0,
                                         layers3_cal=None, layers3_bubble=None, layers3_weight=None, block_labels3=None, actual_time3=0,
                                         layer_names=None,
                                         layer_height=1.0,  
                                         margin_between_groups=2.0):  
    """
    在同一坐标轴绘制三组 timeline（上下排列），支持每组多 block、多 layer、多 batch
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def plot_group(ax, layers_cal, layers_bubble, layers_weight, block_labels, offset, layer_names):
        num_blocks = len(layers_cal)
        all_points = []

        if block_labels is None:
            block_labels = [f"Block {i}" for i in range(num_blocks)]

        CAL_COLOR = '#6495ED'
        BUBBLE_COLOR = 'lightgrey'
        WEIGHT_COLOR = '#D63344'
        BORDER_COLOR = 'black'

        if layer_names is None:
            layer_names = ["Q", "K", "V", "A", "Z0", "Z1", "FFN1", "FFN2"]

        y_offset = offset

        for block_idx in range(num_blocks):
            num_batches = len(layers_cal[block_idx]) if block_idx < len(layers_cal) else 0
            num_layers = len(layers_cal[block_idx][0]) if num_batches > 0 else 0

            for layer_idx in range(num_layers):
                layer_y = y_offset - (layer_idx + 1) * layer_height
                for batch_idx in range(num_batches):
                    # -------------------- compute --------------------
                    if layer_idx < len(layers_cal[block_idx][batch_idx]):
                        cal_data = layers_cal[block_idx][batch_idx][layer_idx]
                        if cal_data:
                            items = cal_data if isinstance(cal_data, list) else [cal_data]
                            for entry in items:
                                if isinstance(entry, tuple) and len(entry) == 2:
                                    start, end = entry
                                    duration = end - start
                                    ax.broken_barh([(start, duration)], (layer_y, layer_height),
                                                   facecolor=CAL_COLOR, edgecolor=BORDER_COLOR, linewidth=0.1)
                                    all_points.extend([start, end])

                    # -------------------- bubble --------------------
                    if layer_idx < len(layers_bubble[block_idx][batch_idx]):
                        bubble_data = layers_bubble[block_idx][batch_idx][layer_idx]
                        if bubble_data:
                            items = bubble_data if isinstance(bubble_data, list) else [bubble_data]
                            for entry in items:
                                if isinstance(entry, tuple) and len(entry) == 2:
                                    start, end = entry
                                    duration = end - start
                                    ax.broken_barh([(start, duration)], (layer_y, layer_height),
                                                   facecolor=BUBBLE_COLOR, edgecolor=BORDER_COLOR,
                                                   linewidth=0.1, hatch='//', alpha=0.4)
                                    all_points.extend([start, end])

                    # -------------------- weight --------------------
                    if layer_idx < len(layers_weight[block_idx][batch_idx]):
                        weight_data = layers_weight[block_idx][batch_idx][layer_idx]
                        if weight_data:
                            items = weight_data if isinstance(weight_data, list) else [weight_data]
                            for entry in items:
                                if isinstance(entry, tuple) and len(entry) == 2:
                                    start, duration = entry
                                    ax.broken_barh([(start, duration)], (layer_y, layer_height),
                                                   facecolor=WEIGHT_COLOR, edgecolor=BORDER_COLOR, linewidth=0.1)
                                    all_points.extend([start, start + duration])

            y_offset = layer_y  # 累积纵向偏移

        return all_points, y_offset

    # ---------------- 绘制 ----------------
    fig, ax = plt.subplots(figsize=(18, 14))

    points1, offset1 = plot_group(ax, layers1_cal, layers1_bubble, layers1_weight,
                                  block_labels1, offset=0, layer_names=layer_names)
    points2, offset2 = plot_group(ax, layers2_cal, layers2_bubble, layers2_weight,
                                  block_labels2, offset=offset1 - margin_between_groups, layer_names=layer_names)
    points3, offset3 = plot_group(ax, layers3_cal, layers3_bubble, layers3_weight,
                                  block_labels3, offset=offset2 - margin_between_groups, layer_names=layer_names)

    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel("")

    ax.set_xlabel("Timestamp", fontsize=18)
    ax.tick_params(axis='x', labelsize=18)

    all_points = points1 + points2 + points3
    if all_points:
        min_t, max_t = min(all_points), max(all_points + [actual_time1, actual_time2, actual_time3])
        ax.set_xlim(min_t * 0.95, max_t * 1.05)

    legend_elements = [
        Patch(facecolor='#6495ED', edgecolor='black', label='Compute time'),
        Patch(facecolor='lightgrey', edgecolor='black', hatch='//', label='Bubble time'),
        Patch(facecolor='#D63344', edgecolor='black', label='Weight update')
    ]
    #ax.legend(handles=legend_elements, loc='upper right', fontsize=16)

    ax.grid(axis='x', alpha=0.6)
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/Three_Groups_Timeline.svg", dpi=600, format='svg')
    plt.close(fig)
    print(f"图像已保存到: {save_path}/Three_Groups_Timeline.svg")


def export_to_excel(data, filename='output.xlsx', sheet_name='Sheet1'):
    """
    将列表数据导出到 Excel 文件
    
    参数:
        data (list): 要导出的数据列表，可以是一维或二维列表
        filename (str): 输出的 Excel 文件名，默认为 'output.xlsx'
        sheet_name (str): Excel 工作表名称，默认为 'Sheet1'
    """
    try:
        # 检查数据是否为一维列表
        if all(not isinstance(item, list) for item in data):
            # 如果是一维列表，转换为包含单个列的 DataFrame
            df = pd.DataFrame(data, columns=['数据'])
        else:
            # 如果是二维列表，直接转换为 DataFrame
            df = pd.DataFrame(data)
        
        # 导出到 Excel
        df.to_excel(filename, sheet_name=sheet_name, index=False)
        print(f"数据已成功导出到 {filename}")
        
    except Exception as e:
        print(f"导出时出错: {e}")
def plot_layer_computation_time(
    non_pipeline_times, pipeline_times, labels, dnn_name, savepath, bar_width=0.4, fontsize_base=12
):
    """
    绘制每层计算时间对比（并排柱状图）
    """
    x = np.arange(len(labels))  # 横轴坐标
    plt.figure(figsize=(20, 9))
    
    # 无流水线柱状图
    plt.bar(x, non_pipeline_times, width=bar_width, label='without pipeline', color='#1f77b4')
    # 有流水线柱状图（右移一个宽度）
    plt.bar(x + bar_width, pipeline_times, width=bar_width, label='with pipeline', color='#ff7f0e')
    
    # 坐标轴设置
    plt.ylabel('Cycles', fontsize=fontsize_base+8)
    plt.xticks(x + bar_width/2, labels, fontsize=fontsize_base+1)
    plt.yticks(fontsize=fontsize_base+3)
    plt.title(f'ISAAC computation time for {dnn_name}', fontsize=fontsize_base+13)
    plt.legend(fontsize=fontsize_base+6)
    
    plt.tight_layout()
    plt.savefig(f"{savepath}/Computation_cycle_comparison_per_layer.png", dpi=400, bbox_inches='tight')
def plot_computation_cycle(
    cycle_non_pipeline, cycle_pipeline, dnn_name, savepath, bar_width=0.4, fontsize_base=12
):
    """
    绘制计算周期对比图
    """
    plt.figure(figsize=(10, 8))
    bars = plt.bar(
        ["without pipeline", "with pipeline"],
        [cycle_non_pipeline, cycle_pipeline],
        bar_width,
        color=['#1f77b4', '#ff7f0e']
    )
    
    # 设置标签和标题
    plt.xticks(fontsize=fontsize_base+1)
    plt.yticks(fontsize=fontsize_base+3)
    plt.ylabel('Computation cycle (Cycle)', fontsize=fontsize_base+8)
    plt.title(f'Computation cycle comparison of {dnn_name}', fontsize=fontsize_base+13)
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.0f}', ha='center', va='bottom', fontsize=fontsize_base+2)
    
    plt.tight_layout()
    plt.savefig(f"{savepath}/Computation_cycle_comparison.png", dpi=400, bbox_inches='tight')

def plot_energy_comparison(origin_energy, pipeline_energy, dnn_name, savepath, bar_width=0.4, fontsize_base=12):
    """
    绘制有无流水线的能量对比柱状图
    
    参数:
    origin_energy (float): 无流水线总能量
    pipeline_energy (float): 有流水线总能量
    dnn_name (str): DNN模型名称
    savepath (str): 图片保存路径
    bar_width (float): 柱状图宽度
    fontsize_base (int): 基础字体大小
    """
    plt.figure(figsize=(10, 8))
    bars = plt.bar(
        ["without pipeline", "with pipeline"],
        [origin_energy, pipeline_energy],
        bar_width,
        color=['#1f77b4', '#ff7f0e']
    )
    
    # 设置字体
    plt.xticks(fontsize=fontsize_base+1)
    plt.yticks(fontsize=fontsize_base+3)
    plt.ylabel("Energy (uJ)", fontsize=fontsize_base+8)
    plt.title(f'Energy comparison of {dnn_name}', fontsize=fontsize_base+13)
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f'{height:.0f}',
            ha='center',
            va='bottom',
            fontsize=fontsize_base+2
        )
    
    plt.tight_layout()
    plt.savefig(f"{savepath}/Energy_comparison.png", dpi=400, bbox_inches='tight')
    # plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

def plot_cycle_comparison(path1, path2, path3, labels,output_path=None, title="Cycle Optimization Comparison", show_plot=True):
    """
    读取三个Excel文件的第一列并生成计算周期比较柱状图
    
    参数:
    path1, path2, path3: Excel文件路径
    labels: 三个数据集的标签列表
    output_path: 图表保存路径(可选)
    title: 图表标题
    show_plot: 是否显示图表
    """
    # 读取Excel文件并处理数据
    try:
        df1 = pd.read_excel(path1).iloc[:, 0]
        df2 = pd.read_excel(path2).iloc[:, 0].drop(0)
        df = pd.read_excel(path3).drop(0)
        df3 = df.iloc[:, 0]
        df4 = df.iloc[:,3]
        print("数据集1:")
        print(df1.to_string())
        print("\n数据集2:")
        print(df2.to_string())
        print("\n数据集3:")
        print(df3.to_string())
        
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return
    
    # 确保所有数据集有相同的长度（取最小长度）
    min_length = min(len(df1), len(df2), len(df3))
    df1 = df1.iloc[:min_length]
    df2 = df2.iloc[:min_length]
    df3 = df3.iloc[:min_length]
    
    # 创建索引（假设为1,2,3...）
    index = np.arange(1, min_length + 1)
    
    # 设置图形大小
    plt.figure(figsize=(16, 10))
    
    # 设置柱状图宽度和位置
    bar_width = 0.25
    positions1 = index - bar_width
    positions2 = index
    positions3 = index + bar_width
    
    # 定义颜色方案
    base_colors = ['#1F77B4', '#FF7F0E', '#2CA02C']  # 基础颜色（更饱和）
    light_colors = ['#AEC7E8', '#FFBB78', '#98DF8A']  # 浅色版本（保持不变）
    dark_colors = ['#0D47A1', '#E65100', '#1B5E20']   # 深色版本（用于边框和强调）
    
    # 绘制柱状图
    bars1 = plt.bar(positions1, df1, bar_width, color=base_colors[0], edgecolor=dark_colors[0], linewidth=1, label="Original Mapping")
    bars2 = plt.bar(positions2, df2, bar_width, color=base_colors[1], edgecolor=dark_colors[1], linewidth=1, label="Pipeline Mapping")
    bars3 = plt.bar(positions3, df3, bar_width, color=base_colors[2], edgecolor=dark_colors[2], linewidth=1, label="Pipeline Mapping(optimization)")
    
    # 为每个柱子添加前一层的标记
    
    def add_previous_layer_markers(bars, data, base_color, light_color,hatch):
        for i in range(1, len(bars)):
            prev_value = data.iloc[i-1]
            current_value = data.iloc[i]
            current_bar = bars[i]
            # 计算变化百分比
            change_percent = ((current_value - prev_value) / prev_value) * 100
            
            # 在当前柱子上叠加显示前一层的值
            overlay_height = min(prev_value, current_value)
            overlay_bar = Rectangle(
                (current_bar.get_x(), 0),
                current_bar.get_width(),
                overlay_height,
                facecolor=light_color,
                alpha=0.7,
                edgecolor='black',
                linewidth=0.5,
                hatch=hatch
            )
            plt.gca().add_patch(overlay_bar)
           
    # 为每个数据集添加前一层标记
    add_previous_layer_markers(bars1, df1, base_colors[0], light_colors[0],'/')
    add_previous_layer_markers(bars2, df2, base_colors[1], light_colors[1],'/')
    add_previous_layer_markers(bars3, df3, base_colors[2], light_colors[2],'/')

    # 图表属性
    plt.xlabel('Layer', fontsize=13)
    plt.ylabel('Cycles', fontsize=13)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(index, index)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # 保存和显示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {output_path}")
    if show_plot:
        plt.show()
    
    # 返回统计数据

def get_metric(file_path):
        try:
            results = {
                'energy': None,
                'utilization': None,
                'cycles': None
            }
            
            with open(file_path, 'r') as file:
                for line in file:
                    # 匹配Energy行
                    match_energy = re.search(r'Energy:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*uJ', line)
                    if match_energy and results['energy'] is None:
                        results['energy'] = float(match_energy.group(1))
                    
                    # 匹配Utilization行
                    match_util = re.search(r'Utilization:\s*([+-]?\d+(?:\.\d+)?)\s*%', line)
                    if match_util and results['utilization'] is None:
                        results['utilization'] = float(match_util.group(1))
                    
                    # 匹配Cycles行
                    match_cycles = re.search(r'Cycles:\s*(\d+)', line)
                    if match_cycles and results['cycles'] is None:
                        results['cycles'] = int(match_cycles.group(1))
                    
                    # 如果三个指标都已获取，则提前结束循环
                    if all(results.values()):
                        break
            
            return results
            
        except FileNotFoundError:
            print(f"错误：文件 '{file_path}' 未找到")
            return None
        except Exception as e:
            print(f"读取文件时发生错误: {e}")
            return None   
def plot_separate_metrics(result_list, title_prefix="Metrics", figsize=(18, 6), show_values=True):
    """
    为utilization、cycles和energy分别绘制独立的柱状图
    
    参数:
    - result_list: 包含多个结果字典的列表，每个字典包含energy、utilization和cycles键
    - title_prefix: 图表标题前缀
    - figsize: 整个图表的尺寸
    - show_values: 是否在柱状图上方显示具体数值
    """
    if not result_list:
        print("错误：结果列表为空")
        return
    
    # 提取数据
    indices = list(range(len(result_list)))
    energy_values = [r.get('energy', 0) for r in result_list]
    util_values = [r.get('utilization', 0) for r in result_list]
    cycles_values = [r.get('cycles', 0) for r in result_list]
    
    # 创建包含3个子图的图表
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plt.subplots_adjust(wspace=0.3)  # 增加子图间距
    
    # 定义每个指标的绘图参数
    metrics = [
        {
            'name': 'Utilization', 
            'values': util_values, 
            'color': 'lightgreen', 
            'ylabel': 'Utilization (%)',
            'scale': 1,
            'yticks_format': '{x:.0f}',
            'padding': 0.1,  # Y轴上方留出10%的空间
            'label_pad': 5   # Y轴标签与轴线的间距
        },
        {
            'name': 'Cycles', 
            'values': cycles_values, 
            'color': 'salmon', 
            'ylabel': 'Cycles (x10e5)',
            'scale': 100000,  # 以10万为单位显示
            'yticks_format': '{x:.1f}',
            'padding': 0.15,  # 循环数值通常较大，留出15%的空间
            'label_pad': 5
        },
        {
            'name': 'Energy', 
            'values': energy_values, 
            'color': 'skyblue', 
            'ylabel': 'Energy (mJ)',
            'scale': 1000,
            'yticks_format': '{x:.1f}',
            'padding': 0.1,
            'label_pad': 5
        }
    ]
    
    # 绘制每个指标的柱状图
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.set_title(f"{title_prefix} - {metric['name']}", fontsize=14)
        
        # 数据标准化处理
        normalized_values = [v/metric['scale'] for v in metric['values']]
        
        # 绘制柱状图
        bars = ax.bar(indices, normalized_values, color=metric['color'])
        
        # 计算Y轴最大值，增加额外空间以避免标签超出边界
        max_value = max(normalized_values) if normalized_values else 0
        y_upper_limit = max_value * (1 + metric['padding'])
        ax.set_ylim(0, y_upper_limit)
        
        # 添加数值标签
        if show_values:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            rotation=270, fontsize=10)
        
        # 设置X轴标签和刻度
        ax.set_xlabel('Index', fontsize=12)
        
        # 设置Y轴标签，调整位置和间距
        ax.set_ylabel(metric['ylabel'], fontsize=12, labelpad=metric['label_pad'])
        
        ax.set_xticks(indices)
        #ax.set_xticklabels([f'Idx {i}' for i in indices])
        
        # 格式化Y轴刻度
        from matplotlib.ticker import StrMethodFormatter
        ax.yaxis.set_major_formatter(StrMethodFormatter(metric['yticks_format']))
        
        # 添加网格线
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    return fig
def plot_single_metric(results_group1, results_group2, metric, 
                      group1_label="Group 1", group2_label="Group 2", 
                      title="Metric Comparison", ylabel="Value", 
                      figsize=(10, 6), show_values=True):
    """
    绘制单个指标的对比柱状图
    
    参数:
    - results_group1: 第一组结果列表（字典列表）
    - results_group2: 第二组结果列表（字典列表）
    - metric: 要绘制的指标名称
    - group1_label: 第一组的标签
    - group2_label: 第二组的标签
    - title: 图表标题
    - ylabel: Y轴标签
    - figsize: 图表尺寸
    - show_values: 是否在柱状图上方显示具体数值
    """
    if not results_group1 or not results_group2:
        print(f"错误：结果列表为空，无法绘制{metric}对比图")
        return None
    
    # 确保两组数据长度相同
    if len(results_group1) != len(results_group2):
        print(f"警告：两组数据长度不一致 ({len(results_group1)} vs {len(results_group2)})")
        min_len = min(len(results_group1), len(results_group2))
        results_group1 = results_group1[:min_len]
        results_group2 = results_group2[:min_len]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 提取数据
    indices = list(range(len(results_group1)))
    group1_values = [r.get(metric, 0) for r in results_group1]
    group2_values = [r.get(metric, 0) for r in results_group2]
    
    # 设置柱状图参数
    bar_width = 0.35
    index = np.arange(len(indices))
    
    # 绘制柱状图
    bar1 = ax.bar(index - bar_width/2, group1_values, bar_width, 
                 label=group1_label, color='skyblue')
    
    bar2 = ax.bar(index + bar_width/2, group2_values, bar_width, 
                 label=group2_label, color='lightgreen')
    
    # 添加数值标签
    if show_values:
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            rotation=90, fontsize=8)
        
        add_labels(bar1)
        add_labels(bar2)
    
    # 设置标题和标签
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(index)
    ax.set_xticklabels([f'Idx {i}' for i in indices])
    
    # 添加网格线和图例
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    
    # 自动调整Y轴范围，留出15%的空间
    max_value = max(max(group1_values), max(group2_values)) if group1_values and group2_values else 0
    ax.set_ylim(0, max_value * 1.15)
    
    plt.tight_layout()
    return fig

def plot_three_metrics_separate(results_group1, results_group2, 
                               group1_label="Group 1", group2_label="Group 2", 
                               title_prefix="Metrics Comparison", figsize=(10, 6), 
                               show_values=True, save_dir=None):
    """
    绘制三个指标的对比图，每个指标一张独立图表
    
    参数:
    - results_group1: 第一组结果列表（字典列表）
    - results_group2: 第二组结果列表（字典列表）
    - group1_label: 第一组的标签
    - group2_label: 第二组的标签
    - title_prefix: 图表标题前缀
    - figsize: 每个图表的尺寸
    - show_values: 是否在柱状图上方显示具体数值
    - save_dir: 保存图表的目录，如果为None则不保存
    """
    metrics = [
        {'name': 'utilization', 'title': 'Utilization Comparison', 'ylabel': 'Utilization (%)'},
        {'name': 'cycles', 'title': 'Cycles Comparison', 'ylabel': 'Cycles'},
        {'name': 'energy', 'title': 'Energy Comparison', 'ylabel': 'Energy (uJ)'}
    ]
    
    figures = []
    
    for metric in metrics:
        fig = plot_single_metric(
            results_group1=results_group1,
            results_group2=results_group2,
            metric=metric['name'],
            group1_label=group1_label,
            group2_label=group2_label,
            title=f"{title_prefix} - {metric['title']}",
            ylabel=metric['ylabel'],
            figsize=figsize,
            show_values=show_values
        )
        
        if fig:
            figures.append(fig)
            if save_dir:
                fig.savefig(f"{save_dir}/{metric['name']}_comparison.png", dpi=300, bbox_inches='tight')
    
    return figures
def plot_multi_group_utilization(results_groups, group_labels=None,
                               title="Utilization Comparison", figsize=(20, 12),
                               show_values=True, save_path=None, edge_width=0.5,save = True):
    """
    绘制多个数据组的 Utilization 指标对比图
    
    参数:
    - results_groups: 结果组的列表，每个元素是一个结果列表（字典列表）
    - group_labels: 各组的标签列表，如果为None则使用默认标签
    - title: 图表标题
    - figsize: 图表的尺寸
    - show_values: 是否在柱状图上方显示具体数值
    - save_path: 保存图表的路径，如果为None则不保存
    - edge_width: 柱状图边框线宽
    """
    
    # 确保有数据
    if not results_groups or not all(results for results in results_groups):
        raise ValueError("At least one non-empty results group is required")
    
    # 获取数据组数量
    num_groups = len(results_groups)
    colors = [
    "#88CCEE",  # 浅蓝（无约束映射）
    "#FFDD88",  # 暖黄（柔性映射 M=4,c=2）
    "#44BB99",  # 草绿（柔性映射 M=8,c=1）
    "#EE6666"   # 浅红（固定映射）
    ]
    # 生成默认标签（如果未提供）
    if group_labels is None:
        group_labels = [f"Group {i+1}" for i in range(num_groups)]
    elif len(group_labels) != num_groups:
        raise ValueError("The number of group labels must match the number of results groups")
    
    # 从每个组中提取 utilization 值
    util_values = []
    for results in results_groups:
        values = [result.get('utilization', 0) for result in results]
        util_values.append(values)
    
    # 确保所有组的数据长度相同
    min_length = min(len(values) for values in util_values)
    util_values = [values[:min_length] for values in util_values]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置柱状图的位置
    x = np.arange(min_length)
    width = 0.8 / num_groups  # 根据组数动态调整宽度
    multiplier = 0
    
    # 绘制所有组的柱状图
    for attribute, measurement in zip(group_labels, util_values):
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color = colors[multiplier],linewidth=edge_width, edgecolor='black')
        if show_values:
            ax.bar_label(rects, padding=3, fmt='%.1f',rotation = 90)
        multiplier += 1
    
    # 设置图表属性
    ax.set_ylabel('Utilization (%)')
    ax.set_title(title)
    ax.set_xticks(x + width * (num_groups - 1) / 2, [f'{i+1}' for i in range(min_length)])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()

    # 保存图表（如果需要）
    if save:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_multi_group_utilization_average(results_groups, group_labels=None,
                                      title="Average Utilization Comparison", figsize=(10, 6),
                                      show_values=True, save_path=None, color_palette=None, edge_width=1.0,save = True):
    """
    绘制多个数据组的 Utilization 平均值对比柱状图
    
    参数:
    - results_groups: 结果组的列表，每个元素是一个结果列表（字典列表）
    - group_labels: 各组的标签列表，如果为None则使用默认标签
    - title: 图表标题
    - figsize: 图表的尺寸
    - show_values: 是否在柱状图上方显示具体数值
    - save_path: 保存图表的路径，如果为None则不保存
    - color_palette: 自定义颜色列表，用于多个柱状图
    - edge_width: 柱状图边框线宽
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 确保有数据
    if not results_groups or not all(results for results in results_groups):
        raise ValueError("At least one non-empty results group is required")
    
    # 获取数据组数量
    num_groups = len(results_groups)
    
    # 生成默认标签（如果未提供）
    if group_labels is None:
        group_labels = [f"Group {i+1}" for i in range(num_groups)]
    elif len(group_labels) != num_groups:
        raise ValueError("The number of group labels must match the number of results groups")

    # 从每个组中提取 utilization 值并计算平均值和标准差
    avg_values = []
    std_values = []
    
    for results in results_groups:
        values = [result.get('utilization', 0) for result in results]
        avg_values.append(np.mean(values))
        std_values.append(np.std(values) if len(values) > 1 else 0)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    colors = [
    "#88CCEE",  # 浅蓝（无约束映射）
    "#FFDD88",  # 暖黄（柔性映射 M=4,c=2）
    "#44BB99",  # 草绿（柔性映射 M=8,c=1）
    "#EE6666"   # 浅红（固定映射）
    ]
    
    # 绘制柱状图
    bars = ax.bar(group_labels, avg_values, 
                 color=colors, linewidth=edge_width, edgecolor='black')
    
    # 在柱状图上方显示平均值
    if show_values:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(avg_values),
                    f'{height:.2f}', ha='center', va='bottom')
    
    # 设置图表属性
    ax.set_ylabel('Average Utilization (%)')
    ax.set_title(title)
    ax.set_ylim(0, max(avg_values) * 1.15)  # 稍微增加y轴上限，为标签留出空间
    
    plt.tight_layout()
    
    # 保存图表（如果需要）
    if save:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
def get_dummy_top_scalar_access(file_path, stats_identifier):
    """
    从文本文件中提取以特定stats部分开头的dummy_top的Total scalar accesses值
    
    参数:
    file_path (str): 要分析的文本文件路径
    stats_identifier (str): 用于标识目标部分的统计信息标题，如"Operational Intensity Stats"
    
    返回:
    int or None: 如果找到匹配项，返回scalar access大小的整数值；否则返回None
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            
            # 使用正则表达式查找指定的stats部分及其后的dummy_top部分
            pattern = re.compile(
                rf'{re.escape(stats_identifier)}(.*?)'  # 匹配stats部分
                rf'=== dummy_top ===(.*?)'             # 匹配目标dummy_top标记
                rf'Op per Byte\s*:\s*([\d.e+]+)',      # 匹配Op per Byte行以确保获取正确的部分
                re.DOTALL
            )
            
            match = pattern.search(content)
            if match:
                # 提取dummy_top部分的内容
                dummy_top_content = match.group(2)
                
                # 使用正则表达式提取Total scalar accesses值
                scalar_pattern = r'Total scalar accesses\s*:\s*(\d+)'
                scalar_match = re.search(scalar_pattern, dummy_top_content)
                
                if scalar_match:
                    return int(scalar_match.group(1))
    
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
    except Exception as e:
        print(f"错误: 读取文件时发生异常: {e}")
    
    return None

def plot_access_comparison(layers, access_reuse, access_no_reuse, access_reuse_output, access_reuse_input, 
                           DNN="vgg16", show_normalized=True):
    """
    可视化不同复用策略下各层的scalar access对比，可选择显示归一化总访问次数对比
    
    参数:
    layers (list): 层名称列表
    access_reuse (list): 完全复用策略的访问次数列表
    access_no_reuse (list): 无复用策略的访问次数列表
    access_reuse_output (list): 输出复用策略的访问次数列表
    access_reuse_input (list): 输入复用策略的访问次数列表
    DNN (str): 神经网络名称，用于图表标题
    show_normalized (bool): 是否显示归一化总访问次数对比图
    """
    # 创建层对比图
    fig1, ax1 = plt.subplots(figsize=(15, 8))
    
    # 设置柱状图参数
    n_layers = len(layers)
    bar_width = 0.8 / 4  # 四个复用策略
    indices = np.arange(n_layers)
    
    # 定义颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['reuse', 'reuse input', 'reuse output', 'no reuse']
    
    # 绘制柱状图
    bars = []
    data_sets = [access_reuse, access_reuse_input, access_reuse_output, access_no_reuse]
    
    for i, data in enumerate(data_sets):
        position = indices + i * bar_width - (4 - 1) * bar_width / 2
        bar = ax1.bar(position, data, bar_width, label=labels[i], color=colors[i])
        bars.append(bar)
    
    # 设置图表属性
    ax1.set_xlabel('layers', fontsize=15)
    ax1.set_ylabel('Scalar Access ', fontsize=15)
    ax1.set_title(f'{DNN} Scalar Access Comparison Of Different Reuse ', fontsize=18)
    ax1.set_xticks(indices)
    ax1.set_xticklabels(layers, rotation=45, ha='right')
    ax1.legend(fontsize=20)
    

    
    # 添加网格线使图表更易读
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 美化图表
    plt.tight_layout()
    
    # 如果需要，创建归一化总访问次数对比图
    # 计算每种复用策略的总Scalar Access次数
    total_reuse = sum(access_reuse)
    total_no_reuse = sum(access_no_reuse)
    total_reuse_output = sum(access_reuse_output)
    total_reuse_input = sum(access_reuse_input)
        
        # 以完全复用为基准进行归一化
    normalization_base = total_reuse
    normalized_values = [
            total_reuse / normalization_base,
            total_reuse_input / normalization_base,
            total_reuse_output / normalization_base,
            total_no_reuse / normalization_base
        ]
        
        # 创建图表
    fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # 定义颜色方案和标签
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['reuse all', 'reuse input', 'reuse output', 'no reuse']
        
        # 绘制归一化柱状图
    bars = ax2.bar(labels, normalized_values, color=colors)
        
        # 设置图表属性
    ax2.set_xlabel('Layers', fontsize=15)
    ax2.set_ylabel('Scalar Access(Normalization)', fontsize=15)
    ax2.set_title(f'{DNN} Scalar Access Comparison of Different Reuse Strategies', fontsize=18)
        
        # 在柱子上方显示数值
    for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # 添加水平线标记基准值1.0
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
        
        # 美化图表
    plt.tight_layout()
        
    plt.show()
def extract_dummy_top_scalar_reads(file_path):
    # 初始化结果字典
    result = {
        'Weights': None,
        'Inputs': None,
        'Outputs': None
    }
    
    current_section = None
    in_dummy_top = False
    
    with open(file_path, 'r') as file:
        for line in file:
            # 检查是否进入dummy_top区域
            if "Level 16" in line:
                in_dummy_top = True
                continue
                
            if not in_dummy_top:
                continue
                
            # 检测子部分开始
            if "Weights:" in line:
                current_section = 'Weights'
            elif "Inputs:" in line:
                current_section = 'Inputs'
            elif "Outputs:" in line:
                current_section = 'Outputs'
                
            # 提取scalar reads值
            if current_section and "Scalar reads (per-instance)" in line:
                # 提取冒号后的数字部分
                value_str = line.split(':')[-1].strip()
                # 去除逗号并转换为整数
                value = int(value_str.replace(',', ''))
                result[current_section]= value
                
            # 当遇到下一个Level时停止处理
            if "Level 17" in line or "Summary Stats" in line:
                break
                
    return result
def extract_component_scalar_reads(file_path, component_name):
    # 初始化结果字典（包含所有可能的数据部分）
    result = {
        'Weights': None,
        'Inputs': None,
        'Outputs': None
    }
    
    current_section = None
    found_component = False
    component_found = False  # 确保只处理第一个匹配的组件
    
    with open(file_path, 'r') as file:
        for line in file:
            # 检查是否进入目标元件区域（使用精确匹配）
            if f"=== {component_name} ===" in line:
                if component_found:
                    # 如果已经找到过组件，跳过后续出现的同名组件
                    continue
                found_component = True
                component_found = True
                continue
            
            if not found_component:
                continue
                
            # 检测子部分开始（支持Weights/Inputs/Outputs）
            if "Weights:" in line:
                current_section = 'Weights'
            elif "Inputs:" in line:
                current_section = 'Inputs'
            elif "Outputs:" in line:
                current_section = 'Outputs'
            elif "=== " in line and " ===" in line:
                # 遇到新组件，停止处理
                break
                
            # 提取scalar reads值（如果当前有激活的部分）
            if current_section and "Scalar reads (per-instance)" in line:
                try:
                    # 提取冒号后的数字部分
                    value_str = line.split(':')[-1].strip()
                    # 去除逗号并转换为整数
                    value = int(value_str.replace(',', ''))
                    result[current_section] = value
                except (ValueError, IndexError):
                    # 忽略格式错误的数据行
                    pass
            if current_section and "Scalar fills (per-instance)" in line:
                try:
                    # 提取冒号后的数字部分
                    value_str = line.split(':')[-1].strip()
                    # 去除逗号并转换为整数
                    value = int(value_str.replace(',', ''))
                    result[current_section] += value
                except (ValueError, IndexError):
                    # 忽略格式错误的数据行
                    pass
            if current_section and "Scalar updates (per-instance)" in line:
                try:
                    # 提取冒号后的数字部分
                    value_str = line.split(':')[-1].strip()
                    # 去除逗号并转换为整数
                    value = int(value_str.replace(',', ''))
                    result[current_section] += value
                except (ValueError, IndexError):
                    # 忽略格式错误的数据行
                    pass    
            # 当遇到下一个组件或文件结束时停止处理
            if "Summary Stats" in line or not line:
                break
                
    # 过滤掉未找到的部分（可选）
    return {k: v for k, v in result.items() if v is not None}
def extract_component_scalar_access(file_path, component_name):
    """
    从Timeloop统计文件中提取指定组件的标量访问数据
    
    参数:
    file_path -- Timeloop统计文件路径
    component_name -- 要提取数据的组件名称
    
    返回:
    包含以下键的字典:
      'Weights' - (reads + fills) 值
      'Inputs' - (reads + fills) 值
      'Outputs' - (reads + fills) 值
    """
    # 初始化结果字典
    result = {
        'Weights': {'reads': 0, 'fills': 0},
        'Inputs': {'reads': 0, 'fills': 0},
        'Outputs': {'reads': 0, 'fills': 0}
    }
    
    current_section = None
    found_component = False
    component_found = False
    
    with open(file_path, 'r') as file:
        for line in file:
            # 检查是否进入目标元件区域
            if f"=== {component_name} ===" in line:
                if component_found:
                    continue
                found_component = True
                component_found = True
                continue
            
            if not found_component:
                continue
                
            # 检测子部分开始
            if "Weights:" in line:
                current_section = 'Weights'
            elif "Inputs:" in line:
                current_section = 'Inputs'
            elif "Outputs:" in line:
                current_section = 'Outputs'
            elif "=== " in line and " ===" in line:
                # 遇到新组件，停止处理
                break
                
            # 提取scalar reads值
            if current_section and "Scalar reads (per-instance)" in line:
                try:
                    value_str = line.split(':')[-1].strip()
                    value = int(value_str.replace(',', ''))
                    result[current_section]['reads'] = value
                except (ValueError, IndexError):
                    pass
                
            # 提取scalar fills值
            if current_section and "Scalar fills (per-instance)" in line:
                try:
                    value_str = line.split(':')[-1].strip()
                    value = int(value_str.replace(',', ''))
                    result[current_section]['fills'] = value
                except (ValueError, IndexError):
                    pass
                
            # 当遇到下一个组件或文件结束时停止处理
            if "Summary Stats" in line or not line:
                break
                
    # 计算每个部分的访问总量 (reads + fills)
    final_result = {}
    for section, values in result.items():
        total = values['reads'] + values['fills']
        if total > 0:  # 只包含有访问量的部分
            final_result[section] = total
    
    return final_result

def plot_scalar_reads_comparison(layers, result_CM, result_MC, save_path=None, save=False):
    # 计算每种情况的总scalar reads，使用get()方法处理可能缺失的部分
    total_ws = [res.get('Weights', 0) + res.get('Inputs', 0) + res.get('Outputs', 0) 
                for res in result_CM]
    total_os = [res.get('Weights', 0) + res.get('Inputs', 0) + res.get('Outputs', 0) 
                for res in result_MC]
    
    # 设置柱状图参数
    bar_width = 0.25
    index = np.arange(len(layers))
    
    # 创建图形
    plt.figure(figsize=(12, 7))
    
    # 颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝, 橙, 绿
    
    # 绘制三种情况的柱状图
    plt.bar(index, total_ws, bar_width, label='Weight Stationary (WS)', 
            edgecolor='black', linewidth=0.5, color=colors[0])
    plt.bar(index + bar_width, total_os, bar_width, label='Output Stationary (OS)', 
            edgecolor='black', linewidth=0.5, color=colors[1])
    
    # 添加标签和标题
    plt.xlabel('Neural Network Layers', fontsize=12)
    plt.ylabel('Total Scalar Reads', fontsize=12)
    plt.title('Comparison of Scalar Reads by Permutation', fontsize=14)
    plt.xticks(index + bar_width, layers, rotation=45, fontsize=10)
    plt.yscale('log')  # 使用对数刻度因为数值可能差异很大
    plt.legend(fontsize=10)


    # 添加网格线以便更好地读取数值
    plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)
    
    # 调整布局并显示
    plt.tight_layout()
    
    # 保存图片
    if save and save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    


def plot_stacked_scalar_reads(layers, result_CM, result_MC, save_path=None, save=False):
    # 提取每种策略下各层的数据，使用get()方法处理可能缺失的部分
    ws_inputs = [res.get('Inputs', 0) for res in result_CM]
    ws_weights = [res.get('Weights', 0) for res in result_CM]
    ws_outputs = [res.get('Outputs', 0) for res in result_CM]
    
    os_inputs = [res.get('Inputs', 0) for res in result_MC]
    os_weights = [res.get('Weights', 0) for res in result_MC]
    os_outputs = [res.get('Outputs', 0) for res in result_MC]
    
    
    # 设置图表参数
    bar_width = 0.25
    index = np.arange(len(layers))
    
    # 创建图形和子图
    fig, ax = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    plt.subplots_adjust(hspace=0.15)
    
    # 颜色定义
    colors = {
        'Weights': '#2ca02c',  # 蓝色
        'Inputs': '#1f77b4' ,  # 橙色
        'Outputs': '#ff7f0e'   # 绿色
    }
    
    # 绘制WS策略
    ws_bars = []
    ws_labels = []
    if any(ws_weights) > 0:
        ws_bars.append(ax[0].bar(index, ws_weights, bar_width, color=colors['Weights']))
        ws_labels.append('Weights')
    if any(ws_inputs) > 0:
        bottom = ws_weights
        ws_bars.append(ax[0].bar(index, ws_inputs, bar_width, bottom=bottom, color=colors['Inputs']))
        ws_labels.append('Inputs')
    if any(ws_outputs) > 0:
        bottom = np.add(ws_weights, ws_inputs)
        ws_bars.append(ax[0].bar(index, ws_outputs, bar_width, bottom=bottom, color=colors['Outputs']))
        ws_labels.append('Outputs')
    
    # 添加图例（仅当有数据时）
    if ws_labels:
        ax[0].legend([bar[0] for bar in ws_bars], ws_labels)
    ax[0].set_title('CMPQ')
    ax[0].set_yscale('log')
    
    # 绘制OS策略
    os_bars = []
    if any(os_weights) > 0:
        os_bars.append(ax[1].bar(index, os_weights, bar_width, color=colors['Weights']))
    if any(os_inputs) > 0:
        bottom = os_weights
        os_bars.append(ax[1].bar(index, os_inputs, bar_width, bottom=bottom, color=colors['Inputs']))
    if any(os_outputs) > 0:
        bottom = np.add(os_weights, os_inputs)
        os_bars.append(ax[1].bar(index, os_outputs, bar_width, bottom=bottom, color=colors['Outputs']))
    ax[1].set_title('MCPQ')
    ax[1].set_yscale('log')
    # 设置公共标签
    plt.xticks(index, layers, rotation=45, fontsize=10)
    fig.text(0.04, 0.5, 'Scalar Reads (log scale)', va='center', rotation='vertical', fontsize=12)
    fig.text(0.5, 0.04, 'Neural Network Layers', ha='center', fontsize=12)
    
    # 添加总数标签
    for i in range(len(layers)):
        total_ws = ws_weights[i] + ws_inputs[i] + ws_outputs[i]
        total_os = os_weights[i] + os_inputs[i] + os_outputs[i]
        
        # 只显示非零值
        if total_ws > 0:
            ax[0].text(i, total_ws*1.1, f'{total_ws:,}', ha='center', va='bottom', fontsize=8)
        if total_os > 0:
            ax[1].text(i, total_os*1.1, f'{total_os:,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout(rect=[0.04, 0.04, 1, 0.96])
    
    # 保存图片
    if save and save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
def plot_combined_dataflow_comparison(layers, result_ws, result_os, save_path=None, save=False):
    """
    绘制三种策略的总数据访问量对比柱状图，并在每个柱子内部堆叠展示输入、权重和输出的访问量
    
    参数:
    layers -- 神经网络层名称列表
    result_ws -- WS策略的提取结果列表
    result_os -- OS策略的提取结果列表
    result_is -- IS策略的提取结果列表
    save_path -- 图片保存路径（可选）
    save -- 是否保存图片（默认为False）
    """
    # 1. 计算每种策略的总访问量（按数据类型）
    def calculate_totals(results):
        weights = sum(res.get('Weights', 0) for res in results)
        inputs = sum(res.get('Inputs', 0) for res in results)
        outputs = sum(res.get('Outputs', 0) for res in results)
        return weights, inputs, outputs
    
    # 计算每种策略的总访问量
    ws_weights, ws_inputs, ws_outputs = calculate_totals(result_ws)
    os_weights, os_inputs, os_outputs = calculate_totals(result_os)
    
    # 2. 准备绘图数据
    strategies = ['CMPQ', 'MCPQ']
    weights = [ws_weights, os_weights]
    inputs = [ws_inputs, os_inputs]
    outputs = [ws_outputs, os_outputs]
    print(inputs,outputs)
    # 3. 创建图表
    plt.figure(figsize=(10, 7))
    
    # 颜色定义
    colors = {
        'Weights': '#2ca02c',  # 蓝色
        'Inputs': '#1f77b4' ,  # 橙色
        'Outputs': '#ff7f0e'   # 绿色
    }
    
    # 4. 绘制堆叠柱状图
    # 首先绘制权重部分（底部）
    p1 = plt.bar(strategies, weights, color=colors['Weights'], edgecolor='black', label='Weights')
    
    # 在权重基础上绘制输入部分
    p2 = plt.bar(strategies, inputs, bottom=weights, color=colors['Inputs'], edgecolor='black', label='Inputs')
    
    # 在输入基础上绘制输出部分
    p3 = plt.bar(strategies, outputs, bottom=np.add(weights, inputs), color=colors['Outputs'], 
                 edgecolor='black', label='Outputs')
    
    # 5. 添加标签和标题
    plt.xlabel('Dataflow Strategy', fontsize=12)
    plt.ylabel('Total Scalar Reads', fontsize=12)
    plt.title('Total Data Access Comparison by Dataflow Strategy', fontsize=14)
    plt.yscale('log')  # 使用对数刻度因为数值可能差异很大
    
    # 6. 添加数值标签
    for i, strategy in enumerate(strategies):
        total = weights[i] + inputs[i] + outputs[i]
        #plt.text(i, total * 1.1, f'{total:,}', ha='center', va='bottom', fontsize=10)
        plt.annotate(f'{total:,}',
             xy=(i, total),
             xytext=(0, 5),  # x轴偏移0，y轴偏移5像素
             textcoords='offset points',
             ha='center', va='bottom', fontsize=10)

        

    
    # 7. 添加图例
    plt.legend()
    
    # 8. 添加网格线
    plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)
    
    # 9. 调整布局
    plt.tight_layout()
    
    # 10. 保存图片
    if save and save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")

def plot_stacked_bar_comparison(layers, result_list_groups, group_labels, save_path=None, plot_detail=False, yscale='linear'):
    num_layers = len(layers)
    num_groups = len(group_labels)

    # 策略对应颜色（非详细模式使用）
    group_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    
    # 策略对应hatch样式（详细模式使用，第一个为无填充）
    group_hatches = [None, '//',  'xx', '\\\\','..', '--', '++']  # 使用None表示无hatch
    
    # 组件对应颜色（详细模式使用）
    component_colors = ['#aec7e8', '#ffbb78', '#98df8a']  # 分别对应Inputs、Weights、Outputs的浅色

    bar_width = 0.18
    group_spacing = 0.2
    x = np.arange(len(layers))*1.2

    plt.figure(figsize=(max(16, len(layers) * 1.5), 10))

    if plot_detail:
        # 详细模式：使用hatch区分策略，组件颜色保持一致
        for group_idx, results in enumerate(result_list_groups):
            offset = (group_idx - (num_groups - 1) / 2) * group_spacing
            x_offset = x + offset
            bottom = np.zeros(num_layers)
            
            for comp_idx, comp_name in enumerate(['Inputs', 'Weights', 'Outputs']):
                comp_values = [res.get(comp_name, 0) for res in results]
                
                # 绘制带描边的柱状图
                bars = plt.bar(x_offset, comp_values, bar_width,
                        bottom=bottom,
                        color=component_colors[comp_idx],
                        edgecolor='black',
                        linewidth=1,  # 增加线条宽度
                        hatch=group_hatches[group_idx % len(group_hatches)],
                        label=comp_name if group_idx == 0 and comp_idx == 0 else "")
                bottom += np.array(comp_values)


        plt.ylabel("Access Count")
        plt.title("Access Breakdown by Component")
        
        # 创建组合图例
        from matplotlib.patches import Patch
        strategy_patches = [Patch(facecolor='white', edgecolor='black', hatch=group_hatches[i], label=group_labels[i]) for i in range(num_groups)]
        component_patches = [Patch(color=component_colors[i], label=['Inputs', 'Weights', 'Outputs'][i]) for i in range(3)]
        
        plt.legend(handles=strategy_patches + component_patches, title='Strategy & Component',
           bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    else:
        # 非详细模式：使用颜色区分策略，不使用hatch
        for group_idx, results in enumerate(result_list_groups):
            totals = []
            for res in results:
                total = sum(res.get(k, 0) for k in ['Inputs', 'Weights', 'Outputs'])
                totals.append(total)

            offset = (group_idx - (num_groups - 1) / 2) * group_spacing
            x_offset = x + offset

            # 绘制带描边的柱状图
            plt.bar(x_offset, totals, bar_width,
                    color=group_colors[group_idx % len(group_colors)],
                    edgecolor='black',
                    linewidth=1.5,  # 增加线条宽度
                    label=group_labels[group_idx])

        plt.ylabel("Total Access Count (Inputs + Weights + Outputs)")
        plt.title("Total Access Comparison (No Breakdown)")
        plt.legend(title='Strategy', loc='upper right')

    # 设置Y轴刻度
    plt.yscale(yscale)
    
    # 如果是对数刻度，添加网格线以帮助读取数值
    if yscale == 'log':
        plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
    
    # 添加坐标轴数值标注
    plt.xticks(x, layers, rotation=45, ha='right')
    plt.grid(False)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        plt.savefig(save_path, dpi=300,bbox_inches='tight')
    else:
        plt.show()

def plot_fusion_access_comparison(excel_file, save_path=None):
    df = pd.read_excel(excel_file).sort_values('Fusion_access')
    
    # 提取数据（注意列名匹配，根据截图调整）
    orig_weights = df['Original_weights']   # 列E
    orig_inputs = df['Original_inputs']     # 列F
    orig_outputs = df['Original_outputs']   # 列G
    fuse_weights = df['Fusion_weights']   # 列H
    fuse_inputs = df['Fusion_inputs']     # 列I
    fuse_outputs = df['Fusion_outputs']   # 列J
    
    n = len(df)
    bar_width = 0.3
    idx = np.arange(n)
    
    fig, ax = plt.subplots(figsize=(25, 15))
    
    # 颜色和hatch配置
    colors = {
        'outputs': '#FF9E4A',  # 温暖橙色 - 输出
        'weights': '#7FBF7B',  # 柔和蓝色 - 权重
        'inputs': '#67A9CF'    # 清新绿色 - 输入
    }
    hatches = {'Original': None, 'Fusion': 'xx'}
    edge_color = 'black'  # 边框颜色
    edge_width = 0.5      # 边框宽度
    # 绘制Original堆叠（左柱，hatch='/'）
    ax.bar(idx - bar_width/2, orig_weights, width=bar_width, color=colors['weights'], hatch=hatches['Original'], label='Original_weights', bottom=0, edgecolor=edge_color, linewidth=edge_width)
    ax.bar(idx - bar_width/2, orig_inputs, width=bar_width, color=colors['inputs'], hatch=hatches['Original'], label='Original_inputs', bottom=orig_weights, edgecolor=edge_color, linewidth=edge_width)
    ax.bar(idx - bar_width/2, orig_outputs, width=bar_width, color=colors['outputs'], hatch=hatches['Original'], label='Original_outputs', bottom=orig_weights + orig_inputs, edgecolor=edge_color, linewidth=edge_width)
    
    # 绘制Fusion堆叠（右柱，hatch='\\'）
    ax.bar(idx + bar_width/2, fuse_weights, width=bar_width, color=colors['weights'], hatch=hatches['Fusion'], label='Fusion_weights', bottom=0, edgecolor=edge_color, linewidth=edge_width)
    ax.bar(idx + bar_width/2, fuse_inputs, width=bar_width, color=colors['inputs'], hatch=hatches['Fusion'], label='Fusion_inputs', bottom=fuse_weights, edgecolor=edge_color, linewidth=edge_width)
    ax.bar(idx + bar_width/2, fuse_outputs, width=bar_width, color=colors['outputs'], hatch=hatches['Fusion'], label='Fusion_outputs', bottom=fuse_weights + fuse_inputs, edgecolor=edge_color, linewidth=edge_width)
    
    # X轴标签（双排序字段换行显示）
    combined_labels = [f"{p1}\n{p2}" for p1, p2 in zip(df['Permutations_first'], df['permutations_second'])]  # 列A和B的列名
    ax.set_xticks(idx)
    ax.set_xticklabels(combined_labels, rotation=0, ha='center', fontsize=9)
    ax.set_xlabel('Permutations (First / Second)', fontsize=14)
    
    ax.set_ylabel('Access Count', fontsize=14)
    plt.title('Stacked Access Components Comparison', fontsize=18, pad=20)
    
    # 自定义图例（合并同变量的Original和Fusion）
    legend_elements = [
        Patch(color=colors['outputs'], hatch=hatches['Original'], label='Original Outputs'),
        Patch(color=colors['outputs'], hatch=hatches['Fusion'], label='Fusion Outputs'),
        Patch(color=colors['weights'], hatch=hatches['Original'], label='Original Weights'),
        Patch(color=colors['weights'], hatch=hatches['Fusion'], label='Fusion Weights'),
        Patch(color=colors['inputs'], hatch=hatches['Original'], label='Original Inputs'),
        Patch(color=colors['inputs'], hatch=hatches['Fusion'], label='Fusion Inputs'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right', ncol=2)  # 两列显示图例
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到 {save_path}")
    else:
        plt.show()
def plot_difference_heatmap(excel_file, save_path=None):
    """绘制Original和Fusion访问量差值的热力图"""
    # 读取数据并计算差值
    df = pd.read_excel(excel_file)
    #df['Difference'] = ((df['Original_access'] - df['Fusion_access'])/df['Original_access'])*100
    df['Difference'] = df['Original_access'] - df['Fusion_access']
    # 创建交叉表（以第一个排列为行，第二个排列为列）
    pivot_table = df.pivot_table(
        index='Permutations_first',
        columns='permutations_second',
        values='Difference',
        aggfunc='mean'  # 如果有重复组合，取平均值
    )
    
    # 检查是否有数据
    if pivot_table.empty:
        print("无法创建热力图：数据不足或格式不匹配。")
        return
    
    # 设置图形大小（根据数据量调整）
    plt.figure(figsize=(12, 10))
    
    # 绘制热力图
    cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 蓝红双色渐变
    ax = sns.heatmap(
        pivot_table, 
        annot=True,          # 显示数值
        fmt='.0f',           # 不显示小数
        cmap=cmap,           # 使用双色渐变
        center=0,            # 中间值设为0（差值正负分界）
        annot_kws={'size': 8},  # 调整注释字体大小
        linewidths=.5,       # 分隔线宽度
        cbar_kws={'label': 'Difference (Original - Fusion)'}  # 颜色条标签
    )
    
    # 设置标题和标签
    plt.title('Access Difference Heatmap (Original vs Fusion)', fontsize=16, pad=20)
    plt.xlabel('Permutations_Second', fontsize=12)
    plt.ylabel('Permutations_First', fontsize=12)
    
    # 旋转X轴标签以便阅读
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示图形
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热力图已保存到 {save_path}")
    else:
        plt.show()
