import os
import sys
# fmt: off
THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(THIS_SCRIPT_DIR, '..')))
from _tests import scripts
from scripts.notebook_utils import *
import joblib
import matplotlib.pyplot as plt
import numpy as np
THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_NAME = os.path.basename(THIS_SCRIPT_DIR)
DNN = "resnet18"
# =============================================================================
# Change this DNN to explore different DNNs!
# =============================================================================

layers = [f for f in os.listdir(f"/home/fufu/Desktop/accelergy-timeloop-infrastructure/cimloop/workspace/models/workloads/{DNN}") if f != "index.yaml" and f.endswith(".yaml")]
layers = sorted(layers)
print(f"Running: ", end="")

# Parallel/Delayed is used to multiprocess. Equivalent to running the following
# code in serial:
# results = [run_layer(DNN, layer.split(".")[0]) for layer in layers]

results = joblib.Parallel(n_jobs=10)(
    joblib.delayed(quick_run)(chip= 'pipeline_origin',macro=MACRO_NAME,tile="isaac",dnn=DNN, layer=layer.split(".")[0]) for layer in layers
)
energy = []
'''
for i in results :
    energy.append(i.energy)
print(energy)

print("")
for r in results:
    r.clear_zero_energies()

# Display an energy breakdown for each layer as a bar chart. Display normalized per-MAC and per-layer energy.
fig, ax = plt.subplots(figsize=(20, 5))

bar_stacked(
    {i: r.per_compute("per_component_energy") * 1e15 for i, r in enumerate(results)},
    title="",
    xlabel="Layer",
    ylabel="Energy (fJ/MAC)",
    ax=ax,
)

bar_stacked(
    {i: r.tops_1b for i, r in enumerate(results)},
    title="",
    xlabel="Layer",
    ylabel="Throughput (1b TOPS)",
    ax=ax,
)


results =[1.11e+03,6.44e+03,3.38e+03,5.89e+03,2.90e+03,5.97e+03,5.99e+03,2.97e+03,6.03e+03,6.05e+03,1.50e+03,1.51e+03,1.51e+03]
labels = [f"CONV{i}" for i in range(1, len(results) + 1)]
# 设置图形大小
plt.figure(figsize=(20, 5))

# 绘制柱状图
plt.bar(labels, results)

# 设置纵轴标签
plt.ylabel('Energy (uJ)')
# 设置横轴标签
plt.xlabel('') 
# 设置图表标题
plt.title('ISAAC energy for VGG-16') 

# 显示图形
plt.show()

plt.show()
'''