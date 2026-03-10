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
def run_layer(dnn: str, layer: str):
    print(f"{layer} ", end="", flush=True)
    spec = get_spec(
        macro="jia_jssc_2020",
        # Weight-stationary, dummy buffer on top to hold inputs & outputs, many
        # macros to ensure we can fit all weights.
        system="ws_dummy_buffer_many_macro",
        # Set the DNN and layer
        dnn=dnn,
        layer=layer,
    )
    spec.architecture.name2leaf("macro").attributes["has_power_gating"] = True
    # Do NOT generate a maximum-utilization workload; we're running a DNN
    # workload.
    #spec.variables["MAX_UTILIZATION"] = False

    # Use a larger array to ensure we can fit all weights
    #spec.architecture.find("column").spatial.meshX = 64
    #spec.architecture.find("row").spatial.meshY = 512
    return run_mapper(spec)


# =============================================================================
# Change this DNN to explore different DNNs!
# =============================================================================
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
results = joblib.Parallel(n_jobs=None)(
    joblib.delayed(run_layer)(DNN, layer.split(".")[0]) for layer in layers
)
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
plt.show()
'''
bar_stacked(
    {i: r.tops_1b for i, r in enumerate(results)},
    title="",
    xlabel="Layer",
    ylabel="Throughput (1b TOPS)",
    ax=ax,
)
plt.show()
'''
