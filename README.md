# TriCIM: A General CIM-Capacity-Aware Framework for Optimizing Dataflows

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **An end-to-end framework for optimizing Model, Layer, and Tile-Stationary dataflows in Compute-in-Memory (CIM) accelerators.**

TriCIM is an advanced Design Space Exploration (DSE) and optimization framework designed to maximize the performance and energy efficiency of Compute-in-Memory (CIM) architectures. 

Unlike traditional frameworks that are constrained to specific hardware scales or limited dataflow types, TriCIM introduces a **CIM-capacity-aware formulation**. It dynamically classifies the execution space into three distinct regions—**Model-Stationary (MS)**, **Layer-Stationary (LS)**, and **Tile-Stationary (TS)**—and automatically applies the optimal spatial and temporal mapping strategies, including critical CIM-specific features like weight-update scheduling and Network-on-Chip (NoC) congestion modeling.

## 🌟 Key Contributions & Features

* **Tri-Regional Capacity Awareness:** Automatically assesses hardware capacity versus workload size to route execution into MS, LS, or TS optimization spaces.
* **CIM-Specific Scheduling:** Accurately models the overhead of weight updates and feature map transfers, ensuring realistic end-to-end latency and energy evaluations.
* **Intelligent Design Space Exploration:** Integrates a purely decoupled **Bayesian Optimization (BO)** engine to navigate the massive mapping space and find global optimums for tile allocations and sub-block groupings.
* **Comprehensive Topology Support:** Out-of-the-box support for Deep Convolutional Neural Networks (CNNs) and complex multi-batch Vision/Language Transformers (Attention & FFN pipelines).


## 📂 Framework Architecture

TriCIM is built with a highly decoupled, modular software architecture, bridging analytical optimization algorithms with cycle-level hardware simulators.

```text
TriCIM/
├── main.py                     # The smart dispatch center: assesses capacity vs. workload and routes execution.
├── Makefile                    # Installation/build helpers for dependencies and BookSim.
├── requirements.txt            # Python dependencies (e.g., PyYAML, GPyOpt, numpy, pandas).
├── LICENSE                     # MIT License.
├── README.md                   # Project documentation.
├── configs/
│   └── default.yaml            # Centralized hardware, model, path, and BO configurations.
├── cimloop/                    # Integrated Timeloop/Accelergy workspace for CIM evaluation.
├── booksim2/                   # Integrated BookSim 2.0 for cycle-level NoC simulation.
├── outputs/                    # Generated plots and intermediate output artifacts.
└── src/
    ├── allocation/
    │   ├── allocation_utils.py # Greedy allocation, legal-tile handling, and grouping helpers.
    │   ├── parallel_executor.py# Multi-process execution for Timeloop/Accelergy mapping tasks.
    │   └── tile_allocator.py   # Explores valid multi-layer grouping/allocation candidates.
    ├── analysis/
    │   ├── analyzer.py         # Extracts cycles, energy, utilization, and tensor traffic from outputs.
    │   └── pipeline_analyzer.py# Parses dataspaces and computes pipeline overlap/bubble timing.
    ├── engine/
    │   ├── __init__.py         # Public package exports for the engine layer.
    │   ├── config.py           # Resolves workspace/output paths from the YAML config.
    │   ├── core.py             # Main orchestrator for analyzers, optimizers, and execution runners.
    │   └── types.py            # Shared lightweight data containers for engine internals.
    ├── engine_runners/
    │   ├── cnn_runner.py       # CNN execution flows: basic pipeline, multi-batch pipeline, and multi-layer mode.
    │   └── transformer_runner.py # Transformer execution flows: pipeline, grouping, and multi-batch scheduling.
    ├── noc/
    │   └── booksim_interface.py# BookSim wrapper for NoC congestion and transfer-latency modeling.
    ├── optimization/
    │   ├── bayes_optimizer.py  # Bayesian Optimization engine wrapper.
    │   └── fitness.py          # Evaluator bridging BO candidates and hardware simulations.
    └── visualization/
        └── timeline_plot.py    # Visualization tools for pipeline bubbles and computation timelines.
```
## 🛠️ Prerequisites
To run the TriCIM framework, ensure the following dependencies are installed:

* **Python 3.8+**

* **Python Libraries: numpy, pandas, PyYAML, GPyOpt, matplotlib**
* **System Dependencies: A C++ compiler (e.g., g++ or clang) and make to build the integrated simulators.**

Hardware Evaluation Backends:

* **Cimloop (Timeloop & Accelergy):** Included in the repository. Required for cycle-accurate CIM array simulation and energy estimation.

* **BookSim 2.0**: Required for cycle-level Network-on-Chip (NoC) routing and latency evaluation.

## 🚀 Quick Start
### 1. Configure the Workspace  
Define your hardware topology, target DNN model, and local dependency paths in configs/default.yaml:
```yaml
model:
  dnn: "resnet18"
  transformer: false
  batch_size: 1

hardware:
  tile_num: 1344
  macro_num: 12
  core_num: 8
  precision: 16

paths:
  workspace_root: "/path/to/cimloop/workspace"
  arch_name: "isaac"
  macro_name: "isaac_isca_2016"
  plot_root: "/path/to/TriCIM/outputs"

optimization:
  bayes:
    alpha: 0.2
    max_calls: 100
    initial_points: 10
    early_stop_patience: 20
    acquisition_weight: 2
    random_state: 42
```
`workspace_root` is the main CIMLoop path you need to provide. TriCIM derives `timeloop_scripts`, `workload_root`, `output_root`, and the fixed `arch_root=models/arch/3_chip` from it. The `optimization.bayes` section controls BO hyperparameters such as initial points, max calls, and early stopping patience.

At minimum, the `hardware` section should provide:
```yaml
hardware:
  tile_num: 1344
  macro_num: 12
```
Other hardware fields such as `core_num`, `array_col`, `array_row`, `cim_depth`, and `precision` are automatically filled with `1` if omitted. For convenience, TriCIM also provides ready-to-run architecture presets under `configs/macros/`.

### 2. Install Dependencies
Use the provided Makefile to install Python dependencies, Accelergy/Timeloop support, and build BookSim:
```bash
make install
```

### 3. Launch the Evaluator
Run the main entry script. TriCIM will automatically parse the workload, determine the optimal stationary region (MS, LS, or TS), and initiate Bayesian Optimization for pipeline mapping.
```bash
make run
```
You can also run it directly:
```bash
python3 main.py --config configs/default.yaml
```

To run a specific macro preset:
```bash
python3 main.py --config configs/macros/raella_isca_2023.yaml
python3 main.py --config configs/macros/jia_jssc_2020.yaml
python3 main.py --config configs/macros/wang_vlsi_2022.yaml
```

### 4. Analyze Results
The framework outputs a clean, hierarchical log to the console detailing the parameter space exploration, current optimums, and Early Stopping triggers, alongside final energy and latency metrics.

## 🧱 Supported Macro Presets

TriCIM currently includes preset configs for the following macros under `configs/macros/`. These presets are derived from the corresponding CiMLoop `arch/1_macro/*` definitions, using the macro spatial factors in `arch.yaml` and CiM depth-related variables in `variables_free.yaml`.

```text
configs/macros/
├── albireo_isca_2021.yaml     # arch_name=isaac,  macro_name=albireo_isca_2021
├── basic_analog.yaml          # arch_name=isaac,  macro_name=basic_analog
├── colonnade_jssc_2021.yaml   # arch_name=isaac,  macro_name=colonnade_jssc_2021
├── example_traditional.yaml   # arch_name=isaac,  macro_name=example_traditional
├── isaac_isca_2016.yaml       # arch_name=isaac,  macro_name=isaac_isca_2016
├── jia_jssc_2020.yaml         # arch_name=isaac,  macro_name=jia_jssc_2020
├── pimca.yaml                 # arch_name=pimca,  macro_name=pimca
├── raella_isca_2023.yaml      # arch_name=raella, macro_name=raella_isca_2023
├── sinangil_jssc_2021.yaml    # arch_name=isaac,  macro_name=sinangil_jssc_2021
├── wan_nature_2022.yaml       # arch_name=isaac,  macro_name=wan_nature_2022
└── wang_vlsi_2022.yaml        # arch_name=isaac,  macro_name=wang_vlsi_2022
```

Recommended hardware values extracted from CiMLoop:

```text
Macro                    array_col  array_row  cim_depth  precision
albireo_isca_2021        5          3          1          8
basic_analog             32         32         1          8
colonnade_jssc_2021      128        8          1          8
example_traditional      14         12         1          8
isaac_isca_2016          128        128        1          8
jia_jssc_2020            192        768        1          8
pimca                    128        256        1          8
raella_isca_2023         512        512        1          8
sinangil_jssc_2021       16         64         1          4
wan_nature_2022          256        256        1          8
wang_vlsi_2022           16         64         8          8
```

Notes:
* `precision` is standardized to `8` for most presets, with `sinangil_jssc_2021` kept at `4` because its supported bitwidth in CiMLoop is 4-bit.
* `array_col` and `array_row` are inferred from the effective macro spatial factors in the CiMLoop macro descriptions.
* `cim_depth` is inferred from the CiMLoop macro variables, typically `CIM_UNIT_DEPTH_CELLS`.

## 📊 Result Showcase

Example Transformer multi-layer scheduling result on `bert`:

![Transformer Multi-Layer Timeline](outputs/ISAAC-bert/Combined_Timelines_Block_Batch.svg)

Final performance summary:

```text
Cal-only Time: 104621.0
Weight-only Time: 6912.0
Overlap Time: 324864.0

Tile allocation = [3, 36, 36, 12, 12, 12, 1, 12]
Cycles (Pipeline) = 436654.00
Weight update cycles (Pipeline) = 331776.00
Overlap cycles (Pipeline) = 324864.00
Energy (Pipeline) = 1703534.7734 pJ

Pipeline Compute Energy = 1499305.3851
Pipeline Feature Update Energy = 76866.6422
Pipeline Weight Update Energy = 127362.7461
```

This example illustrates TriCIM's ability to expose the interaction between computation, weight movement, and pipeline overlap in multi-layer Transformer execution.
## 📜 Citation
If you use TriCIM in your research, please cite our paper:
```
@article{tricim2026,
  title={TriCIM: A General CIM-Capacity-Aware Framework for Optimizing Model, Layer, and Tile-Stationary Dataflows in CIM Accelerators},
  author={Wang, Jin and Zhang, Yufu and Lin, Longyang},
  journal={IEEE Journal of ... (Update with publication details)},
  year={2026}
}
```
## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
