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
├── main.py                     # The smart dispatch center: assesses capacity vs. workload.
├── configs/
│   └── default.yaml            # Centralized hardware, model, and path configurations.
├── src/
│   ├── engine.py               # Orchestrator handling resource allocation and scheduling.
│   ├── Bayes_opt.py            # Pure Bayesian Optimization engine (GPyOpt).
│   ├── fitness.py              # Evaluator bridging BO constraints and hardware simulations.
│   ├── ParallelExecutor.py     # Multi-process execution for Timeloop/Accelergy tasks.
│   ├── pipeline_analyzer.py    # Parses dataspaces, calculates pipeline bubbles and strides.
│   ├── noc.py                  # BookSim 2.0 wrapper for NoC congestion modeling.
│   └── function.py             # Heuristics for greedy tile allocation and subgraph grouping.
```
## 🛠️ Prerequisites
To run the TriCIM framework, ensure the following dependencies are installed:

* Python 3.8+

* Python Libraries: numpy, pandas, PyYAML, GPyOpt, matplotlib

Hardware Evaluation Backends:

* Timeloop & Accelergy: Required for cycle-accurate CIM array simulation and energy estimation.

* BookSim 2.0: Required for cycle-level Network-on-Chip (NoC) routing and latency evaluation.

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
  timeloop_scripts: "/path/to/timeloop/scripts"
  arch_root: "/path/to/models/arch"
  workload_root: "/path/to/models/workloads"
  output_root: "/path/to/outputs"
```
### 2. Launch the Evaluator
Run the main entry script. TriCIM will automatically parse the workload, determine the optimal stationary region (MS, LS, or TS), and initiate Bayesian Optimization for pipeline mapping.
```bash
python main.py --config configs/default.yaml
```
### 3. Analyze Results
The framework outputs a clean, hierarchical log to the console detailing the parameter space exploration, current optimums, and Early Stopping triggers, alongside final energy and latency metrics.
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