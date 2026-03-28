import argparse
import yaml
import logging
import time
import sys
from src.engine import TriCIMEngine, resolve_config_paths
from src.allocation import TileAllocator
from src.allocation import allocation_utils

# Configure standard logging format
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path):
    with open(config_path, "r") as f:
        return resolve_config_paths(yaml.safe_load(f))


def build_legal_tiles_layer_map(engine, include_ceil_equivalence=False):
    workload = [engine.analyzer.get_workload(layer) for layer in engine.layers]
    compute = [
        wl.get("C", 1)
        * wl.get("M", 1)
        * wl.get("R", 1)
        * wl.get("S", 1)
        * wl.get("P", 1)
        * wl.get("Q", 1)
        for wl in workload
    ]
    min_tiles = [
        min(
            allocation_utils.tile_allocation(
                wl,
                macro_num=engine.hw["macro_num"],
                core_num=engine.hw["core_num"],
                array_col=engine.hw["array_col"],
                array_row=engine.hw["array_row"],
                cim_depth=engine.hw["cim_depth"],
                precision=engine.hw["precision"],
            ),
            engine.hw["tile_num"],
        )
        for wl in workload
    ]

    allocator = TileAllocator(
        layers=engine.layers,
        compute_workloads=compute,
        min_tiles_per_layer=min_tiles,
        max_tiles_per_group=engine.hw["tile_num"],
        min_layers_per_group=1,
        max_layers_per_group=16,
        workload_shape=workload,
    )

    layer_map = allocator.build_legal_tiles_map(
        max_tile=engine.hw["tile_num"],
        include_divisors=True,
        include_ceil_equivalence=include_ceil_equivalence,
        include_powers_of_two=True,
    )
    summary = allocator.summarize_legal_tiles(layer_map, max_tile=engine.hw["tile_num"])

    avg_ratio = (
        sum(v["compression_ratio"] for v in summary.values()) / len(summary)
        if summary
        else 1.0
    )
    return layer_map, summary, avg_ratio


def main():
    parser = argparse.ArgumentParser(
        description="TriCIM Architecture Evaluation Engine"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Dynamically append timeloop scripts path
    scripts_path = config.get("paths", {}).get("timeloop_scripts", "")
    if scripts_path and scripts_path not in sys.path:
        sys.path.append(scripts_path)

    logging.info(f"Initializing TriCIM Engine for DNN: {config['model']['dnn']}")
    engine = TriCIMEngine(config)
    (
        engine.legal_tiles_layer_map,
        engine.legal_tiles_summary,
        engine.legal_tiles_avg_compression,
    ) = build_legal_tiles_layer_map(
        engine,
        include_ceil_equivalence=False,
    )

    # =====================================================================
    # 1. Capacity vs Workload Assessment
    # =====================================================================
    weights = []
    for layer in engine.layers:
        wl = engine.analyzer.get_workload(layer)
        if "R" in wl.keys() or "S" in wl.keys():
            w_size = wl.get("C", 1) * wl.get("M", 1) * wl.get("R", 1) * wl.get("S", 1)
        else:
            w_size = wl.get("C", 1) * wl.get("M", 1)
        weights.append(w_size)

    is_transformer = config["model"].get("transformer", False)
    head_num = config["model"].get("head_num", 1)
    block = config["model"].get("block", 1)
    precision = config["hardware"].get("precision", 16)

    # Apply multi-head attention replication factor for Transformers
    if is_transformer:
        try:
            # Dynamically find indices for A and Z0 to avoid hardcoding
            a_idx = engine.layers.index("A")
            z0_idx = engine.layers.index("Z0")
            weights[a_idx] *= head_num
            weights[z0_idx] *= head_num
        except ValueError:
            logging.warning("Layers 'A' or 'Z0' not found during weight calculation.")

    effective_block = block if is_transformer else 1
    total_workload_bits = sum(weights) * effective_block * precision

    hw = config["hardware"]
    arch_size_bits = (
        hw["tile_num"]
        * hw["macro_num"]
        * hw["core_num"]
        * hw["array_col"]
        * hw["array_row"]
        * hw["cim_depth"]
    )

    logging.info(
        f"Arch size(bit) = {arch_size_bits} | Workload size(bit) = {total_workload_bits}"
    )
    layer_bits = [w * precision for w in weights]
    can_fit_adjacent_pair = any(
        layer_bits[i] + layer_bits[i + 1] <= arch_size_bits
        for i in range(len(layer_bits) - 1)
    )

    if len(layer_bits) <= 1:
        can_fit_adjacent_pair = False

    weights = [
        w * precision for w in weights
    ]  # Scale weights by block factor for grouping
    if is_transformer:
        grouped_indices = allocation_utils.capacity_aware_transformer_grouping(
            weights=weights,
            layer_names=engine.layers,
            arch_size=arch_size_bits,
            ops_per_block=8,
        )

        # =====================================================================
        # 2. Smart Execution Routing
        # =====================================================================
    start_time = time.time()

    if not can_fit_adjacent_pair:
        logging.info(
            "💡 [Decision] Hardware cannot keep two adjacent operators resident. Routing to one-tile serial evaluation."
        )
        engine.run_one_tile_evaluation()
    elif arch_size_bits >= total_workload_bits:
        logging.info(
            "💡 [Decision] Arch Size >= Workload. Routing to Basic Pipeline Evaluation."
        )
        if is_transformer:
            engine.run_transformer_evaluation()
        else:
            batch_size = config["model"].get("batch_size", 1)
            if batch_size > 1:
                logging.info(
                    "💡 [Routing] CNN batch_size > 1 detected. Running explicit multi-batch pipeline evaluation..."
                )
                engine.run_multi_batch_cnn_pipeline(batch_size=batch_size)
            else:
                engine.run_cnn_evaluation()
    else:
        logging.info(
            "💡 [Decision] Arch Size < Workload. Routing to Capacity-Aware Evaluation."
        )
        if is_transformer:
            group_count = len(grouped_indices)
            if group_count == 1:
                logging.info(
                    "💡 [Routing] Single group (Level 1) detected. Running standard Multi-Layer BO Optimization..."
                )
                engine.run_multi_layer_transformer_batch(
                    batch_size=config["model"].get("batch_size", 1)
                )
            else:
                logging.info(
                    f"💡 [Routing] {group_count} groups detected! Routing to Multi-Group Sequential BO Optimization..."
                )
                engine.run_multi_layer_transformer_batch_group(
                    batch_size=config["model"].get("batch_size", 1),
                    grouped_indices=grouped_indices,
                )
        else:
            logging.info("Running Multi-Layer Grouping with BO Optimization for CNN...")
            engine.run_multi_layer_cnn(
                batch_size=config["model"].get("batch_size", 1)
            )

    end_time = time.time()
    total_seconds = end_time - start_time
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60

    logging.info(f"==================================================")
    logging.info(f"🎉 Total Execution Time: {minutes} min {seconds:.2f} sec")
    logging.info(f"==================================================")


if __name__ == "__main__":
    main()
