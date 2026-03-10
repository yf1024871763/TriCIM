import argparse
import yaml
import logging
import time
import sys
from src.engine import TriCIMEngine

# Configure standard logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="TriCIM Architecture Evaluation Engine")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Dynamically append timeloop scripts path
    scripts_path = config.get('paths', {}).get('timeloop_scripts', '')
    if scripts_path and scripts_path not in sys.path:
        sys.path.append(scripts_path)

    logging.info(f"Initializing TriCIM Engine for DNN: {config['model']['dnn']}")
    engine = TriCIMEngine(config)
    
    # =====================================================================
    # 1. Capacity vs Workload Assessment
    # =====================================================================
    weights = []
    for layer in engine.layers:
        wl = engine.analyzer.get_workload(layer)
        if 'R' in wl.keys() or 'S' in wl.keys():
            w_size = wl.get('C', 1) * wl.get('M', 1) * wl.get('R', 1) * wl.get('S', 1)
        else:
            w_size = wl.get('C', 1) * wl.get('M', 1)
        weights.append(w_size)
        
    is_transformer = config['model'].get('transformer', False)
    head_num = config['model'].get('head_num', 1)
    block = config['model'].get('block', 1)
    precision = config['hardware'].get('precision', 16)
    
    # Apply multi-head attention replication factor for Transformers
    if is_transformer:
        try:
            # Dynamically find indices for A and Z0 to avoid hardcoding
            a_idx = engine.layers.index('A')
            z0_idx = engine.layers.index('Z0')
            weights[a_idx] *= head_num
            weights[z0_idx] *= head_num
        except ValueError:
            logging.warning("Layers 'A' or 'Z0' not found during weight calculation.")
            
    total_workload_bits = sum(weights) * block * precision
    
    hw = config['hardware']
    arch_size_bits = (hw['tile_num'] * hw['macro_num'] * hw['core_num'] * hw['array_col'] * hw['array_row'] * hw['cim_depth'])
    
    logging.info(f"Arch size(bit) = {arch_size_bits} | Workload size(bit) = {total_workload_bits}")
    
    # =====================================================================
    # 2. Smart Execution Routing
    # =====================================================================
    start_time = time.time()
    
    if arch_size_bits >= total_workload_bits:
        logging.info("💡 [Decision] Arch Size >= Workload. Routing to Basic Pipeline Evaluation.")
        if is_transformer:
            engine.run_transformer_evaluation()
        else:
            engine.run_cnn_evaluation()
    else:
        logging.info("💡 [Decision] Arch Size < Workload. Routing to Multi-Layer Grouping Evaluation.")
        if is_transformer:
            logging.info("Running Multi-Layer Grouping with BO Optimization for Transformer...")
            engine.run_multi_layer_transformer_batch(batch_size=config['model'].get('batch_size', 1))
        else:
            logging.info("Running Multi-Layer Grouping with BO Optimization for CNN...")
            engine.construct_allocation_space(batch_size=config['model'].get('batch_size', 1))
            
    end_time = time.time()
    total_seconds = end_time - start_time
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    
    logging.info(f"==================================================")
    logging.info(f"🎉 Total Execution Time: {minutes} min {seconds:.2f} sec")
    logging.info(f"==================================================")

if __name__ == "__main__":
    main()