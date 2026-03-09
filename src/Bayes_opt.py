import numpy as np
import GPyOpt
import pandas as pd
import os
import glob
import logging

class Bayesian_Optimizer:
    """
    贝叶斯优化引擎。
    内部保留了原版的约束判断逻辑 (Constraints Logic)，并调用外部传入的 Callback 进行评估。
    """
    def __init__(self, bounds, evaluate_callback, n_calls=30, dnn_name='unknown', alpha=0.2,
                 tile_num=32, layers=None, head_num=1, block_num=1, 
                 transformer=False, max_block=1, multi_layer=False, batch=False):
        
        self.bounds = bounds
        self.evaluate_callback = evaluate_callback  # 外部传入的评估函数（CNN或Transformer的Callback）
        self.n_calls = n_calls
        self.dnn_name = dnn_name
        self.alpha = alpha
        
        # --- 保留原版的约束控制参数 ---
        self.tile_num = tile_num
        self.layers = layers if layers is not None else []
        self.head_num = head_num
        self.block_num = block_num
        self.transformer = transformer
        self.max_block = max_block
        self.multi_layer = multi_layer
        self.batch = batch

        self.convergence_history = []
        
    def run_optimization(self):
        if not self.bounds:
            raise ValueError("Bounds must be provided for optimization.")
        
        # 1. 定义参数空间 (注意：变量名严格映射为 x0, x1, x2...)
        domain = [
            {'name': f'x{i}', 'type': 'discrete', 'domain': tuple(range(int(low), int(high) + 1))}
            for i, (low, high) in enumerate(self.bounds)
        ]
        
        # ==========================================
        # 2. 原版的约束生成逻辑 (Constraints)
        # ==========================================
        need_constraint = True
        if self.multi_layer:
            if self.transformer:
                if self.batch:
                    constraint_str = (
                    f"(x[:, 0] + x[:, 3]) * {self.head_num} + 2 * x[:, 1] + 3 * x[:, 2] + x[:, 4] - {self.tile_num}")
                else:
                    constraint_str = (
                    f"(x[:, 0] + x[:, 3]) * {self.head_num} + 2 * x[:, 1] + 3 * x[:, 2] + x[:, 4] - {self.tile_num/self.max_block}")
            else:
                need_constraint = False
        else:
            if self.transformer:
                # 避免找不到 'A' 的异常
                try:
                    a_idx = self.layers.index('A')
                    z0_idx = self.layers.index('Z0')
                except ValueError:
                    pass 
                constraint_str = (
                    f"np.sum(x, axis=1) + ({self.head_num - 1})*(x[:, 0] + x[:, 3]) - {self.tile_num/self.block_num}"
                )
            else:
                constraint_str = f"np.sum(x, axis=1) - {self.tile_num}"

        if need_constraint:
            constraints = [{'name': 'tile_constraint', 'constraint': constraint_str, 'type': 'ineq'}]
            #logging.info(f"BO Enforcing Constraint: {constraint_str}")
        else:
            constraints = []  # 没有约束
            logging.info("BO Enforcing No Constraints.")

        # ==========================================
        
        # 3. 包装目标函数 (调用传进来的 Callback)
        def objective_function(x_array):
            results = []
            for individual in x_array:
                fitness = self.evaluate_callback(individual)
                results.append(fitness)
            return np.array(results).reshape(-1, 1)

        # 4. 创建 GPyOpt 优化器并挂载 constraints
        optimizer = GPyOpt.methods.BayesianOptimization(
            f=objective_function,
            domain=domain,
            constraints=constraints,
            acquisition_type='EI',
            acquisition_weight=2,
            initial_design_numdata=10,
            random_state=42,
            feasible_region=True  # 确保只在可行区域内采样
        )

        patience = 20  # 提前停止容忍度
        no_improve_count = 0
        best_fx_so_far = float('inf')

        # 5. 迭代循环
        for step in range(self.n_calls - 10):
            optimizer.run_optimization(max_iter=1, eps=0)
            
            current_y = optimizer.Y[-1][0] 
            current_best_fx = optimizer.fx_opt
            current_best_x = optimizer.x_opt
            
            logging.info(f"\n=== Completed Iteration {step+1} ===")
            logging.info(f"Current Best Params: {current_best_x}")
            logging.info(f"Current Best Fitness: {current_best_fx}")
            
            self.convergence_history.append({
                'Iteration': step + 1,
                'Current_Sample_Latency': current_y,
                'Best_Latency_So_Far': current_best_fx
            })
        
            # Early Stopping 判断
            if current_best_fx < best_fx_so_far:
                best_fx_so_far = current_best_fx
                no_improve_count = 0
                logging.info(">> 🚀 Found new global optimum!")
            else:
                no_improve_count += 1
                logging.info(f">> ⚠️ Optimum not improved ({no_improve_count}/{patience})")
                
            if no_improve_count >= patience:
                logging.info(f"\n[Early Stopping] No improvement for {patience} iterations. Stopping.")
                break
                    
        # 6. 保存结果
        self._save_results()
        
        logging.info("\nOptimization Finished!")
        logging.info(f"Optimal Parameters: {optimizer.x_opt}")
        logging.info(f"Optimal Fitness: {optimizer.fx_opt}")
        
        return optimizer, step

    def _save_results(self):
        df = pd.DataFrame(self.convergence_history)
        save_dir = "bo_sensitivity_results"
        os.makedirs(save_dir, exist_ok=True)

        base_name = f"{self.dnn_name}_convergence_alpha_{self.alpha}"
        existing_files = glob.glob(os.path.join(save_dir, f"{base_name}_run_*.xlsx"))
        run_id = len(existing_files) + 1
        
        excel_filename = os.path.join(save_dir, f"{base_name}_run_{run_id}.xlsx")
        df.to_excel(excel_filename, index=False) 
        logging.info(f"\n📊 Data saved to: {excel_filename}")