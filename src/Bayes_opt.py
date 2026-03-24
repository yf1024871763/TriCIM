import numpy as np
import GPyOpt
import pandas as pd
import os
import glob
import logging


class Bayesian_Optimizer:
    """
    Bayesian optimization engine.
    Preserves legacy constraint logic and supports dynamic grouped constraints.
    """

    def __init__(
        self,
        bounds,
        evaluate_callback,
        n_calls=30,
        dnn_name="unknown",
        alpha=0.2,
        tile_num=32,
        layers=None,
        head_num=1,
        block_num=1,
        transformer=False,
        max_block=1,
        multi_layer=False,
        batch=False,
        grouped_indices=None,
        var_map=None,
        candidate_domains=None,
    ):

        self.bounds = bounds
        self.evaluate_callback = evaluate_callback
        self.n_calls = n_calls
        self.dnn_name = dnn_name
        self.alpha = alpha

        self.tile_num = tile_num
        self.layers = layers if layers is not None else []
        self.head_num = head_num
        self.block_num = block_num
        self.transformer = transformer
        self.max_block = max_block
        self.multi_layer = multi_layer
        self.batch = batch
        self.grouped_indices = grouped_indices
        self.candidate_domains = (
            candidate_domains if candidate_domains is not None else []
        )

        self.convergence_history = []
        self.var_map = var_map if var_map is not None else {}

    def run_optimization(self):
        if not self.bounds:
            raise ValueError("Bounds must be provided for optimization.")

        domain = []
        for i, (low, high) in enumerate(self.bounds):
            low_i, high_i = int(low), int(high)
            low_i, high_i = min(low_i, high_i), max(low_i, high_i)

            values = []
            if self.candidate_domains and i < len(self.candidate_domains):
                raw_vals = self.candidate_domains[i] or []
                values = sorted({int(v) for v in raw_vals if low_i <= int(v) <= high_i})
                if not values:
                    values = sorted({int(v) for v in raw_vals})
            if not values:
                values = list(range(low_i, high_i + 1))

            domain.append(
                {"name": f"x{i}", "type": "discrete", "domain": tuple(values)}
            )

        # ==========================================
        # ==========================================
        constraints = []

        if self.grouped_indices is not None and len(self.grouped_indices) > 1:
            logging.info("BO Enforcing Dynamic Constraints (Shared Head Tiles).")
            for stage_idx, group in enumerate(self.grouped_indices):
                if len(group) > 1:
                    terms = [
                        f"x[:, {self.var_map[i] if self.var_map else i}]" for i in group
                    ]
                    term_str = " + ".join(terms)
                    constraint_str = f"{term_str} - {self.tile_num} <= 0"
                    constraints.append(
                        {
                            "name": f"stage_{stage_idx}_limit",
                            "constraint": constraint_str,
                        }
                    )
                    logging.info(f"   - Group {stage_idx+1}: {constraint_str}")
        else:
            need_constraint = True
            if self.multi_layer:
                if self.transformer:
                    if self.batch:
                        constraint_str = f"(x[:, 0] + x[:, 3]) * {self.head_num} + 2 * x[:, 1] + 3 * x[:, 2] + x[:, 4] - {self.tile_num}"
                    else:
                        constraint_str = f"(x[:, 0] + x[:, 3]) * {self.head_num} + 2 * x[:, 1] + 3 * x[:, 2] + x[:, 4] - {self.tile_num/self.max_block}"
                else:
                    need_constraint = False
            else:
                if self.transformer:
                    try:
                        a_idx = self.layers.index("A")
                        z0_idx = self.layers.index("Z0")
                    except ValueError:
                        pass
                    constraint_str = f"np.sum(x, axis=1) + ({self.head_num - 1})*(x[:, 0] + x[:, 3]) - {self.tile_num/self.block_num}"
                else:
                    constraint_str = f"np.sum(x, axis=1) - {self.tile_num}"

            if need_constraint:
                constraints = [
                    {"name": "tile_constraint", "constraint": constraint_str}
                ]
                logging.info(f"BO Enforcing Legacy Constraint: {constraint_str} <= 0")
            else:
                logging.info("BO Enforcing No Constraints.")

        # ==========================================

        def objective_function(x_array):
            results = []
            for individual in x_array:
                fitness = self.evaluate_callback(individual)
                results.append(fitness)
            return np.array(results).reshape(-1, 1)

        logging.info(f"=== Starting Bayesian Optimization Initialization ===")
        optimizer = GPyOpt.methods.BayesianOptimization(
            f=objective_function,
            domain=domain,
            constraints=constraints,
            acquisition_type="EI",
            acquisition_weight=2,
            initial_design_numdata=10,
            random_state=42,
            feasible_region=True,
        )

        patience = 20
        no_improve_count = 0
        best_fx_so_far = float("inf")

        for step in range(self.n_calls - 10):
            optimizer.run_optimization(max_iter=1, eps=0)

            current_y = optimizer.Y[-1][0]
            current_best_fx = optimizer.fx_opt
            current_best_x = optimizer.x_opt

            logging.info(f"\n=== Completed Iteration {step+1} ===")
            logging.info(f"Current Best Params: {current_best_x}")
            logging.info(f"Current Best Fitness: {current_best_fx}")

            self.convergence_history.append(
                {
                    "Iteration": step + 1,
                    "Current_Sample_Latency": current_y,
                    "Best_Latency_So_Far": current_best_fx,
                }
            )

            if current_best_fx < best_fx_so_far:
                best_fx_so_far = current_best_fx
                no_improve_count = 0
                logging.info(">> 🚀 Found new global optimum!")
            else:
                no_improve_count += 1
                logging.info(
                    f">> ⚠️ Optimum not improved ({no_improve_count}/{patience})"
                )

            if no_improve_count >= patience:
                logging.info(
                    f"\n[Early Stopping] No improvement for {patience} iterations. Stopping."
                )
                break

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
