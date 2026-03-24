import os
import math
import logging
import copy
import time
from src.pipeline_analyzer import PipelineAnalyzer
from src.ParallelExecutor import ParallelExecutor
from src.Bayes_opt import Bayesian_Optimizer
import src.function as function
import src.plot as plot
from src.fitness import FitnessEvaluator
from src.noc import BookSimInterface


class TriCIMEngine:
    def __init__(self, config):
        self.config = config
        self.hw = config["hardware"]
        self.model = config["model"]
        self.paths = config["paths"]

        self.dnn = self.model["dnn"]
        self.layers = []  # Will be populated by pipeline_analyzer
        self.legal_tiles_layer_map = {}
        self.legal_tiles_summary = {}
        self.legal_tiles_avg_compression = 1.0

        # Initialize the underlying parser and analyzer
        self.pipeline_analyzer = PipelineAnalyzer(config)
        self.layers = self.pipeline_analyzer.layers
        self.analyzer = (
            self.pipeline_analyzer.analyzer
        )  # Access to underlying Timeloop analysis
        try:
            booksim_path = config["paths"].get(
                "booksim_binary", "./booksim2/src/booksim"
            )
            if os.path.exists(booksim_path):
                self.booksim = BookSimInterface(booksim_binary_path=booksim_path)
            else:
                import logging

                logging.warning(
                    f"BookSim binary not found at {booksim_path}. NoC simulation will be bypassed."
                )
                self.booksim = None

        except ImportError:
            import logging

            logging.warning(
                "NoC module (BookSimInterface) not found. Running without NoC modeling."
            )
            self.booksim = None

    def _positive_divisors(self, n):
        n = int(n)
        if n <= 0:
            return {1}
        divs = set()
        r = int(math.sqrt(n))
        for d in range(1, r + 1):
            if n % d == 0:
                divs.add(d)
                divs.add(n // d)
        return divs

    def _prime_factors(self, n):
        n = abs(int(n))
        if n < 2:
            return set()
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    def _normalize_bound(self, bound_pair):
        low, high = bound_pair
        low_i = int(math.floor(min(low, high)))
        high_i = int(math.ceil(max(low, high)))
        return low_i, high_i

    def _build_hw_aligned_candidates(self, low_i, high_i):
        """
        Build transformer BO candidates from hardware prime factors.
        Keep only values whose prime factors are all in the hardware factor set.
        Example: allow 2/3, then 26(2*13) and 34(2*17) are removed.
        """
        low_i = max(1, int(low_i))
        high_i = max(low_i, int(high_i))

        macro_num = int(self.hw.get("macro_num", 1))
        core_num = int(self.hw.get("core_num", 1))
        tile_num = int(self.hw.get("tile_num", 1))

        allowed_primes = set()
        allowed_primes |= self._prime_factors(macro_num)
        allowed_primes |= self._prime_factors(core_num)
        allowed_primes |= self._prime_factors(math.gcd(macro_num, core_num))
        allowed_primes |= self._prime_factors(tile_num)

        if not allowed_primes:
            return list(range(low_i, high_i + 1))

        def is_hw_smooth(v):
            if v == 1:
                return True
            x = int(v)
            for p in sorted(allowed_primes):
                while x % p == 0:
                    x //= p
            return x == 1

        candidates = [v for v in range(low_i, high_i + 1) if is_hw_smooth(v)]

        if not candidates:
            # Defensive fallback to avoid empty BO domain.
            candidates = list(range(low_i, high_i + 1))

        return sorted(set(candidates))

    def _generate_output_paths(self):
        """Dynamically generate output paths based on config to avoid hardcoding."""
        output_dir = self.paths["output_root"]
        pipeline_paths = [
            os.path.join(output_dir, f"pipeline-isaac-{self.dnn}-{self.layers[i]}")
            for i in range(len(self.layers))
        ]
        origin_paths = [
            os.path.join(
                output_dir, f"pipeline_origin-isaac-{self.dnn}-{self.layers[i]}"
            )
            for i in range(len(self.layers))
        ]
        return pipeline_paths, origin_paths

    def run_cnn_evaluation(self):
        """
        Refactored from CNN_pipeline_analyzer.
        Handles Workload parsing, Tile Allocation, BO optimization, and Energy simulation.
        """
        output_path_pipeline, output_path_no_pipeline = self._generate_output_paths()

        workload = [self.analyzer.get_workload(layer) for layer in self.layers]
        compute = []
        weights_access = []

        # 1. Extract workload constraints
        for i, layer in enumerate(self.layers):
            if "R" in workload[i].keys() or "S" in workload[i].keys():
                workload_size = (
                    workload[i]["C"]
                    * workload[i]["M"]
                    * workload[i]["R"]
                    * workload[i]["S"]
                    * self.hw["precision"]
                )
            else:
                workload_size = (
                    workload[i]["C"] * workload[i]["M"] * self.hw["precision"]
                )

            compute.append(
                workload[i].get("C", 1)
                * workload[i].get("M", 1)
                * workload[i].get("R", 1)
                * workload[i].get("S", 1)
                * workload[i].get("P", 1)
                * workload[i].get("Q", 1)
            )
            weights_access.append(workload_size)

        # 2. Minimum Tile Allocation calculation
        min_tile_allocation = [
            function.tile_allocation(
                workload[i],
                macro_num=self.hw["macro_num"],
                core_num=self.hw["core_num"],
                array_col=self.hw["array_col"],
                array_row=self.hw["array_row"],
                cim_depth=self.hw["cim_depth"],
                precision=self.hw["precision"],
            )
            for i in range(len(self.layers))
        ]

        logging.info(f"Min tiles required: {min_tile_allocation}")

        # 3. Greedy Allocation
        allocation = function.greedy_tile_allocation(
            self.layers,
            compute,
            min_tile_allocation,
            self.hw["tile_num"],
            self.model["head_num"],
        )
        allocation_list = list(allocation.values())
        logging.info(f"Greedy allocation: {allocation_list}")

        # 4. Bayesian Optimization Setup
        bound = []
        for i, j in enumerate(allocation_list):
            lb = (
                round(allocation_list[i] * 0.5)
                if round(allocation_list[i] * 0.5) != 0
                else 1
            )
            if min_tile_allocation[i] < j:
                bound.append(
                    (
                        max(min_tile_allocation[i], lb),
                        min(
                            allocation_list[i] + 0.02 * self.hw["tile_num"],
                            allocation_list[i] * 2,
                        ),
                    )
                )
            else:
                bound.append((allocation_list[i], allocation_list[i] + 1))

        evaluator = FitnessEvaluator(self)

        callback = evaluator.cnn_fitness_callback

        logging.info("Initializing Bayesian Optimizer for CNN Evaluation...")
        optimizer = Bayesian_Optimizer(
            bounds=bound,
            evaluate_callback=callback,
            n_calls=100,
            dnn_name=self.dnn,
            alpha=0.2,
            tile_num=self.hw["tile_num"],
            layers=self.layers,
            head_num=self.model.get("head_num", 1),
            block_num=self.model.get("block", 1),
            transformer=False,
            multi_layer=False,
            batch=False,
            max_block=1,
        )
        result, _ = optimizer.run_optimization()
        tile_allocation = result.x_opt.astype(int)

        # 5. Parallel Execution
        logging.info("Starting Parallel Execution...")
        executor = ParallelExecutor(
            layer_num=len(self.layers),
            arch_path=self.paths["arch_root"],
            tile_num=tile_allocation,
            layers=self.layers,
            DNN=self.dnn,
            MACRO_NAME="isaac_isca_2016",
        )
        executor.run_parallel()

        # 6. Pipeline Dependency & Bubble Analysis
        Dataspace = [[] for _ in range(len(self.layers))]
        cal_time_list = []
        bubble_list = []
        time_scale = []
        start_time = 0

        for i in range(len(self.layers)):
            _, cur_time_scale, _, last_time, Dataspace[i] = (
                self.pipeline_analyzer.parse_dataspace(
                    output_path_pipeline[i], workload[i], start_time=start_time
                )
            )
            time_scale.append(cur_time_scale)
            if i == 0:
                cal_time_list.append((start_time, last_time))
                bubble_list.append({})

        for i in range(len(self.layers) - 1):
            logging.info(
                f"Processing Pipeline stage: {self.layers[i]} -> {self.layers[i+1]}"
            )
            actual_time, _, _, _, _, bubble, cal_time, _, Dataspace[i + 1] = (
                self.pipeline_analyzer.pipeline_analysis(
                    time_scale[i + 1],
                    self.pipeline_analyzer.maxpool[i],
                    self.pipeline_analyzer.fc[i + 1],
                    Input_dataspace=Dataspace[i],
                    Output_dataspace=Dataspace[i + 1],
                    shortcut=self.pipeline_analyzer.shortcut[i + 1],
                )
            )
            cal_time_list.append(cal_time)
            bubble_list.append(bubble)

        # Plotting
        plot_dir = os.path.join(self.paths["script_root"], f"ISAAC-{self.dnn}")
        os.makedirs(plot_dir, exist_ok=True)
        plot.plot_bubble(plot_dir, cal_time_list, bubble_list, actual_time=actual_time)

        # 7. Energy & Cycle Calculation
        self._calculate_cnn_energy_metrics(
            output_path_pipeline, output_path_no_pipeline, actual_time, weights_access
        )

    def _calculate_cnn_energy_metrics(
        self, output_path_pipeline, output_path_no_pipeline, actual_time, weights_access
    ):
        """Extracted Energy logic to keep run_cnn_evaluation clean."""
        logging.info(f"Cycle with pipeline = {actual_time}")
        pipeline_access = []
        util_pipeline = 0
        for i in range(len(self.layers)):
            pipeline_access.append(
                self.analyzer.input_output_gen(output_path_pipeline[i])
            )
            util_pipeline += self.analyzer.get_utilization(output_path_pipeline[i])
        inputs = [d[0]["inputs"] for d in pipeline_access]
        outputs = [2 * d[0]["outputs"] for d in pipeline_access]
        weights = [d[0]["weights"] for d in pipeline_access]

        energy_pipeline = []
        for i in range(len(self.layers)):
            energy_pipeline.append(
                self.analyzer.get_total_energy(
                    output_path_pipeline[i] + "/timeloop-mapper.stats.txt"
                )
                - self.analyzer.extract_cim_utilized_instances(output_path_pipeline[i])
                * self.analyzer.extract_cim_write_energy(output_path_pipeline[i])
                * self.analyzer.extract_vector_access_by_module(
                    output_path_pipeline[i], "random_fill", "cim_unit"
                )
                / 1e6
            )
            # print(self.analyzer.get_total_energy(output_path_pipeline[i]+"/timeloop-mapper.stats.txt"),self.analyzer.extract_cim_utilized_instances(output_path_pipeline[i])*self.analyzer.extract_cim_write_energy(output_path_pipeline[i])*self.analyzer.extract_vector_access_by_module(output_path_pipeline[i],'random_fill','cim_unit')/1e6)
        total_energy_pipeline = (
            sum(energy_pipeline) + (inputs[0] + outputs[-1]) * 112.54 / 8 / 1e6
        )

        print("cycle with pipeline = ", actual_time)
        print("energy with pipeline = ", total_energy_pipeline)

        print("Pipeline Compute Energy = ", sum(energy_pipeline))
        print(
            "Pipeline Feature Update Energy = ",
            (inputs[0] + outputs[-1]) * 112.54 / 8 / 1e6,
        )

        print("Pipeline Utilization = ", util_pipeline / len(self.layers))

    def construct_allocation_space(self, batch_size=1):
        """
        Explores the hardware-aware mapping and allocation space using a Boundary-Constrained DFS algorithm.
        Calculates theoretical vs. valid search space and initiates Bayesian Optimization.
        """
        import time
        from src.TileAllocator import TileAllocator
        from src.Bayes_opt import Bayesian_Optimizer

        min_tiles = []
        workload = []
        compute = []
        workload_group = []

        # 1. Extract workload features and compute bounds
        for layer in self.layers:
            wl = self.analyzer.get_workload(layer)
            workload.append(wl)

            if "R" in wl.keys() or "S" in wl.keys():
                workload_size = (
                    wl["C"]
                    * wl["M"]
                    * wl.get("R", 1)
                    * wl.get("S", 1)
                    * self.hw["precision"]
                )
            else:
                workload_size = wl["C"] * wl["M"] * self.hw["precision"]

            compute.append(
                wl.get("C", 1)
                * wl.get("M", 1)
                * wl.get("R", 1)
                * wl.get("S", 1)
                * wl.get("P", 1)
                * wl.get("Q", 1)
            )
            workload_group.append(workload_size)

        # 2. Calculate minimum tiles required per layer
        for i, layer in enumerate(self.layers):
            tile_alloc = function.tile_allocation(
                workload[i],
                macro_num=self.hw["macro_num"],
                core_num=self.hw["core_num"],
                array_col=self.hw["array_col"],
                array_row=self.hw["array_row"],
                cim_depth=self.hw["cim_depth"],
                precision=self.hw["precision"],
            )
            # Clip min_tiles to the maximum available tiles
            min_tiles.append(min(tile_alloc, self.hw["tile_num"]))

        logging.info(f"Minimum tiles per layer: {min_tiles}")

        start_time = time.time()

        # 3. Explore allocations using TileAllocator
        allocator = TileAllocator(
            layers=self.layers,
            compute_workloads=compute,
            min_tiles_per_layer=min_tiles,
            max_tiles_per_group=self.hw["tile_num"],
            min_layers_per_group=1,  # Minimum 1 layer per group
            max_layers_per_group=16,  # Maximum 16 layers per group
            workload_shape=workload,
        )
        all_allocations = allocator.explore_allocations()
        analysis = allocator.analyze_allocations()

        # 4. Search Space Analysis Logging
        N = len(self.layers)
        theoretical_space = 2 ** (N - 1) if N > 0 else 0

        logging.info("==================================================")
        logging.info(f"Network Layers (N): {N}")
        logging.info(
            f"Theoretical exhaustive search space (2^(N-1)): {theoretical_space:,} combinations"
        )
        logging.info("==================================================")
        logging.info(f"Total valid allocations found: {analysis['total_allocations']}")

        # 5. Setup dynamic paths
        savepath = os.path.join(
            self.paths.get("script_root", os.path.dirname(__file__)),
            f"ISAAC-{self.dnn}",
        )
        os.makedirs(savepath, exist_ok=True)

        output_path_pipeline, output_path_origin = self._generate_output_paths()

        # 6. Initialize and run Bayesian Optimizer
        logging.info("Initializing Bayesian Optimizer...")
        # Store all allocations in the engine instance so the callback can access them
        self.all_allocations = all_allocations

        # Instantiate the fitness evaluator and select the appropriate callback
        evaluator = FitnessEvaluator(self)
        callback = evaluator.cnn_multi_layer_fitness_callback

        optimizer = Bayesian_Optimizer(
            bounds=[(0, len(all_allocations) - 1)],
            evaluate_callback=callback,
            n_calls=100,
            dnn_name=self.dnn,
            alpha=0.2,
            # --- Constraint Control Parameters ---
            tile_num=self.hw.get("tile_num", 1344),
            layers=self.layers,
            head_num=self.model.get("head_num", 1),
            block_num=self.model.get("block", 1),
            transformer=False,
            multi_layer=True,
            batch=(batch_size > 1),
            max_block=getattr(self, "max_block", 1),
        )

        result, step = optimizer.run_optimization()

        # 7. Time tracking and return
        end_time = time.time()
        total_seconds = end_time - start_time
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60

        logging.info(
            f"Bayesian Optimization Total Time: {minutes} min {seconds:.2f} sec"
        )

        return total_seconds, step, result.fx_opt

    def run_transformer_evaluation(self):
        """
        Executes pipeline analysis specifically tailored for Transformer architectures.
        Includes operator grouping (Q, K, V and FFNs), nested scheduling for blocks/batches,
        and energy/cycle estimation.
        """
        import time
        import math
        import os
        import logging
        from src.ParallelExecutor import ParallelExecutor
        from src.Bayes_opt import Bayesian_Optimizer
        import src.function as function

        # 1. Load parameters from config
        head_num = self.model.get("head_num", 1)
        block = self.model.get("block", 1)
        batch_size = self.model.get("batch_size", 1)
        batch = 1 if batch_size > 1 else 0
        output_path_pipeline, output_path_no_pipeline = self._generate_output_paths()

        # 2. Extract Workload & Compute limits
        workload = []
        compute = []
        weights = []
        for i, layer in enumerate(self.layers):
            wl = self.analyzer.get_workload(layer)
            workload.append(wl)

            if "R" in wl.keys() or "S" in wl.keys():
                workload_size = (
                    wl["C"]
                    * wl["M"]
                    * wl.get("R", 1)
                    * wl.get("S", 1)
                    * self.hw["precision"]
                )
            else:
                workload_size = wl["C"] * wl["M"] * self.hw["precision"]

            compute.append(
                wl.get("C", 1)
                * wl.get("M", 1)
                * wl.get("R", 1)
                * wl.get("S", 1)
                * wl.get("P", 1)
                * wl.get("Q", 1)
            )
            weights.append(
                wl.get("C", 1)
                * wl.get("M", 1)
                * wl.get("R", 1)
                * wl.get("S", 1)
                * self.hw["precision"]
            )

        # 3. Minimum Tile Allocation & Greedy Allocation
        min_tile_allocation = []
        for i, layer in enumerate(self.layers):
            min_tiles = function.tile_allocation(
                workload[i],
                macro_num=self.hw["macro_num"],
                core_num=self.hw["core_num"],
                array_col=self.hw["array_col"],
                array_row=self.hw["array_row"],
                cim_depth=self.hw["cim_depth"],
                precision=self.hw["precision"],
            )
            min_tile_allocation.append(min_tiles)

        logging.info(f"Min tile allocation: {min_tile_allocation}")

        allocation_dict = function.greedy_tile_allocation(
            self.layers,
            compute,
            min_tile_allocation,
            self.hw["tile_num"] / block,
            head_num,
            transformer=True,
        )
        allocation = [math.ceil(x) for x in allocation_dict.values()]
        logging.info(f"Greedy allocation: {allocation}")

        # 4. Bayesian Optimization Bounds Setup
        Projection = ["Q", "K", "V"]
        Mh_attention = ["A", "Z0"]
        FNN = ["Z1", "FFN1", "FFN2"]

        bound = []
        selected_indices = []
        i = 0
        alpha = 0.2
        while i < len(self.layers):
            lb = round(allocation[i] * 0.5) if round(allocation[i] * 0.5) != 0 else 1
            ub = min(
                allocation[i] + alpha * self.hw["tile_num"] / block, allocation[i] * 2
            )
            bound.append((max(min_tile_allocation[i], lb), ub))
            selected_indices.append(i)
            if self.layers[i] in FNN[1:]:
                i += 2
            elif self.layers[i] in Projection:
                i += 3
            else:
                i += 1

        candidate_domains = []
        for var_i, _layer_idx in enumerate(selected_indices):
            low_i, high_i = self._normalize_bound(bound[var_i])
            candidate_domains.append(self._build_hw_aligned_candidates(low_i, high_i))

        space_dimensions = [len(d) for d in candidate_domains]
        total_search_space = math.prod(space_dimensions)

        logging.info(f"Variables dimension (N): {len(bound)}")
        logging.info(f"Options per dimension: {space_dimensions}")
        logging.info(
            f"🚀 Total Search Space Size: {total_search_space:.2e} ({total_search_space} possible combinations)"
        )

        # 5. Execute Bayesian Optimization (Optional: Enable if needed, currently fast-forwarding to default assignment)
        evaluator = FitnessEvaluator(self)

        callback = evaluator.transformer_fitness_callback

        logging.info("Initializing Bayesian Optimizer for Transformer Evaluation...")
        optimizer = Bayesian_Optimizer(
            bounds=bound,
            evaluate_callback=callback,
            n_calls=100,
            dnn_name=self.dnn,
            alpha=alpha,
            tile_num=self.hw["tile_num"],
            layers=self.layers,
            head_num=head_num,
            block_num=block,
            transformer=True,
            multi_layer=False,
            batch=(batch_size > 1),
            max_block=1,
            candidate_domains=candidate_domains,
        )

        result, _ = optimizer.run_optimization()
        tile_allocation = result.x_opt.astype(int)
        x0, x1, x3, x6, x7 = tile_allocation

        # Structure variable mapping (QKV share mapping, FFN share mapping)
        tile_allocation = [
            int(x0),  # x0 (e.g. initial layer)
            int(x1),  # x1 (Q)
            int(x1),  # x2 = x1 (K)
            int(x3),  # x3 (V)
            int(x3),  # x4 = x3 (A)
            int(x3),  # x5 = x3 (Z0)
            int(x6),  # x6 (Z1)
            int(x7),  # x7 (FFN1/2)
        ]

        # 6. Parallel Hardware Simulation
        logging.info("Starting Parallel Execution...")
        macro_num = [self.hw.get("macro_num", 12)] * len(self.layers)
        executor = ParallelExecutor(
            layer_num=len(self.layers),
            arch_path=self.paths["arch_root"],
            tile_num=tile_allocation,
            layers=self.layers,
            DNN=self.dnn,
            MACRO_NAME="isaac_isca_2016",
        )
        executor.run_parallel()

        # 7. Pipeline Flow & Bubble Analysis Initialization
        start_time = 0
        cal_time_list = []
        bubble_list = []
        layers_cal = []
        layers_bubble = []
        layers_weight_update = []
        A_start_time = 0

        end_time = {key: 0 for key in self.layers}
        Attention_dataspace = {key: [] for key in Mh_attention}
        FFN_dataspace = {key: [[] for _ in range(batch_size)] for key in FNN}
        batch_start_time = [0 for _ in range(batch_size)]

        # 8. Nested Scheduling Loop (Blocks x Batches)
        for idx in range(block):
            logging.info(f"--- Processing Block {idx} | Start Time: {start_time} ---")
            Projection_dataspace = {key: [] for key in Projection}
            current_layer_cal = []
            current_layer_bubble = []
            current_layer_weight = []

            for batch in range(batch_size):
                current_batch_cal = []
                current_batch_bubble = []
                current_batch_weight = []
                time_scale = []

                # --- Projection Phase ---
                if idx == 0:
                    start_time = max(end_time["K"], 0)
                    for i in Projection:
                        _, cur_time_scale, _, last_time, Projection_dataspace[i] = (
                            self.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[self.layers.index(i)],
                                workload[self.layers.index(i)],
                                start_time=start_time,
                            )
                        )
                        cal_time_list.append((start_time, last_time))
                        current_batch_cal.append((start_time, last_time))
                        bubble_list.append({})
                        current_batch_bubble.append({})
                        current_batch_weight.append([])
                        time_scale.append(cur_time_scale)
                        A_start_time = max(A_start_time, last_time)
                        end_time[i] = last_time
                else:
                    start_time = (
                        max(batch_start_time[batch], 0)
                        if batch == 0
                        else max(start_time, end_time["K"])
                    )
                    for i in Projection:
                        _, cur_time_scale, _, last_time, Projection_dataspace[i] = (
                            self.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[self.layers.index(i)],
                                workload[self.layers.index(i)],
                                start_time=start_time,
                            )
                        )
                        (
                            actual_time,
                            _,
                            _,
                            _,
                            _,
                            bubble,
                            cal_time,
                            _,
                            Projection_dataspace[i],
                        ) = self.pipeline_analyzer.pipeline_analysis(
                            cur_time_scale,
                            0,
                            0,
                            Input_dataspace=FFN_dataspace["FFN2"][batch],
                            Output_dataspace=Projection_dataspace[i],
                            transformer=True,
                        )
                        cal_time_list.append(cal_time)
                        bubble_list.append(bubble)
                        time_scale.append(cur_time_scale)
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)
                        current_batch_weight.append([])
                        end_time[i] = actual_time
                        A_start_time = max(A_start_time, actual_time)

                # --- Attention Phase ---
                start_time = (
                    max(start_time, end_time["K"])
                    if batch == 0
                    else max(start_time, end_time["K"], end_time["A"])
                )
                for i in Mh_attention:
                    if i == "A":
                        _, cur_time_scale, _, last_time, Attention_dataspace[i] = (
                            self.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[self.layers.index(i)],
                                workload[self.layers.index(i)],
                                start_time=start_time,
                            )
                        )
                        cal_time_list.append((start_time, last_time))
                        bubble_list.append({})
                        current_batch_cal.append((start_time, last_time))
                        current_batch_bubble.append({})
                        current_batch_weight.append([])
                        end_time[i] = last_time
                    else:
                        if batch != 0:
                            start_time = end_time["Z0"]
                        _, cur_time_scale, _, last_time, Attention_dataspace[i] = (
                            self.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[self.layers.index(i)],
                                workload[self.layers.index(i)],
                                start_time=start_time,
                            )
                        )
                    time_scale.append(cur_time_scale)

                (
                    actual_time,
                    _,
                    _,
                    _,
                    _,
                    bubble,
                    cal_time,
                    _,
                    Attention_dataspace["Z0"],
                ) = self.pipeline_analyzer.pipeline_analysis(
                    time_scale[4],
                    0,
                    0,
                    Input_dataspace=Attention_dataspace["A"],
                    Weight_dataspace=Projection_dataspace["V"],
                    Output_dataspace=Attention_dataspace["Z0"],
                    transformer=True,
                    attention=True,
                    transpose=True,
                )
                cal_time_list.append(cal_time)
                bubble_list.append(bubble)
                current_batch_cal.append(cal_time)
                current_batch_bubble.append(bubble)
                current_batch_weight.append([])
                end_time["Z0"] = actual_time

                # --- FFN Phase ---
                if batch != 0:
                    start_time = max(start_time, end_time["Z1"])
                for i in FNN:
                    _, cur_time_scale, _, last_time, FFN_dataspace[i][batch] = (
                        self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index(i)],
                            workload[self.layers.index(i)],
                            start_time=start_time,
                        )
                    )
                    time_scale.append(cur_time_scale)

                (
                    actual_time,
                    _,
                    _,
                    _,
                    _,
                    bubble,
                    cal_time,
                    _,
                    FFN_dataspace["Z1"][batch],
                ) = self.pipeline_analyzer.pipeline_analysis(
                    time_scale[5],
                    0,
                    0,
                    Input_dataspace=Attention_dataspace["Z0"],
                    Output_dataspace=FFN_dataspace["Z1"][batch],
                    transformer=True,
                    output_projetion=True,
                )
                cal_time_list.append(cal_time)
                bubble_list.append(bubble)
                current_batch_cal.append(cal_time)
                current_batch_bubble.append(bubble)
                current_batch_weight.append([])
                end_time["Z1"] = actual_time

                (
                    actual_time,
                    _,
                    _,
                    _,
                    _,
                    bubble,
                    cal_time,
                    _,
                    FFN_dataspace["FFN1"][batch],
                ) = self.pipeline_analyzer.pipeline_analysis(
                    time_scale[6],
                    0,
                    0,
                    Input_dataspace=FFN_dataspace["Z1"][batch],
                    Output_dataspace=FFN_dataspace["FFN1"][batch],
                    transformer=True,
                )
                cal_time_list.append(cal_time)
                bubble_list.append(bubble)
                current_batch_cal.append(cal_time)
                current_batch_bubble.append(bubble)
                current_batch_weight.append([])
                end_time["FFN1"] = actual_time

                (
                    actual_time,
                    _,
                    _,
                    _,
                    _,
                    bubble,
                    cal_time,
                    _,
                    FFN_dataspace["FFN2"][batch],
                ) = self.pipeline_analyzer.pipeline_analysis(
                    time_scale[7],
                    0,
                    0,
                    Input_dataspace=FFN_dataspace["FFN1"][batch],
                    Output_dataspace=FFN_dataspace["FFN2"][batch],
                    transformer=True,
                )
                cal_time_list.append(cal_time)
                bubble_list.append(bubble)
                current_batch_cal.append(cal_time)
                current_batch_bubble.append(bubble)
                current_batch_weight.append([])
                end_time["FFN2"] = actual_time

                current_layer_cal.append(current_batch_cal)
                current_layer_bubble.append(current_batch_bubble)
                current_layer_weight.append(current_batch_weight)
                batch_start_time[batch] = start_time

            layers_cal.append(current_layer_cal)
            layers_bubble.append(current_layer_bubble)
            layers_weight_update.append(current_layer_weight)

        plot_path = os.path.join(self.paths["script_root"], f"ISAAC-{self.dnn}")
        os.makedirs(plot_path, exist_ok=True)
        plot.plot_combined_timelines_block_batch(
            plot_path,
            layers_cal,
            layers_bubble,
            layers_weight_update,
            actual_time=actual_time,
        )
        # 9. Energy & Cycle Calculation
        pipeline_access = [
            self.analyzer.input_output_gen(path) for path in output_path_pipeline
        ]
        inputs = [d[0]["inputs"] for d in pipeline_access]
        outputs = [d[0]["outputs"] for d in pipeline_access]

        energy_pipeline = []

        for i in range(len(self.layers)):
            cim_utilized = self.analyzer.extract_cim_utilized_instances(
                output_path_pipeline[i]
            )
            cim_write = self.analyzer.extract_cim_write_energy(output_path_pipeline[i])
            vec_access = self.analyzer.extract_vector_access_by_module(
                output_path_pipeline[i], "random_fill", "cim_unit"
            )

            e_pipe = self.analyzer.get_total_energy(
                output_path_pipeline[i] + "/timeloop-mapper.stats.txt"
            ) - (cim_utilized * cim_write * vec_access / 1e6)
            energy_pipeline.append(e_pipe)


        # Scale by head_num for attention layers
        for i in Mh_attention:
            idx = self.layers.index(i)
            weights[idx] *= head_num
            energy_pipeline[idx] *= head_num

        cycle_pipeline = actual_time
        energy_const = 112.54 / 8 / 1e6

        idx_Q = self.layers.index("Q")
        idx_A = self.layers.index("A")
        idx_FFN2 = self.layers.index("FFN2")

        total_energy_pipeline = (
            sum(energy_pipeline)
            + (inputs[idx_Q] + 2 * inputs[idx_A] + 2 * outputs[idx_FFN2]) * energy_const
        )

        # 10. Summary Logging
        logging.info("================ Evaluation Summary ================")
        logging.info(f"Cycles (With Pipeline): {cycle_pipeline:.2f}")
        logging.info(f"Energy (With Pipeline): {total_energy_pipeline * block:.4f} pJ")
        logging.info("---------------- Detail ----------------")
        logging.info(f"Pipeline Compute Energy: {sum(energy_pipeline) * block:.4f}")
        logging.info(f"Pipeline Feature Update Energy: {block * (inputs[idx_Q] + 2 * inputs[idx_A] + 2 * outputs[idx_FFN2]) * energy_const:.4f}")

        return layers_cal, layers_bubble, layers_weight_update

    def run_multi_layer_transformer_batch(self, batch_size=None, batch=True):
        """
        Executes multi-layer Transformer pipeline scheduling with batch processing.
        Handles fine-grained weight update overlapping, block-by-block dependencies,
        and calculates cycles/energy with or without pipeline parallelism.
        """
        import time
        import math
        import os
        import logging
        import src.function as function
        import src.plot as plot
        from src.ParallelExecutor import ParallelExecutor
        from src.Bayes_opt import Bayesian_Optimizer

        batch_size = batch_size or self.model.get("batch_size", 1)
        head_num = self.model.get("head_num", 1)
        block = self.model.get("block", 1)
        output_path_pipeline, output_path_no_pipeline = self._generate_output_paths()

        workload = []
        compute = []
        weight_access = []

        # 1. Extract workload constraints
        for i, layer in enumerate(self.layers):
            wl = self.analyzer.get_workload(layer)
            workload.append(wl)
            workload_size = wl["C"] * wl["M"] * self.hw["precision"]
            compute.append(
                wl.get("C", 1)
                * wl.get("M", 1)
                * wl.get("R", 1)
                * wl.get("S", 1)
                * wl.get("P", 1)
                * wl.get("Q", 1)
            )
            weight_access.append(workload_size)

        min_tile_allocation = []
        for i, layer in enumerate(self.layers):
            min_tile_allocation.append(
                function.tile_allocation(
                    workload[i],
                    macro_num=self.hw["macro_num"],
                    core_num=self.hw["core_num"],
                    array_col=self.hw["array_col"],
                    array_row=self.hw["array_row"],
                    cim_depth=self.hw["cim_depth"],
                    precision=self.hw["precision"],
                )
            )

        logging.info(f"Minimum tile allocation per layer: {min_tile_allocation}")

        # Compute max blocks dynamically based on available tiles
        max_block_tiles = sum(min_tile_allocation) + (head_num - 1) * (
            min_tile_allocation[self.layers.index("A")]
            + min_tile_allocation[self.layers.index("Z0")]
        )
        max_block = (
            math.floor(self.hw["tile_num"] / max_block_tiles)
            if max_block_tiles > 0
            else 1
        )

        allocation_dict = function.greedy_tile_allocation(
            self.layers,
            weight_access,
            min_tile_allocation,
            self.hw["tile_num"],
            head_num,
            transformer=True,
        )
        allocation = list(allocation_dict.values())
        logging.info(
            f"Greedy allocation: {allocation}"
        )

        # 2. Setup Bounds for Bayesian Optimization
        bound = []
        Projection = ["Q", "K", "V"]
        Mh_attention = ["A", "Z0"]
        FNN = ["Z1", "FFN1", "FFN2"]
        i = 0
        alpha = 0.2

        selected_indices = []
        while i < len(self.layers):
            lb = round(allocation[i] * 0.5) if round(allocation[i] * 0.5) != 0 else 1
            ub = min(allocation[i] + self.hw["tile_num"] * alpha, allocation[i] * 3)
            bound.append((max(min_tile_allocation[i], lb), ub))
            selected_indices.append(i)
            if self.layers[i] in FNN[1:]:
                i += 2
            elif self.layers[i] in Projection:
                i += 3
            else:
                i += 1

        candidate_domains = []
        for var_i, _layer_idx in enumerate(selected_indices):
            low_i, high_i = self._normalize_bound(bound[var_i])
            candidate_domains.append(self._build_hw_aligned_candidates(low_i, high_i))

        space_dimensions = [len(d) for d in candidate_domains]
        total_search_space = math.prod(space_dimensions) if space_dimensions else 0

        logging.info(f"--- Search Space Analysis ---")
        logging.info(f"Alpha: {alpha}")
        logging.info(f"Variables dimension (N): {len(bound)}")
        logging.info(f"Options per dimension: {space_dimensions}")
        logging.info(f"Total Search Space Size: {total_search_space:.2e} combinations")

        max_block = 1  # Forced limit as per original logic

        # 3. Optimization (Bayes_opt execution)
        real_start_time = time.time()
        evaluator = FitnessEvaluator(self)
        callback = evaluator.transformer_multi_batch_fitness_callback
        optimizer = Bayesian_Optimizer(
            bounds=bound,
            evaluate_callback=callback,
            n_calls=15,
            dnn_name=self.dnn,
            alpha=alpha,
            tile_num=self.hw.get("tile_num", 1344),
            layers=self.layers,
            head_num=self.model.get("head_num", 1),
            block_num=self.model.get("block", 1),
            transformer=True,
            multi_layer=True,
            batch=batch,
            max_block=getattr(self, "max_block", 1),
        )

        result, step = optimizer.run_optimization()
        x0, x1, x3, x6, x7 = result.x_opt.astype(int)

        # Override with fallback mapping directly using greedy allocation array
        # x0, x1, x3, x6, x7 = allocation[0], allocation[1], allocation[3], allocation[6], allocation[7]
        tile_allocation = [
            int(math.floor(x0)),
            int(x1),
            int(x1),
            int(x3),
            int(x3),
            int(x3),
            int(math.floor(x6)),
            int(x7),
        ]

        # Override to strict mapping logic per user code
        # tile_allocation = allocation

        end_time = time.time()
        logging.info(
            f"BO Optimization Cost: {int((end_time - real_start_time) // 60)} min {(end_time - real_start_time) % 60:.2f} sec"
        )

        # 4. Parallel Simulation Execution
        macro_num = [self.hw.get("macro_num", 12)] * len(self.layers)
        executor = ParallelExecutor(
            layer_num=len(self.layers),
            arch_path=self.paths["arch_root"],
            tile_num=tile_allocation,
            layers=self.layers,
            DNN=self.dnn,
            MACRO_NAME="isaac_isca_2016",
            macro_num=macro_num,
        )
        executor.run_parallel()

        pipeline_access = [
            self.analyzer.input_output_gen(path) for path in output_path_pipeline
        ]
        inputs = [d[0]["inputs"] for d in pipeline_access]
        outputs = [d[0]["outputs"] for d in pipeline_access]
        weights = [d[0]["weights"] for d in pipeline_access]

        # 5. Pipeline State Variables
        start_time = 0
        A_start_time = 0
        width = 256 * 8 * 2

        Projection_dataspace = {key: [] for key in Projection}
        end_time = {key: 0 for key in self.layers}
        start_time_dict = {key: 0 for key in self.layers}
        weight_update_start = 0

        layers_cal = []
        layers_bubble = []
        layers_weight_update = []
        batch_end_time = [0 for _ in range(batch_size)]

        pipeline_weight_update_cost = 0
        pipeline_feature = 0
        total_energy_pipeline = 0

        # 6. Core Scheduling Loop (Blocks x Batches)
        for idx in range(int(block)):
            layer_idx = idx
            current_layer_cal = []
            current_layer_bubble = []
            current_layer_weight = []

            logging.info(f"Processing Block {layer_idx} | Start time: {start_time}")

            if idx == 0:
                for batch in range(int(batch_size)):
                    current_batch_cal, current_batch_bubble, current_batch_weight = (
                        [],
                        [],
                        [],
                    )
                    Projection_dataspace = {key: [] for key in Projection}
                    time_scale = []

                    if batch == 0:
                        # 1.1 Q Weight update
                        weight_update_start = max(
                            weight_update_start,
                            end_time["Q"],
                            end_time["K"],
                            end_time["V"],
                        )
                        weight_update_cost = (
                            weight_access[self.layers.index("Q")] / width
                        )
                        pipeline_weight_update_cost += weight_update_cost
                        QKV_weight_update_end = (
                            weight_update_start + 3 * weight_update_cost
                        )
                        current_batch_weight.append(
                            (weight_update_start, weight_update_cost)
                        )
                        weight_update_start += weight_update_cost

                        Q_start_time = (
                            3 * weight_update_cost
                            if layer_idx == 0
                            else max(
                                weight_update_start,
                                end_time["A"],
                                batch_end_time[batch],
                                QKV_weight_update_end,
                            )
                        )
                        _, cur_ts, _, last_t, Projection_dataspace["Q"] = (
                            self.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[self.layers.index("Q")],
                                workload[self.layers.index("Q")],
                                start_time=Q_start_time,
                            )
                        )
                        current_batch_cal.append((Q_start_time, last_t))
                        current_batch_bubble.append({})
                        time_scale.append(cur_ts)
                        end_time["Q"] = last_t

                        # 1.2 K Weight update
                        weight_update_cost = (
                            weight_access[self.layers.index("K")] / width
                        )
                        current_batch_weight.append(
                            (weight_update_start, weight_update_cost)
                        )
                        weight_update_start += weight_update_cost
                        pipeline_weight_update_cost += weight_update_cost

                        K_start_time = (
                            3 * weight_update_cost
                            if layer_idx == 0
                            else max(
                                weight_update_start,
                                end_time["A"],
                                batch_end_time[batch],
                                QKV_weight_update_end,
                            )
                        )
                        _, cur_ts, _, last_t, Projection_dataspace["K"] = (
                            self.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[self.layers.index("K")],
                                workload[self.layers.index("K")],
                                start_time=K_start_time,
                            )
                        )
                        current_batch_cal.append((K_start_time, last_t))
                        current_batch_bubble.append({})
                        time_scale.append(cur_ts)
                        end_time["K"] = last_t

                        # 1.3 V Weight update
                        weight_update_cost = (
                            weight_access[self.layers.index("V")] / width
                        )
                        current_batch_weight.append(
                            (weight_update_start, weight_update_cost)
                        )
                        weight_update_start += weight_update_cost
                        pipeline_weight_update_cost += weight_update_cost

                        V_start_time = (
                            3 * weight_update_cost
                            if layer_idx == 0
                            else max(
                                weight_update_start,
                                end_time["Z0"],
                                batch_end_time[batch],
                                QKV_weight_update_end,
                            )
                        )
                        _, cur_ts, _, last_t, Projection_dataspace["V"] = (
                            self.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[self.layers.index("V")],
                                workload[self.layers.index("V")],
                                start_time=V_start_time,
                            )
                        )
                        current_batch_cal.append((V_start_time, last_t))
                        current_batch_bubble.append({})
                        time_scale.append(cur_ts)
                        end_time["V"] = last_t

                        start_time = max(batch_end_time[batch], last_t)
                        batch_end_time[batch] = last_t
                    else:
                        for i in Projection:
                            _, cur_ts, _, last_t, Projection_dataspace[i] = (
                                self.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[self.layers.index(i)],
                                    workload[self.layers.index(i)],
                                    start_time=end_time[i],
                                )
                            )
                            current_batch_cal.append((end_time[i], last_t))
                            current_batch_bubble.append({})
                            time_scale.append(cur_ts)
                            end_time[i] = last_t
                            batch_end_time[batch] = last_t

                    current_layer_cal.append(current_batch_cal)
                    current_layer_bubble.append(current_batch_bubble)
                    current_layer_weight.append(current_batch_weight)

            else:
                for batch in range(int(batch_size)):
                    current_batch_cal, current_batch_bubble, current_batch_weight = (
                        [],
                        [],
                        [],
                    )
                    time_scale = []

                    if batch == 0:
                        Attention_dataspace = {key: [] for key in Mh_attention}
                        FFN_dataspace = {key: [] for key in FNN}
                        for i in Mh_attention:
                            current_batch_weight.append([])

                        if layer_idx != 0:
                            weight_update_start = max(
                                weight_update_start, end_time["Z1"]
                            )
                        start_time = max(
                            batch_end_time[batch], end_time["A"], end_time["V"]
                        )

                        for i in Mh_attention:
                            _, cur_ts, _, last_t, Attention_dataspace[i] = (
                                self.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[self.layers.index(i)],
                                    workload[self.layers.index(i)],
                                    start_time=start_time,
                                )
                            )
                            if i == "A":
                                current_batch_cal.append((start_time, last_t))
                                end_time["A"] = last_t
                                start_time_dict["A"] = start_time
                                current_batch_bubble.append({})
                            time_scale.append(cur_ts)

                        (
                            actual_time,
                            _,
                            _,
                            _,
                            _,
                            bubble,
                            cal_time,
                            next_st,
                            Attention_dataspace["Z0"],
                        ) = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[1],
                            0,
                            0,
                            Input_dataspace=Attention_dataspace["A"],
                            Output_dataspace=Attention_dataspace["Z0"],
                            transformer=True,
                        )
                        end_time["Z0"] = actual_time
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)
                        start_time = next_st

                        # Z1 FFN setup
                        weight_update_cost = (
                            weight_access[self.layers.index("Z1")] / width
                        )
                        current_batch_weight.append(
                            (weight_update_start, weight_update_cost)
                        )
                        weight_update_start += weight_update_cost
                        pipeline_weight_update_cost += weight_update_cost
                        start_time = max(weight_update_start, start_time)

                        _, cur_ts, _, _, FFN_dataspace["Z1"] = (
                            self.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[self.layers.index("Z1")],
                                workload[self.layers.index("Z1")],
                                start_time=start_time,
                            )
                        )
                        time_scale.append(cur_ts)
                        (
                            actual_time,
                            _,
                            _,
                            _,
                            _,
                            bubble,
                            cal_time,
                            _,
                            FFN_dataspace["Z1"],
                        ) = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[2],
                            0,
                            0,
                            Input_dataspace=Attention_dataspace["Z0"],
                            Output_dataspace=FFN_dataspace["Z1"],
                            transformer=True,
                            output_projetion=True,
                        )
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)
                        end_time["Z1"] = actual_time

                        # FFN1
                        weight_update_cost = (
                            weight_access[self.layers.index("FFN1")] / width
                        )
                        weight_update_start = max(weight_update_start, end_time["FFN1"])
                        current_batch_weight.append(
                            (weight_update_start, weight_update_cost)
                        )
                        weight_update_start += weight_update_cost
                        pipeline_weight_update_cost += weight_update_cost
                        start_time = max(weight_update_start, start_time)

                        if start_time > end_time["Z1"]:
                            total_energy_pipeline += (
                                (2 * inputs[self.layers.index("FFN1")])
                                * 112.54
                                / 8
                                / 1e6
                            )
                            pipeline_feature += (
                                (2 * inputs[self.layers.index("FFN1")])
                                * 112.54
                                / 8
                                / 1e6
                            )

                        _, cur_ts, _, _, FFN_dataspace["FFN1"] = (
                            self.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[self.layers.index("FFN1")],
                                workload[self.layers.index("FFN1")],
                                start_time=start_time,
                            )
                        )
                        time_scale.append(cur_ts)
                        (
                            actual_time,
                            _,
                            _,
                            _,
                            _,
                            bubble,
                            cal_time,
                            _,
                            FFN_dataspace["FFN1"],
                        ) = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[3],
                            0,
                            0,
                            Input_dataspace=FFN_dataspace["Z1"],
                            Output_dataspace=FFN_dataspace["FFN1"],
                            transformer=True,
                        )
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)
                        end_time["FFN1"] = actual_time

                        # FFN2
                        weight_update_cost = (
                            weight_access[self.layers.index("FFN2")] / width
                        )
                        weight_update_start = max(weight_update_start, end_time["FFN2"])
                        current_batch_weight.append(
                            (weight_update_start, weight_update_cost)
                        )
                        weight_update_start += weight_update_cost
                        pipeline_weight_update_cost += weight_update_cost
                        start_time = max(weight_update_start, start_time)

                        if start_time > end_time["FFN1"]:
                            total_energy_pipeline += (
                                (2 * inputs[self.layers.index("FFN2")])
                                * 112.54
                                / 8
                                / 1e6
                            )
                            pipeline_feature += (
                                (2 * inputs[self.layers.index("FFN2")])
                                * 112.54
                                / 8
                                / 1e6
                            )

                        _, cur_ts, _, _, FFN_dataspace["FFN2"] = (
                            self.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[self.layers.index("FFN2")],
                                workload[self.layers.index("FFN2")],
                                start_time=start_time,
                            )
                        )
                        time_scale.append(cur_ts)
                        (
                            actual_time,
                            _,
                            _,
                            _,
                            _,
                            bubble,
                            cal_time,
                            next_st,
                            FFN_dataspace["FFN2"],
                        ) = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[4],
                            0,
                            0,
                            Input_dataspace=FFN_dataspace["FFN1"],
                            Output_dataspace=FFN_dataspace["FFN2"],
                            transformer=True,
                        )
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)
                        end_time["FFN2"] = actual_time
                        batch_end_time[batch] = actual_time

                        # Follow-up Q, K, V Projections for batch 0
                        for proj_key in ["Q", "K", "V"]:
                            weight_update_start = max(
                                weight_update_start,
                                end_time["Q"] if proj_key == "Q" else start_time,
                                end_time["K"] if proj_key == "Q" else 0,
                                end_time["V"] if proj_key == "Q" else 0,
                            )
                            weight_update_cost = (
                                weight_access[self.layers.index(proj_key)] / width
                            )
                            pipeline_weight_update_cost += weight_update_cost
                            current_batch_weight.append(
                                (weight_update_start, weight_update_cost)
                            )
                            weight_update_start += weight_update_cost

                            start_time = max(
                                weight_update_start, start_time, end_time[proj_key]
                            )
                            _, cur_ts, _, _, Projection_dataspace[proj_key] = (
                                self.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[self.layers.index(proj_key)],
                                    workload[self.layers.index(proj_key)],
                                    start_time=start_time,
                                )
                            )
                            time_scale.append(cur_ts)
                            (
                                actual_time,
                                _,
                                _,
                                _,
                                _,
                                bubble,
                                cal_time,
                                _,
                                Projection_dataspace[proj_key],
                            ) = self.pipeline_analyzer.pipeline_analysis(
                                time_scale[-1],
                                0,
                                0,
                                Input_dataspace=FFN_dataspace["FFN2"],
                                Output_dataspace=Projection_dataspace[proj_key],
                                transformer=True,
                            )

                            current_batch_cal.append(cal_time)
                            current_batch_bubble.append(bubble)
                            end_time[proj_key] = actual_time
                            if proj_key == "V":
                                batch_end_time[batch] = actual_time

                    else:
                        # Batch != 0 logic block
                        Attention_dataspace = {key: [] for key in Mh_attention}
                        FFN_dataspace = {key: [] for key in FNN}
                        start_time = end_time["A"]

                        for i in Mh_attention:
                            if i == "A":
                                _, cur_ts, _, last_t, Attention_dataspace[i] = (
                                    self.pipeline_analyzer.parse_dataspace(
                                        output_path_pipeline[self.layers.index(i)],
                                        workload[self.layers.index(i)],
                                        start_time=start_time,
                                    )
                                )
                                current_batch_cal.append((start_time, last_t))
                                end_time["A"] = last_t
                                start_time_dict["A"] = start_time
                                current_batch_bubble.append({})
                            else:
                                start_time = end_time["Z0"]
                                _, cur_ts, _, last_t, Attention_dataspace[i] = (
                                    self.pipeline_analyzer.parse_dataspace(
                                        output_path_pipeline[self.layers.index(i)],
                                        workload[self.layers.index(i)],
                                        start_time=start_time,
                                    )
                                )
                            time_scale.append(cur_ts)

                        (
                            actual_time,
                            _,
                            _,
                            _,
                            _,
                            bubble,
                            cal_time,
                            next_st,
                            Attention_dataspace["Z0"],
                        ) = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[1],
                            0,
                            0,
                            Input_dataspace=Attention_dataspace["A"],
                            Output_dataspace=Attention_dataspace["Z0"],
                            transformer=True,
                        )
                        end_time["Z0"] = actual_time
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)
                        start_time = next_st

                        # Process FFN pipeline sequence
                        for f_key, input_key in [
                            ("Z1", "Z0"),
                            ("FFN1", "Z1"),
                            ("FFN2", "FFN1"),
                        ]:
                            start_time = end_time[f_key]
                            _, cur_ts, _, _, FFN_dataspace[f_key] = (
                                self.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[self.layers.index(f_key)],
                                    workload[self.layers.index(f_key)],
                                    start_time=start_time,
                                )
                            )
                            time_scale.append(cur_ts)

                            (
                                actual_time,
                                _,
                                _,
                                _,
                                _,
                                bubble,
                                cal_time,
                                next_st,
                                FFN_dataspace[f_key],
                            ) = self.pipeline_analyzer.pipeline_analysis(
                                time_scale[-1],
                                0,
                                0,
                                Input_dataspace=(
                                    Attention_dataspace[input_key]
                                    if input_key == "Z0"
                                    else FFN_dataspace[input_key]
                                ),
                                Output_dataspace=FFN_dataspace[f_key],
                                transformer=True,
                                output_projetion=(f_key == "Z1"),
                            )

                            if (
                                f_key in ["FFN1", "FFN2"]
                                and start_time > end_time[input_key]
                            ):
                                total_energy_pipeline += (
                                    (2 * inputs[self.layers.index(f_key)])
                                    * 112.54
                                    / 8
                                    / 1e6
                                )
                                pipeline_feature += (
                                    (2 * inputs[self.layers.index(f_key)])
                                    * 112.54
                                    / 8
                                    / 1e6
                                )

                            end_time[f_key] = actual_time
                            current_batch_cal.append(cal_time)
                            current_batch_bubble.append(bubble)

                        batch_end_time[batch] = actual_time

                        # Process QKV mapping for subsequent batches
                        for proj_key in ["Q", "K", "V"]:
                            current_batch_weight.append([])
                            start_time = max(start_time, end_time[proj_key])
                            _, cur_ts, _, _, Projection_dataspace[proj_key] = (
                                self.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[self.layers.index(proj_key)],
                                    workload[self.layers.index(proj_key)],
                                    start_time=start_time,
                                )
                            )
                            time_scale.append(cur_ts)
                            (
                                actual_time,
                                _,
                                _,
                                _,
                                _,
                                bubble,
                                cal_time,
                                _,
                                Projection_dataspace[proj_key],
                            ) = self.pipeline_analyzer.pipeline_analysis(
                                time_scale[-1],
                                0,
                                0,
                                Input_dataspace=FFN_dataspace["FFN2"],
                                Output_dataspace=Projection_dataspace[proj_key],
                                transformer=True,
                            )

                            current_batch_cal.append(cal_time)
                            current_batch_bubble.append(bubble)
                            end_time[proj_key] = actual_time
                            if proj_key == "V":
                                batch_end_time[batch] = actual_time

                    current_layer_cal.append(current_batch_cal)
                    current_layer_bubble.append(current_batch_bubble)
                    current_layer_weight.append(current_batch_weight)

            # Append current block timeline data for plotting/statistics.
            layers_cal.append(current_layer_cal)
            layers_bubble.append(current_layer_bubble)
            layers_weight_update.append(current_layer_weight)
            if idx == block - 1:
                current_layer_cal = []
                current_layer_bubble = []
                current_layer_weight = []
                for batch in range(int(batch_size)):
                    current_batch_cal = []
                    current_batch_bubble = []
                    current_batch_weight = []
                    time_scale = []

                    if batch == 0:
                        Attention_dataspace = {key: [] for key in Mh_attention}
                        FFN_dataspace = {key: [] for key in FNN}

                        for i in Mh_attention:
                            current_batch_weight.append([])

                        if layer_idx != 0:
                            weight_update_start = max(
                                weight_update_start, end_time["Z1"]
                            )

                        start_time = max(
                            batch_end_time[batch], end_time["A"], end_time["V"]
                        )
                        for i in Mh_attention:
                            if i == "A":
                                (
                                    _,
                                    cur_time_scale,
                                    _,
                                    last_time,
                                    Attention_dataspace[i],
                                ) = self.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[self.layers.index(i)],
                                    workload[self.layers.index(i)],
                                    start_time=start_time,
                                )
                                current_batch_cal.append((start_time, last_time))
                                end_time["A"] = last_time
                                start_time_dict["A"] = start_time
                                current_batch_bubble.append({})
                            else:
                                (
                                    _,
                                    cur_time_scale,
                                    _,
                                    last_time,
                                    Attention_dataspace[i],
                                ) = self.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[self.layers.index(i)],
                                    workload[self.layers.index(i)],
                                    start_time=start_time,
                                )
                            time_scale.append(cur_time_scale)

                        (
                            actual_time,
                            _,
                            _,
                            _,
                            _,
                            bubble,
                            cal_time,
                            next_start_time,
                            Attention_dataspace["Z0"],
                        ) = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[1],
                            0,
                            0,
                            Input_dataspace=Attention_dataspace["A"],
                            Output_dataspace=Attention_dataspace["Z0"],
                            transformer=True,
                        )
                        end_time["Z0"] = actual_time
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)
                        start_time = next_start_time

                        weight_update_cost = (
                            weight_access[self.layers.index("Z1")] / width
                        )
                        weight_update_start = max(weight_update_start, end_time["Z1"])
                        current_batch_weight.append(
                            (weight_update_start, weight_update_cost)
                        )
                        weight_update_start += weight_update_cost
                        pipeline_weight_update_cost += weight_update_cost
                        start_time = max(
                            weight_update_start, start_time, end_time["Z1"]
                        )

                        _, cur_time_scale, _, last_time, FFN_dataspace["Z1"] = (
                            self.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[self.layers.index("Z1")],
                                workload[self.layers.index("Z1")],
                                start_time=start_time,
                            )
                        )
                        time_scale.append(cur_time_scale)
                        (
                            actual_time,
                            _,
                            _,
                            _,
                            _,
                            bubble,
                            cal_time,
                            next_start_time,
                            FFN_dataspace["Z1"],
                        ) = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[2],
                            0,
                            0,
                            Input_dataspace=Attention_dataspace["Z0"],
                            Output_dataspace=FFN_dataspace["Z1"],
                            transformer=True,
                            output_projetion=True,
                        )
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)
                        end_time["Z1"] = actual_time

                        weight_update_cost = (
                            weight_access[self.layers.index("FFN1")] / width
                        )
                        weight_update_start = max(weight_update_start, end_time["FFN1"])
                        current_batch_weight.append(
                            (weight_update_start, weight_update_cost)
                        )
                        weight_update_start += weight_update_cost
                        pipeline_weight_update_cost += weight_update_cost
                        start_time = max(
                            weight_update_start, start_time, end_time["FFN1"]
                        )

                        _, cur_time_scale, _, last_time, FFN_dataspace["FFN1"] = (
                            self.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[self.layers.index("FFN1")],
                                workload[self.layers.index("FFN1")],
                                start_time=start_time,
                            )
                        )
                        time_scale.append(cur_time_scale)
                        (
                            actual_time,
                            _,
                            _,
                            _,
                            _,
                            bubble,
                            cal_time,
                            next_start_time,
                            FFN_dataspace["FFN1"],
                        ) = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[3],
                            0,
                            0,
                            Input_dataspace=FFN_dataspace["Z1"],
                            Output_dataspace=FFN_dataspace["FFN1"],
                            transformer=True,
                        )
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)
                        end_time["FFN1"] = actual_time

                        weight_update_cost = (
                            weight_access[self.layers.index("FFN2")] / width
                        )
                        current_batch_weight.append(
                            (weight_update_start, weight_update_cost)
                        )
                        weight_update_start = max(weight_update_start, end_time["FFN2"])
                        weight_update_start += weight_update_cost
                        pipeline_weight_update_cost += weight_update_cost
                        start_time = max(
                            weight_update_start, start_time, end_time["FFN2"]
                        )

                        _, cur_time_scale, _, last_time, FFN_dataspace["FFN2"] = (
                            self.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[self.layers.index("FFN2")],
                                workload[self.layers.index("FFN2")],
                                start_time=start_time,
                            )
                        )
                        time_scale.append(cur_time_scale)
                        (
                            actual_time,
                            _,
                            _,
                            _,
                            _,
                            bubble,
                            cal_time,
                            next_start_time,
                            FFN_dataspace["FFN2"],
                        ) = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[4],
                            0,
                            0,
                            Input_dataspace=FFN_dataspace["FFN1"],
                            Output_dataspace=FFN_dataspace["FFN2"],
                            transformer=True,
                        )
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)
                        end_time["FFN2"] = actual_time

                        start_time = next_start_time
                        batch_end_time[batch] = actual_time
                    else:
                        Attention_dataspace = {key: [] for key in Mh_attention}
                        FFN_dataspace = {key: [] for key in FNN}
                        start_time = end_time["A"]

                        for i in Mh_attention:
                            if i == "A":
                                (
                                    _,
                                    cur_time_scale,
                                    _,
                                    last_time,
                                    Attention_dataspace[i],
                                ) = self.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[self.layers.index(i)],
                                    workload[self.layers.index(i)],
                                    start_time=start_time,
                                )
                                current_batch_cal.append((start_time, last_time))
                                end_time["A"] = last_time
                                start_time_dict["A"] = start_time
                                current_batch_bubble.append({})
                            else:
                                start_time = end_time["Z0"]
                                (
                                    _,
                                    cur_time_scale,
                                    _,
                                    last_time,
                                    Attention_dataspace[i],
                                ) = self.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[self.layers.index(i)],
                                    workload[self.layers.index(i)],
                                    start_time=start_time,
                                )
                            time_scale.append(cur_time_scale)

                        (
                            actual_time,
                            _,
                            _,
                            _,
                            _,
                            bubble,
                            cal_time,
                            next_start_time,
                            Attention_dataspace["Z0"],
                        ) = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[1],
                            0,
                            0,
                            Input_dataspace=Attention_dataspace["A"],
                            Output_dataspace=Attention_dataspace["Z0"],
                            transformer=True,
                        )
                        end_time["Z0"] = actual_time
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)
                        start_time = next_start_time

                        start_time = end_time["Z1"]
                        _, cur_time_scale, _, last_time, FFN_dataspace["Z1"] = (
                            self.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[self.layers.index("Z1")],
                                workload[self.layers.index("Z1")],
                                start_time=start_time,
                            )
                        )
                        time_scale.append(cur_time_scale)
                        (
                            actual_time,
                            _,
                            _,
                            _,
                            _,
                            bubble,
                            cal_time,
                            next_start_time,
                            FFN_dataspace["Z1"],
                        ) = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[2],
                            0,
                            0,
                            Input_dataspace=Attention_dataspace["Z0"],
                            Output_dataspace=FFN_dataspace["Z1"],
                            transformer=True,
                            output_projetion=True,
                        )
                        end_time["Z1"] = actual_time
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)

                        start_time = end_time["FFN1"]
                        _, cur_time_scale, _, last_time, FFN_dataspace["FFN1"] = (
                            self.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[self.layers.index("FFN1")],
                                workload[self.layers.index("FFN1")],
                                start_time=start_time,
                            )
                        )
                        time_scale.append(cur_time_scale)
                        (
                            actual_time,
                            _,
                            _,
                            _,
                            _,
                            bubble,
                            cal_time,
                            next_start_time,
                            FFN_dataspace["FFN1"],
                        ) = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[3],
                            0,
                            0,
                            Input_dataspace=FFN_dataspace["Z1"],
                            Output_dataspace=FFN_dataspace["FFN1"],
                            transformer=True,
                        )
                        end_time["FFN1"] = actual_time
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)

                        start_time = end_time["FFN2"]
                        _, cur_time_scale, _, last_time, FFN_dataspace["FFN2"] = (
                            self.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[self.layers.index("FFN2")],
                                workload[self.layers.index("FFN2")],
                                start_time=start_time,
                            )
                        )
                        time_scale.append(cur_time_scale)
                        (
                            actual_time,
                            _,
                            _,
                            _,
                            _,
                            bubble,
                            cal_time,
                            next_start_time,
                            FFN_dataspace["FFN2"],
                        ) = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[4],
                            0,
                            0,
                            Input_dataspace=FFN_dataspace["FFN1"],
                            Output_dataspace=FFN_dataspace["FFN2"],
                            transformer=True,
                        )
                        end_time["FFN2"] = actual_time
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)

                        start_time = next_start_time
                        batch_end_time[batch] = actual_time

                    current_layer_cal.append(current_batch_cal)
                    current_layer_bubble.append(current_batch_bubble)
                    current_layer_weight.append(current_batch_weight)

                layers_cal.append(current_layer_cal)
                layers_bubble.append(current_layer_bubble)
                layers_weight_update.append(current_layer_weight)
            # Aggregate total feature energy
            idx_Q = self.layers.index("Q")
            total_energy_pipeline += (
                (inputs[idx_Q] * batch_size + outputs[idx_Q] * batch_size)
                * 112.54
                / 8
                / 1e6
            )
            pipeline_feature += (
                (inputs[idx_Q] * batch_size + outputs[idx_Q] * batch_size)
                * 112.54
                / 8
                / 1e6
            )

        # 7. Visualization & Stats Plotting
        plot_path = os.path.join(self.paths["script_root"], f"ISAAC-{self.dnn}")
        os.makedirs(plot_path, exist_ok=True)
        plot.plot_combined_timelines_block_batch(
            plot_path,
            layers_cal,
            layers_bubble,
            layers_weight_update,
            actual_time=actual_time,
        )

        cal_time, weight_time, overlap_time = function.compute_time_statistics(
            layers_cal, layers_weight_update
        )
        logging.info(
            f"Cal-only Time: {cal_time} | Weight-only Time: {weight_time} | Overlap Time: {overlap_time}"
        )

        # 8. Un-pipelined baseline calculation
        no_pipeline_access = [
            self.analyzer.input_output_gen(path) for path in output_path_no_pipeline
        ]
        inputs_no_pipeline = [d[0]["inputs"] for d in no_pipeline_access]
        outputs_no_pipeline = [d[0]["outputs"] for d in no_pipeline_access]
        weights_no_pipeline = [d[0]["weights"] for d in no_pipeline_access]

        write_weight_energy_pipeline = []
        energy_pipeline = []

        for i in range(len(self.layers)):
            cim_util_pipe = self.analyzer.extract_cim_utilized_instances(
                output_path_pipeline[i]
            )
            cim_write_pipe = (
                self.analyzer.extract_cim_write_energy(output_path_pipeline[i]) / 1e6
            )
            energy_pipeline.append(
                self.analyzer.get_total_energy(
                    output_path_pipeline[i] + "/timeloop-mapper.stats.txt"
                )
                - (cim_util_pipe * cim_write_pipe)
            )
            write_weight_energy_pipeline.append(cim_util_pipe * cim_write_pipe)


        for i in Mh_attention:
            idx_i = self.layers.index(i)
            energy_pipeline[idx_i] *= head_num
            write_weight_energy_pipeline[idx_i] *= head_num

        cycle_pipeline = actual_time
        weight_access_energy = pipeline_weight_update_cost * width * 112.54 / 8 / 1e6

        total_energy_pipeline += (
            sum(energy_pipeline) * batch_size + sum(write_weight_energy_pipeline)
        ) * block + weight_access_energy

        logging.info("================ Final Performance Summary ================")
        logging.info(f"Tile allocation = {tile_allocation}")
        logging.info(f"Cycles (Pipeline) = {cycle_pipeline:.2f}")
        logging.info(
            f"Weight update cycles (Pipeline) = {pipeline_weight_update_cost:.2f}"
        )
        logging.info(f"Energy (Pipeline) = {total_energy_pipeline:.4f} pJ")
        logging.info("---------------- Detail ----------------")
        logging.info(
            f"Pipeline Compute Energy = {sum(energy_pipeline) * batch_size * block:.4f}"
        )
        logging.info(f"Pipeline Feature Update Energy = {pipeline_feature:.4f}")
        logging.info(
            f"Pipeline Weight Update Energy = {weight_access_energy + sum(write_weight_energy_pipeline) * block:.4f}"
        )

        return layers_cal, layers_bubble, layers_weight_update

    def run_multi_layer_transformer_batch_group(
        self, batch_size=None, batch=True, grouped_indices=[]
    ):
        """
        [Ultimate Edition] Multi-Group Transformer Pipeline Evaluation
        Handles dynamic cross-block grouping, topological pipeline tracking,
        and detailed energy/power assessments!
        """
        batch_size = batch_size or self.model.get("batch_size", 1)
        head_num = self.model.get("head_num", 1)
        block = self.model.get("block", 1)
        output_path_pipeline, output_path_no_pipeline = self._generate_output_paths()

        workload = []
        weight_access = []

        # 1. Extract workload constraints
        for i, layer in enumerate(self.layers):
            wl = self.analyzer.get_workload(layer)
            workload.append(wl)
            workload_size = (
                wl.get("C", 1)
                * wl.get("M", 1)
                * wl.get("R", 1)
                * wl.get("S", 1)
                * self.hw.get("precision", 16)
            )
            weight_access.append(workload_size)

        try:
            weight_access[self.layers.index("A")] *= head_num
            weight_access[self.layers.index("Z0")] *= head_num
        except ValueError:
            pass

        # 2. Dynamic 1D Grouping
        hw_total_tiles = self.hw.get("tile_num", 1344)
        arch_size_bits = (
            hw_total_tiles
            * self.hw.get("macro_num", 1)
            * self.hw.get("core_num", 1)
            * self.hw.get("array_col", 1)
            * self.hw.get("array_row", 1)
            * self.hw.get("cim_depth", 1)
        )
        """
        grouped_indices = function.capacity_aware_transformer_grouping(
            weights=weight_access, layer_names=self.layers, arch_size=arch_size_bits, ops_per_block=8
        )
        """
        # 3. Setup Bounds & Alpha Dimension Reduction
        tile_capacity_bits = (
            self.hw.get("macro_num", 16)
            * self.hw.get("core_num", 4)
            * self.hw.get("array_col", 128)
            * self.hw.get("array_row", 128)
            * self.hw.get("cim_depth", 1)
        )

        true_min_t = []
        for i, layer in enumerate(self.layers):
            layer_name = layer.split(".")[0]
            if layer_name in ["A", "Z0"]:
                min_t = max(1, int(math.ceil(weight_access[i] / tile_capacity_bits)))
            else:
                min_t = function.tile_allocation(
                    workload[i],
                    macro_num=self.hw["macro_num"],
                    core_num=self.hw["core_num"],
                    array_col=self.hw["array_col"],
                    array_row=self.hw["array_row"],
                    cim_depth=self.hw["cim_depth"],
                    precision=self.hw["precision"],
                )
            true_min_t.append(min_t)

        allocation_dict = {}

        compute_list = []
        for i, layer in enumerate(self.layers):
            if hasattr(self, "compute") and self.compute:
                compute_list.append(self.compute[i])
            else:
                wl = workload[i]
                # M: Output channel, C: Input channel, N/P/Q: Sequence/Batch, R/S: Filter
                macs = (
                    wl.get("M", 1)
                    * wl.get("C", 1)
                    * wl.get("N", 1)
                    * wl.get("P", 1)
                    * wl.get("Q", 1)
                    * wl.get("R", 1)
                    * wl.get("S", 1)
                )

                if layer.split(".")[0] in ["A", "Z0"]:
                    macs *= head_num

                compute_list.append(macs)

        for group in grouped_indices:
            group_min_sum = sum(true_min_t[i] for i in group)
            group_compute = [compute_list[i] for i in group]
            total_compute = sum(group_compute)

            remaining_tiles = hw_total_tiles - group_min_sum
            extra_tiles = [0] * len(group)

            if remaining_tiles > 0 and total_compute > 0:
                for idx, layer_i in enumerate(group):
                    ratio = group_compute[idx] / total_compute
                    extra_tiles[idx] = int(ratio * remaining_tiles)

                leftover = remaining_tiles - sum(extra_tiles)
                if leftover > 0:
                    sorted_idx = sorted(
                        range(len(group)), key=lambda x: group_compute[x], reverse=True
                    )
                    for idx in sorted_idx:
                        extra_tiles[idx] += 1
                        leftover -= 1
                        if leftover == 0:
                            break

            for idx, layer_i in enumerate(group):
                layer_name = self.layers[layer_i].split(".")[0]
                allocation_dict[layer_name] = true_min_t[layer_i] + extra_tiles[idx]

        bounds = []
        var_map = []
        alpha = 0.2
        Projection = ["Q", "K", "V"]

        i = 0
        var_idx = 0
        while i < len(self.layers):
            layer_name = self.layers[i].split(".")[0]
            alloc = allocation_dict[layer_name]
            min_t = true_min_t[i]
            static_allocs = {}
            current_group = None
            for g in grouped_indices:
                if i in g:
                    current_group = g
                    break
            if len(current_group) == 1:
                actual_lb = hw_total_tiles
                actual_ub = hw_total_tiles + 0.01

                bounds.append((int(actual_lb), int(actual_ub)))

                if layer_name in Projection:
                    while (
                        i < len(self.layers)
                        and self.layers[i].split(".")[0] in Projection
                    ):
                        var_map.append(var_idx)
                        i += 1
                elif layer_name in ["FFN1", "FFN2"]:
                    while i < len(self.layers) and self.layers[i].split(".")[0] in [
                        "FFN1",
                        "FFN2",
                    ]:
                        var_map.append(var_idx)
                        i += 1
                else:
                    var_map.append(var_idx)
                    i += 1
                var_idx += 1
                continue
            shared_count = 0
            other_min_sum = 0

            for j in current_group:
                j_name = self.layers[j].split(".")[0]
                if (
                    (layer_name in Projection and j_name in Projection)
                    or (layer_name in ["FFN1", "FFN2"] and j_name in ["FFN1", "FFN2"])
                    or (layer_name == j_name)
                ):
                    shared_count += 1
                else:
                    other_min_sum += true_min_t[j]

            absolute_max = int((hw_total_tiles - other_min_sum) / shared_count)

            lb = round(alloc * 0.5) if round(alloc * 0.5) != 0 else 1
            ub = min(alloc + hw_total_tiles * alpha, alloc * 2)

            actual_lb = max(min_t, lb)
            actual_ub = min(ub, hw_total_tiles, absolute_max)
            actual_lb = min(actual_lb, actual_ub)

            if int(actual_lb) == int(actual_ub):
                if int(actual_ub) < absolute_max:
                    actual_ub += 1
                elif int(actual_lb) > min_t:
                    actual_lb -= 1

            bounds.append((int(actual_lb), int(actual_ub)))

            if layer_name in Projection:
                while (
                    i < len(self.layers) and self.layers[i].split(".")[0] in Projection
                ):
                    var_map.append(var_idx)
                    i += 1
            elif layer_name in ["FFN1", "FFN2"]:
                while i < len(self.layers) and self.layers[i].split(".")[0] in [
                    "FFN1",
                    "FFN2",
                ]:
                    var_map.append(var_idx)
                    i += 1
            else:
                var_map.append(var_idx)
                i += 1
            var_idx += 1

        logging.info(f"--- Search Space Analysis ---")
        logging.info(f"Reduced BO Dimension (N): {len(bounds)}")
        logging.info(f"Dynamic Bounds: {bounds}")

        # 4. Bayesian Optimization
        real_start_time = time.time()
        evaluator = FitnessEvaluator(self)

        optimizer = Bayesian_Optimizer(
            bounds=bounds,
            evaluate_callback=lambda x: evaluator.multi_group_transformer_fitness_callback(
                x, grouped_indices, self.layers, var_map
            ),
            n_calls=60,
            dnn_name=self.dnn,
            alpha=alpha,
            tile_num=hw_total_tiles,
            layers=self.layers,
            head_num=head_num,
            block_num=block,
            transformer=True,
            multi_layer=True,
            batch=batch,
            grouped_indices=grouped_indices,
            var_map=var_map,
        )

        result, step = optimizer.run_optimization()

        best_alloc_raw = result.x_opt.tolist()
        full_alloc_raw = [best_alloc_raw[v] for v in var_map]

        default_macros = self.hw.get(
            "macro_num", 1
        )
        tile_allocation = [1] * len(self.layers)
        macro_allocation = [default_macros] * len(self.layers)

        for i, layer in enumerate(self.layers):
            layer_name = layer.split(".")[0]
            val = max(1, int(math.floor(full_alloc_raw[i])))

            if layer_name in ["A", "Z0"]:
                total_macros = val * default_macros
                macros_per_head = total_macros // head_num

                if macros_per_head >= default_macros:
                    tile_allocation[i] = macros_per_head // default_macros
                    macro_allocation[i] = default_macros
                else:
                    tile_allocation[i] = 1
                    macro_allocation[i] = max(1, macros_per_head)
            else:
                tile_allocation[i] = val
                macro_allocation[i] = default_macros

        end_time = time.time()
        logging.info(f"Final Optimal Tile Allocation: {tile_allocation}")
        logging.info(f"Final Optimal Macro Allocation: {macro_allocation}")

        # 5. Parallel Simulation Execution
        executor = ParallelExecutor(
            layer_num=len(self.layers),
            arch_path=self.paths["arch_root"],
            tile_num=tile_allocation,
            macro_num=macro_allocation,
            layers=self.layers,
            DNN=self.dnn,
            MACRO_NAME="isaac_isca_2016",
        )
        executor.run_parallel()

        # Parse inputs/outputs for Energy tracking
        pipeline_access = [
            self.analyzer.input_output_gen(path) for path in output_path_pipeline
        ]
        inputs = [d[0]["inputs"] for d in pipeline_access]
        outputs = [d[0]["outputs"] for d in pipeline_access]
        weights = [d[0]["weights"] for d in pipeline_access]

        # ==============================================================
        # 6. Topological Pipeline Tracer & Energy Tracker
        # ==============================================================
        layers_cal = []
        layers_bubble = []
        layers_weight_update = []
        canonical_layers = ["Q", "K", "V", "A", "Z0", "Z1", "FFN1", "FFN2"]

        actual_time = 0
        weight_update_start = 0
        pipeline_weight_update_cost = 0
        pipeline_feature = 0
        total_energy_pipeline = 0
        width = 256 * 8 * 2

        end_time_dict = [{layer: 0 for layer in self.layers} for _ in range(batch_size)]
        Dataspace = [{key: [] for key in self.layers} for _ in range(batch_size)]
        batch_end_time = [0 for _ in range(batch_size)]

        # ==============================================================
        # ==============================================================
        layers_cal = []
        layers_bubble = []
        layers_weight_update = []

        canonical_layers = ["Q", "K", "V", "A", "Z0", "Z1", "FFN1", "FFN2"]

        actual_time = 0
        weight_update_start = 0
        pipeline_weight_update_cost = 0
        pipeline_feature = 0
        width = 256 * 8 * 2

        end_time_dict = [{layer: 0 for layer in self.layers} for _ in range(batch_size)]
        Dataspace = [{layer: [] for layer in self.layers} for _ in range(batch_size)]
        batch_end_time = [0 for _ in range(batch_size)]

        hw_available_tiles = self.hw.get("tile_num", 1344)
        weight_noc_ready_time = 0
        active_ops = (
            []
        )

        def get_req_tiles(l_idx):
            t = tile_allocation[l_idx]
            name = self.layers[l_idx].split(".")[0]
            return t * head_num if name in ["A", "Z0"] else t

        for b_idx in range(int(block)):
            temp_cal = {
                b: {l: [] for l in canonical_layers} for b in range(int(batch_size))
            }
            temp_bubble = {
                b: {l: {} for l in canonical_layers} for b in range(int(batch_size))
            }
            temp_weight = {
                b: {l: [] for l in canonical_layers} for b in range(int(batch_size))
            }

            for stage_idx, group in enumerate(grouped_indices):

                for layer_idx in group:
                    curr_layer = self.layers[layer_idx]
                    curr_name = curr_layer.split(".")[0]
                    req_tiles = get_req_tiles(layer_idx)

                    while hw_available_tiles < req_tiles and len(active_ops) > 0:
                        active_ops.sort(key=lambda x: x[0])
                        fin_time, freed_tiles, op_name = active_ops.pop(0)
                        hw_available_tiles += freed_tiles
                        weight_noc_ready_time = max(weight_noc_ready_time, fin_time)

                    hw_available_tiles -= req_tiles

                    w_size = weight_access[layer_idx]
                    real_delay = evaluator.get_noc_weight_delay(
                        w_size, tile_allocation[layer_idx]
                    )

                    w_start = weight_noc_ready_time
                    w_end = w_start + real_delay
                    weight_noc_ready_time = w_end
                    pipeline_weight_update_cost += real_delay

                    for batch in range(int(batch_size)):

                        if batch == 0:
                            start_time = w_end
                        else:
                            start_time = end_time_dict[batch - 1][curr_layer]
                        if batch == 0:
                            temp_weight[batch][curr_name] = [(w_start, real_delay)]
                        else:
                            temp_weight[batch][curr_name] = []
                        if curr_name in ["FFN1", "FFN2"] and b_idx > 0:
                            total_energy_pipeline += (
                                (2 * inputs[layer_idx]) * 112.54 / 8 / 1e6
                            )
                            pipeline_feature += (
                                (2 * inputs[layer_idx]) * 112.54 / 8 / 1e6
                            )

                        _, cur_ts, _, last_t, Dataspace[batch][curr_layer] = (
                            self.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[layer_idx],
                                workload[layer_idx],
                                start_time=start_time,
                            )
                        )
                        current_stage_ops = [
                            self.layers[i].split(".")[0] for i in group
                        ]
                        is_attn = curr_name in [
                            "A"
                        ]
                        is_proj = curr_name == "Z1"
                        is_transpose = curr_name == "Z0"
                        should_pipeline = True
                        in_ds = None
                        weight_ds = None

                        if curr_name in ["Q", "K", "V"]:
                            if b_idx > 0:
                                in_ds = copy.deepcopy(Dataspace[batch]["FFN2"])
                            else:
                                should_pipeline = False
                        elif curr_name == "A":
                            in_ds = copy.deepcopy(Dataspace[batch]["Q"])
                            weight_ds = copy.deepcopy(Dataspace[batch]["K"])

                        elif curr_name == "Z0":
                            in_ds = copy.deepcopy(Dataspace[batch]["A"])
                            weight_ds = copy.deepcopy(Dataspace[batch]["V"])
                        elif curr_name == "Z1":
                            in_ds = copy.deepcopy(Dataspace[batch]["Z0"])
                        elif curr_name == "FFN1":
                            in_ds = copy.deepcopy(Dataspace[batch]["Z1"])
                        elif curr_name == "FFN2":
                            in_ds = copy.deepcopy(Dataspace[batch]["FFN1"])
                        upstream_map = {
                            "A": ["Q", "K"],
                            "Z0": ["A", "V"],
                            "Z1": ["Z0"],
                            "FFN1": ["Z1"],
                            "FFN2": ["FFN1"],
                        }

                        if curr_name in upstream_map:
                            for up_op in upstream_map[curr_name]:
                                if up_op not in current_stage_ops:
                                    should_pipeline = False
                                    in_ds = None
                                    weight_ds = None
                                    break
                        if should_pipeline:
                            (
                                pipe_time,
                                _,
                                _,
                                _,
                                _,
                                bubble,
                                cal_t,
                                _,
                                Dataspace[batch][curr_layer],
                            ) = self.pipeline_analyzer.pipeline_analysis(
                                cur_ts,
                                0,
                                0,
                                Input_dataspace=in_ds,
                                Output_dataspace=copy.deepcopy(
                                    Dataspace[batch][curr_layer]
                                ),
                                Weight_dataspace=weight_ds,
                                data_space_path=output_path_pipeline[layer_idx],
                                transformer=True,
                                attention=is_attn,
                                transpose=is_transpose,
                                output_projetion=is_proj,
                                head=head_num,
                            )
                        else:
                            pipe_time = last_t
                            cal_t = [(start_time, last_t)]
                            bubble = {}

                        end_time_dict[batch][curr_layer] = pipe_time
                        actual_time = max(actual_time, pipe_time)

                        temp_cal[batch][curr_name] = cal_t
                        temp_bubble[batch][curr_name] = bubble

                        if curr_name == "FFN2":
                            batch_end_time[batch] = pipe_time

                    fin_time = end_time_dict[batch_size - 1][curr_layer]
                    active_ops.append((fin_time, req_tiles, curr_layer))

            block_cal = []
            block_bubble = []
            block_weight = []

            for batch in range(int(batch_size)):
                b_cal = [temp_cal[batch][l] for l in canonical_layers]
                b_bub = [temp_bubble[batch][l] for l in canonical_layers]
                b_wei = [temp_weight[batch][l] for l in canonical_layers]

                block_cal.append(b_cal)
                block_bubble.append(b_bub)
                block_weight.append(b_wei)

            layers_cal.append(block_cal)
            layers_bubble.append(block_bubble)
            layers_weight_update.append(block_weight)

            # Block level Feature Energy
            idx_Q = self.layers.index("Q")
            total_energy_pipeline += (
                (inputs[idx_Q] * batch_size + outputs[idx_Q] * batch_size)
                * 112.54
                / 8
                / 1e6
            )
            pipeline_feature += (
                (inputs[idx_Q] * batch_size + outputs[idx_Q] * batch_size)
                * 112.54
                / 8
                / 1e6
            )
        # ==============================================================
        # 7. Plots and Baseline Energy Evaluation
        # ==============================================================
        plot_path = os.path.join(self.paths["script_root"], f"ISAAC-{self.dnn}")
        os.makedirs(plot_path, exist_ok=True)
        plot.plot_combined_timelines_block_batch(
            plot_path,
            layers_cal,
            layers_bubble,
            layers_weight_update,
            layer_names=self.layers,
            actual_time=actual_time,
        )

        cal_time, weight_time, overlap_time = function.compute_time_statistics(
            layers_cal, layers_weight_update
        )
        logging.info(
            f"Cal-only Time: {cal_time} | Weight-only Time: {weight_time} | Overlap Time: {overlap_time}"
        )
        logging.info(f"🎉 Pipeline Final Latency: {actual_time:.2f} cycles")

        # 8. Un-pipelined baseline calculation
        no_pipeline_access = [
            self.analyzer.input_output_gen(path) for path in output_path_no_pipeline
        ]
        inputs_no_pipeline = [d[0]["inputs"] for d in no_pipeline_access]
        outputs_no_pipeline = [d[0]["outputs"] for d in no_pipeline_access]
        weights_no_pipeline = [d[0]["weights"] for d in no_pipeline_access]

        write_weight_energy_pipeline = []
        energy_pipeline = []

        for i in range(len(self.layers)):
            cim_util_pipe = self.analyzer.extract_cim_utilized_instances(
                output_path_pipeline[i]
            )
            cim_write_pipe = (
                self.analyzer.extract_cim_write_energy(output_path_pipeline[i]) / 1e6
            )
            energy_pipeline.append(
                self.analyzer.get_total_energy(
                    output_path_pipeline[i] + "/timeloop-mapper.stats.txt"
                )
                - (cim_util_pipe * cim_write_pipe)
            )
            write_weight_energy_pipeline.append(cim_util_pipe * cim_write_pipe)


        Mh_attention = ["A", "Z0"]
        for i in Mh_attention:
            idx_i = self.layers.index(i)
            energy_pipeline[idx_i] *= head_num
            write_weight_energy_pipeline[idx_i] *= head_num

        cycle_pipeline = actual_time
        weight_access_energy = pipeline_weight_update_cost * width * 112.54 / 8 / 1e6

        total_energy_pipeline += (
            sum(energy_pipeline) * batch_size + sum(write_weight_energy_pipeline)
        ) * block + weight_access_energy

        logging.info("================ Final Performance Summary ================")
        logging.info(f"Tile allocation = {tile_allocation}")
        logging.info(f"Cycles (Pipeline) = {cycle_pipeline:.2f}")
        logging.info(
            f"Weight update cycles (Pipeline) = {pipeline_weight_update_cost:.2f}"
        )
        logging.info(f"Energy (Pipeline) = {total_energy_pipeline:.4f} pJ")
        logging.info("---------------- Detail ----------------")
        logging.info(
            f"Pipeline Compute Energy = {sum(energy_pipeline) * batch_size * block:.4f}"
        )
        logging.info(f"Pipeline Feature Update Energy = {pipeline_feature:.4f}")
        logging.info(
            f"Pipeline Weight Update Energy = {weight_access_energy + sum(write_weight_energy_pipeline) * block:.4f}"
        )

        return layers_cal, layers_bubble, layers_weight_update
