import os
import math
import logging
from src.engine.config import resolve_config_paths
from src.analysis import PipelineAnalyzer
from src.allocation import ParallelExecutor
from src.allocation import allocation_utils
from src.noc import BookSimInterface
from src.optimization import BayesianOptimizer
from src.engine.types import LayerMetricsBundle
from src.engine_runners import cnn_runner
from src.engine_runners import transformer_runner


class TriCIMEngine:
    def __init__(self, config):
        self.config = resolve_config_paths(config)
        self.hw = self.config["hardware"]
        self.model = self.config["model"]
        self.paths = self.config["paths"]

        self.dnn = self.model["dnn"]
        self.layers = []  # Will be populated by pipeline_analyzer
        self.compute = []
        self.legal_tiles_layer_map = {}
        self.legal_tiles_summary = {}
        self.legal_tiles_avg_compression = 1.0

        # Initialize the underlying parser and analyzer
        self.pipeline_analyzer = PipelineAnalyzer(self.config)
        self.layers = self.pipeline_analyzer.layers
        self.analyzer = (
            self.pipeline_analyzer.analyzer
        )  # Access to underlying Timeloop analysis
        try:
            booksim_path = self.paths.get("booksim_binary", "./booksim2/src/booksim")
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
        arch_name = self.paths.get("arch_name", "isaac")
        pipeline_paths = [
            os.path.join(output_dir, f"pipeline-{arch_name}-{self.dnn}-{self.layers[i]}")
            for i in range(len(self.layers))
        ]
        origin_paths = [
            os.path.join(
                output_dir, f"pipeline_origin-{arch_name}-{self.dnn}-{self.layers[i]}"
            )
            for i in range(len(self.layers))
        ]
        return pipeline_paths, origin_paths

    def _tile_allocation_kwargs(self):
        return {
            "macro_num": self.hw["macro_num"],
            "core_num": self.hw["core_num"],
            "array_col": self.hw["array_col"],
            "array_row": self.hw["array_row"],
            "cim_depth": self.hw["cim_depth"],
            "precision": self.hw["precision"],
        }

    def _compute_ops(self, workload):
        return (
            workload.get("C", 1)
            * workload.get("M", 1)
            * workload.get("R", 1)
            * workload.get("S", 1)
            * workload.get("P", 1)
            * workload.get("Q", 1)
        )

    def _weight_access(self, workload, include_kernel=True):
        value = workload.get("C", 1) * workload.get("M", 1)
        if include_kernel:
            value *= workload.get("R", 1) * workload.get("S", 1)
        value *= self.hw.get("precision", 16)
        return value

    def _collect_layer_metrics(
        self,
        include_kernel_in_weights,
        clip_min_tiles=False,
        attention_weight_scale=1,
    ):
        workloads = [self.analyzer.get_workload(layer) for layer in self.layers]
        compute = [self._compute_ops(workload) for workload in workloads]
        weight_access = [
            self._weight_access(workload, include_kernel=include_kernel_in_weights)
            for workload in workloads
        ]
        min_tiles = [
            allocation_utils.tile_allocation(
                workload, **self._tile_allocation_kwargs()
            )
            for workload in workloads
        ]

        if clip_min_tiles:
            min_tiles = [min(tile, self.hw["tile_num"]) for tile in min_tiles]

        if attention_weight_scale != 1:
            for name in ("A", "Z0"):
                if name in self.layers:
                    weight_access[self.layers.index(name)] *= attention_weight_scale

        return LayerMetricsBundle(
            workloads=workloads,
            compute=compute,
            weight_access=weight_access,
            min_tiles=min_tiles,
        )

    def _build_candidate_domains(self, bounds, selected_indices=None):
        candidate_domains = []
        if selected_indices is None:
            selected_indices = list(range(len(bounds)))

        for var_i, _layer_idx in enumerate(selected_indices):
            low_i, high_i = self._normalize_bound(bounds[var_i])
            candidate_domains.append(self._build_hw_aligned_candidates(low_i, high_i))
        return candidate_domains

    def _run_parallel_mapping(self, tile_num, macro_num=None):
        macro_num = (
            macro_num
            if macro_num is not None
            else [self.hw.get("macro_num", 12)] * len(self.layers)
        )
        executor = ParallelExecutor(
            layer_num=len(self.layers),
            arch_path=self.paths["arch_root"],
            tile_num=tile_num,
            layers=self.layers,
            DNN=self.dnn,
            macro_name=self.paths.get("macro_name", "isaac_isca_2016"),
            tile_name=self.paths.get("arch_name", "isaac"),
            macro_num=macro_num,
        )
        executor.run_parallel()

    def _bo_config(self):
        return self.config.get("optimization", {}).get("bayes", {})

    def _bo_value(self, key, default=None):
        return self._bo_config().get(key, default)

    def _build_optimizer(self, **kwargs):
        defaults = {
            "dnn_name": self.dnn,
            "tile_num": self.hw.get("tile_num", 1344),
            "layers": self.layers,
            "head_num": self.model.get("head_num", 1),
            "block_num": self.model.get("block", 1),
            "n_calls": self._bo_value("max_calls", 100),
            "alpha": self._bo_value("alpha", 0.2),
            "initial_points": self._bo_value("initial_points", 10),
            "early_stop_patience": self._bo_value("early_stop_patience", 20),
            "acquisition_weight": self._bo_value("acquisition_weight", 2),
            "random_state": self._bo_value("random_state", 42),
        }
        defaults.update(kwargs)
        return BayesianOptimizer(**defaults)

    def _get_plot_dir(self):
        arch_name = self.paths.get("arch_name", "isaac").upper()
        plot_dir = os.path.join(self.paths["plot_root"], f"{arch_name}-{self.dnn}")
        os.makedirs(plot_dir, exist_ok=True)
        return plot_dir

    def run_cnn_evaluation(self):
        return cnn_runner.run_cnn_evaluation(self)

    def run_one_tile_evaluation(self):
        if self.model.get("transformer", False):
            return transformer_runner.run_one_tile_evaluation(self)
        return cnn_runner.run_one_tile_evaluation(self)

    def run_multi_batch_cnn_pipeline(self, batch_size=1):
        return cnn_runner.run_multi_batch_cnn_pipeline(self, batch_size=batch_size)

    def _calculate_cnn_energy_metrics(
        self, output_path_pipeline, output_path_no_pipeline, actual_time, weights_access
    ):
        return cnn_runner.calculate_cnn_energy_metrics(
            self,
            output_path_pipeline,
            output_path_no_pipeline,
            actual_time,
            weights_access,
        )

    def run_multi_layer_cnn(self, batch_size=1):
        return cnn_runner.run_multi_layer_cnn(self, batch_size=batch_size)

    def construct_allocation_space(self, batch_size=1):
        return self.run_multi_layer_cnn(batch_size=batch_size)

    def run_transformer_evaluation(self):
        return transformer_runner.run_transformer_evaluation(self)

    def run_multi_layer_transformer_batch(self, batch_size=None, batch=True):
        return transformer_runner.run_multi_layer_transformer_batch(
            self, batch_size=batch_size, batch=batch
        )

    def run_multi_layer_transformer_batch_group(
        self, batch_size=None, batch=True, grouped_indices=[]
    ):
        return transformer_runner.run_multi_layer_transformer_batch_group(
            self,
            batch_size=batch_size,
            batch=batch,
            grouped_indices=grouped_indices,
        )
