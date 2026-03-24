import os
import concurrent.futures
import logging


class ParallelExecutor:
    def __init__(
        self, layer_num, arch_path, tile_num, layers, DNN, MACRO_NAME, macro_num=None
    ):
        self.layer_num = layer_num
        self.arch_path = arch_path
        self.tile_num = tile_num
        self.macro_num = macro_num if macro_num is not None else [12] * layer_num
        self.layers = layers
        self.DNN = DNN
        self.MACRO_NAME = MACRO_NAME

    def _worker(self, i):
        """Run one mapping task for a single layer index."""
        try:
            logging.getLogger().setLevel(logging.WARNING)
            import utils as utl

            utl.quick_run(
                chip="pipeline",
                macro=self.MACRO_NAME,
                tile="isaac",
                dnn=self.DNN,
                layer=self.layers[i],
                variables={
                    "N_TILE": self.tile_num[i],
                    "N_MACRO": self.macro_num[i],
                },
            )
            return True

        except Exception as e:
            print(f"Worker {i} ({self.layers[i]}) failed: {str(e)}")
            return None

    def run_parallel(self, max_workers=None):
        """Run mapping tasks in parallel and return ordered results."""
        if max_workers is None:
            max_workers = min(os.cpu_count(), self.layer_num)

        results = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            future_to_index = {
                executor.submit(self._worker, i): i for i in range(self.layer_num)
            }

            for future in concurrent.futures.as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    result = future.result()
                    results.append((i, result))
                except Exception as e:
                    print(f"Worker {i} ({self.layers[i]}) failed: {str(e)}")
                    results.append((i, None))

        results.sort(key=lambda x: x[0])
        return [r for i, r in results]
