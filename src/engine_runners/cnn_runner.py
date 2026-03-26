import logging
import time
import copy

from src.optimization import FitnessEvaluator
from src.allocation import TileAllocator
from src.allocation import allocation_utils
from src.visualization import timeline_plot


def _build_cnn_bo_bounds(engine, allocation_list, min_tile_allocation):
    alpha = engine._bo_value("alpha", 0.2)
    bound = []
    for i, j in enumerate(allocation_list):
        lb = round(allocation_list[i] * 0.5) or 1
        if min_tile_allocation[i] < j:
            bound.append(
                (
                    max(min_tile_allocation[i], lb),
                    min(
                        allocation_list[i] + alpha * engine.hw["tile_num"],
                        allocation_list[i] * 2,
                    ),
                )
            )
        else:
            bound.append((allocation_list[i], allocation_list[i] + 1))
    return bound


def run_cnn_evaluation(engine):
    """
    Refactored from CNN_pipeline_analyzer.
    Handles workload parsing, tile allocation, BO optimization, and energy simulation.
    """
    output_path_pipeline, output_path_no_pipeline = engine._generate_output_paths()
    metrics = engine._collect_layer_metrics(
        include_kernel_in_weights=True,
        clip_min_tiles=False,
    )
    workload = metrics.workloads
    compute = metrics.compute
    weights_access = metrics.weight_access
    min_tile_allocation = metrics.min_tiles

    logging.info(f"Min tiles required: {min_tile_allocation}")

    allocation = allocation_utils.greedy_tile_allocation(
        engine.layers,
        compute,
        min_tile_allocation,
        engine.hw["tile_num"],
        engine.model["head_num"],
    )
    allocation_list = list(allocation.values())
    logging.info(f"Greedy allocation: {allocation_list}")

    bound = _build_cnn_bo_bounds(engine, allocation_list, min_tile_allocation)

    evaluator = FitnessEvaluator(engine)
    callback = evaluator.cnn_fitness_callback

    logging.info("Initializing Bayesian Optimizer for CNN Evaluation...")
    optimizer = engine._build_optimizer(
        bounds=bound,
        evaluate_callback=callback,
        transformer=False,
        multi_layer=False,
        batch=False,
        max_block=1,
    )
    result, _ = optimizer.run_optimization()
    tile_allocation = [int(x) for x in result.x_opt]

    logging.info("Starting Parallel Execution...")
    engine._run_parallel_mapping(tile_allocation)

    dataspace = [[] for _ in range(len(engine.layers))]
    cal_time_list = []
    bubble_list = []
    time_scale = []
    start_time = 0
    actual_time = 0

    for i in range(len(engine.layers)):
        _, cur_time_scale, _, last_time, dataspace[i] = (
            engine.pipeline_analyzer.parse_dataspace(
                output_path_pipeline[i], workload[i], start_time=start_time
            )
        )
        time_scale.append(cur_time_scale)
        if i == 0:
            cal_time_list.append((start_time, last_time))
            bubble_list.append({})
            actual_time = last_time

    for i in range(len(engine.layers) - 1):
        logging.info(
            f"Processing Pipeline stage: {engine.layers[i]} -> {engine.layers[i+1]}"
        )
        actual_time, _, _, _, _, bubble, cal_time, _, dataspace[i + 1] = (
            engine.pipeline_analyzer.pipeline_analysis(
                time_scale[i + 1],
                engine.pipeline_analyzer.maxpool[i],
                engine.pipeline_analyzer.fc[i + 1],
                Input_dataspace=dataspace[i],
                Output_dataspace=dataspace[i + 1],
                shortcut=engine.pipeline_analyzer.shortcut[i + 1],
            )
        )
        cal_time_list.append(cal_time)
        bubble_list.append(bubble)

    plot_dir = engine._get_plot_dir()
    timeline_plot.plot_bubble(
        plot_dir, cal_time_list, bubble_list, actual_time=actual_time
    )

    calculate_cnn_energy_metrics(
        engine,
        output_path_pipeline,
        output_path_no_pipeline,
        actual_time,
        weights_access,
    )


def run_multi_batch_cnn_pipeline(engine, batch_size=1):
    """
    CNN basic pipeline with explicit inter-batch scheduling.
    Keeps the legacy single-batch path unchanged and models overlap across batches.
    """
    if batch_size <= 1:
        return run_cnn_evaluation(engine)

    output_path_pipeline, output_path_no_pipeline = engine._generate_output_paths()
    metrics = engine._collect_layer_metrics(
        include_kernel_in_weights=True,
        clip_min_tiles=False,
    )
    workload = metrics.workloads
    compute = metrics.compute
    weights_access = metrics.weight_access
    min_tile_allocation = metrics.min_tiles

    logging.info(f"Min tiles required: {min_tile_allocation}")

    allocation = allocation_utils.greedy_tile_allocation(
        engine.layers,
        compute,
        min_tile_allocation,
        engine.hw["tile_num"],
        engine.model["head_num"],
    )
    allocation_list = list(allocation.values())
    logging.info(f"Greedy allocation: {allocation_list}")

    bound = _build_cnn_bo_bounds(engine, allocation_list, min_tile_allocation)

    evaluator = FitnessEvaluator(engine)
    callback = evaluator.cnn_fitness_callback

    logging.info("Initializing Bayesian Optimizer for CNN Multi-Batch Pipeline...")
    optimizer = engine._build_optimizer(
        bounds=bound,
        evaluate_callback=callback,
        transformer=False,
        multi_layer=False,
        batch=False,
        max_block=1,
    )
    result, _ = optimizer.run_optimization()
    tile_allocation = [int(x) for x in result.x_opt]

    logging.info("Starting Parallel Execution...")
    engine._run_parallel_mapping(tile_allocation)

    time_scale = []
    for i in range(len(engine.layers)):
        _, cur_time_scale, _, _, _ = engine.pipeline_analyzer.parse_dataspace(
            output_path_pipeline[i],
            workload[i],
            start_time=0,
        )
        time_scale.append(cur_time_scale)

    end_time_dict = [{layer: 0 for layer in engine.layers} for _ in range(batch_size)]
    dataspace_dict = [{layer: [] for layer in engine.layers} for _ in range(batch_size)]
    layers_cal = [[None for _ in range(batch_size)] for _ in range(len(engine.layers))]
    layers_bubble = [[{} for _ in range(batch_size)] for _ in range(len(engine.layers))]
    layers_weight = [[[] for _ in range(batch_size)] for _ in range(len(engine.layers))]

    actual_time = 0
    for batch in range(int(batch_size)):
        logging.info(f"Scheduling CNN pipeline for batch {batch + 1}/{batch_size}")
        for layer_idx, layer_name in enumerate(engine.layers):
            start_time = 0
            if batch > 0:
                start_time = end_time_dict[batch - 1][layer_name]

            _, _, _, last_time, dataspace_dict[batch][layer_name] = (
                engine.pipeline_analyzer.parse_dataspace(
                    output_path_pipeline[layer_idx],
                    workload[layer_idx],
                    start_time=start_time,
                )
            )

            if layer_idx == 0:
                cal_time = (start_time, last_time)
                bubble = {}
                pipe_time = last_time
            else:
                (
                    pipe_time,
                    _,
                    _,
                    _,
                    _,
                    bubble,
                    cal_time,
                    _,
                    dataspace_dict[batch][layer_name],
                ) = engine.pipeline_analyzer.pipeline_analysis(
                    time_scale[layer_idx],
                    engine.pipeline_analyzer.maxpool[layer_idx - 1],
                    engine.pipeline_analyzer.fc[layer_idx],
                    Input_dataspace=copy.deepcopy(
                        dataspace_dict[batch][engine.layers[layer_idx - 1]]
                    ),
                    Output_dataspace=copy.deepcopy(dataspace_dict[batch][layer_name]),
                    shortcut=engine.pipeline_analyzer.shortcut[layer_idx],
                )

            end_time_dict[batch][layer_name] = pipe_time
            actual_time = max(actual_time, pipe_time)
            layers_cal[layer_idx][batch] = cal_time
            layers_bubble[layer_idx][batch] = bubble

    plot_dir = engine._get_plot_dir()
    timeline_plot.plot_combined_timelines_batch_layers(
        plot_dir,
        layers_cal,
        layers_bubble,
        layers_weight,
        labels=engine.layers,
        batch_labels=[f"Batch {i + 1}" for i in range(int(batch_size))],
        actual_time=actual_time,
    )

    calculate_cnn_energy_metrics(
        engine,
        output_path_pipeline,
        output_path_no_pipeline,
        actual_time,
        weights_access,
        batch_size=batch_size,
    )


def calculate_cnn_energy_metrics(
    engine,
    output_path_pipeline,
    output_path_no_pipeline,
    actual_time,
    weights_access,
    batch_size=1,
):
    """Extracted energy logic to keep run_cnn_evaluation clean."""
    logging.info(f"Cycle with pipeline = {actual_time}")
    pipeline_access = []
    util_pipeline = 0
    for i in range(len(engine.layers)):
        pipeline_access.append(engine.analyzer.input_output_gen(output_path_pipeline[i]))
        util_pipeline += engine.analyzer.get_utilization(output_path_pipeline[i])
    inputs = [d[0]["inputs"] for d in pipeline_access]
    outputs = [2 * d[0]["outputs"] for d in pipeline_access]
    weights = [d[0]["weights"] for d in pipeline_access]

    energy_pipeline = []
    for i in range(len(engine.layers)):
        energy_pipeline.append(
            engine.analyzer.get_total_energy(
                output_path_pipeline[i] + "/timeloop-mapper.stats.txt"
            )
            - engine.analyzer.extract_cim_utilized_instances(output_path_pipeline[i])
            * engine.analyzer.extract_cim_write_energy(output_path_pipeline[i])
            * engine.analyzer.extract_vector_access_by_module(
                output_path_pipeline[i], "random_fill", "cim_unit"
            )
            / 1e6
        )
    total_energy_pipeline = (
        sum(energy_pipeline) * batch_size
        + (inputs[0] + outputs[-1]) * batch_size * 112.54 / 8 / 1e6
    )

    print("cycle with pipeline = ", actual_time)
    print("energy with pipeline = ", total_energy_pipeline)
    print("Pipeline Compute Energy = ", sum(energy_pipeline) * batch_size)
    print(
        "Pipeline Feature Update Energy = ",
        (inputs[0] + outputs[-1]) * batch_size * 112.54 / 8 / 1e6,
    )
    print("Pipeline Utilization = ", util_pipeline / len(engine.layers))


def run_multi_layer_cnn(engine, batch_size=1):
    """
    Explore the hardware-aware mapping/allocation space and initiate BO.
    """
    metrics = engine._collect_layer_metrics(
        include_kernel_in_weights=True,
        clip_min_tiles=True,
    )
    workload = metrics.workloads
    compute = metrics.compute
    min_tiles = metrics.min_tiles

    logging.info(f"Minimum tiles per layer: {min_tiles}")

    start_time = time.time()

    allocator = TileAllocator(
        layers=engine.layers,
        compute_workloads=compute,
        min_tiles_per_layer=min_tiles,
        max_tiles_per_group=engine.hw["tile_num"],
        min_layers_per_group=1,
        max_layers_per_group=16,
        workload_shape=workload,
    )
    all_allocations = allocator.explore_allocations()
    analysis = allocator.analyze_allocations()

    layer_count = len(engine.layers)
    theoretical_space = 2 ** (layer_count - 1) if layer_count > 0 else 0

    logging.info("==================================================")
    logging.info(f"Network Layers (N): {layer_count}")
    logging.info(
        f"Theoretical exhaustive search space (2^(N-1)): {theoretical_space:,} combinations"
    )
    logging.info("==================================================")
    logging.info(f"Total valid allocations found: {analysis['total_allocations']}")

    engine._get_plot_dir()

    logging.info("Initializing Bayesian Optimizer...")
    engine.all_allocations = all_allocations

    evaluator = FitnessEvaluator(engine)
    callback = evaluator.cnn_multi_layer_fitness_callback

    optimizer = engine._build_optimizer(
        bounds=[(0, len(all_allocations) - 1)],
        evaluate_callback=callback,
        transformer=False,
        multi_layer=True,
        batch=(batch_size > 1),
        max_block=getattr(engine, "max_block", 1),
    )
    result, step = optimizer.run_optimization()

    logging.info("Running final CNN multi-layer evaluation for the best allocation...")
    final_report = evaluator._evaluate_cnn_multi_layer(
        result.x_opt,
        generate_artifacts=True,
        log_prefix="Final CNN Multi-Layer Evaluation",
    )

    end_time = time.time()
    total_seconds = end_time - start_time
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60

    logging.info(
        f"Bayesian Optimization Total Time: {minutes} min {seconds:.2f} sec"
    )

    return total_seconds, step, final_report["latency"]


def construct_allocation_space(engine, batch_size=1):
    """Backward-compatible wrapper for the older entry name."""
    return run_multi_layer_cnn(engine, batch_size=batch_size)
