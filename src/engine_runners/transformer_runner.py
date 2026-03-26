import copy
import logging
import math
import time

from src.optimization import FitnessEvaluator
from src.allocation import allocation_utils
from src.visualization import timeline_plot

def run_transformer_evaluation(engine):
    """
    Executes pipeline analysis specifically tailored for Transformer architectures.
    Includes operator grouping (Q, K, V and FFNs), nested scheduling for blocks/batches,
    and energy/cycle estimation.
    """
    # 1. Load parameters from config
    head_num = engine.model.get("head_num", 1)
    block = engine.model.get("block", 1)
    batch_size = engine.model.get("batch_size", 1)
    batch = 1 if batch_size > 1 else 0
    output_path_pipeline, output_path_no_pipeline = engine._generate_output_paths()
    metrics = engine._collect_layer_metrics(
        include_kernel_in_weights=True,
        clip_min_tiles=False,
    )
    workload = metrics.workloads
    compute = metrics.compute
    weights = metrics.weight_access
    min_tile_allocation = metrics.min_tiles

    logging.info(f"Min tile allocation: {min_tile_allocation}")

    allocation_dict = allocation_utils.greedy_tile_allocation(
        engine.layers,
        compute,
        min_tile_allocation,
        engine.hw["tile_num"] / block,
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
    alpha = engine._bo_value("alpha", 0.2)
    while i < len(engine.layers):
        lb = round(allocation[i] * 0.5) if round(allocation[i] * 0.5) != 0 else 1
        ub = min(
            allocation[i] + alpha * engine.hw["tile_num"] / block, allocation[i] * 2
        )
        bound.append((max(min_tile_allocation[i], lb), ub))
        selected_indices.append(i)
        if engine.layers[i] in FNN[1:]:
            i += 2
        elif engine.layers[i] in Projection:
            i += 3
        else:
            i += 1

    candidate_domains = engine._build_candidate_domains(bound, selected_indices)

    space_dimensions = [len(d) for d in candidate_domains]
    total_search_space = math.prod(space_dimensions)

    logging.info(f"Variables dimension (N): {len(bound)}")
    logging.info(f"Options per dimension: {space_dimensions}")
    logging.info(
        f"🚀 Total Search Space Size: {total_search_space:.2e} ({total_search_space} possible combinations)"
    )

    # 5. Execute Bayesian Optimization (Optional: Enable if needed, currently fast-forwarding to default assignment)
    evaluator = FitnessEvaluator(engine)

    callback = evaluator.transformer_fitness_callback

    logging.info("Initializing Bayesian Optimizer for Transformer Evaluation...")
    optimizer = engine._build_optimizer(
        bounds=bound,
        evaluate_callback=callback,
        alpha=alpha,
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
    engine._run_parallel_mapping(tile_allocation)

    # 7. Pipeline Flow & Bubble Analysis Initialization
    start_time = 0
    cal_time_list = []
    bubble_list = []
    layers_cal = []
    layers_bubble = []
    layers_weight_update = []
    A_start_time = 0

    end_time = {key: 0 for key in engine.layers}
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
                        engine.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[engine.layers.index(i)],
                            workload[engine.layers.index(i)],
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
                        engine.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[engine.layers.index(i)],
                            workload[engine.layers.index(i)],
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
                    ) = engine.pipeline_analyzer.pipeline_analysis(
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
                        engine.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[engine.layers.index(i)],
                            workload[engine.layers.index(i)],
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
                        engine.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[engine.layers.index(i)],
                            workload[engine.layers.index(i)],
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
            ) = engine.pipeline_analyzer.pipeline_analysis(
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
                    engine.pipeline_analyzer.parse_dataspace(
                        output_path_pipeline[engine.layers.index(i)],
                        workload[engine.layers.index(i)],
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
            ) = engine.pipeline_analyzer.pipeline_analysis(
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
            ) = engine.pipeline_analyzer.pipeline_analysis(
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
            ) = engine.pipeline_analyzer.pipeline_analysis(
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

    plot_path = engine._get_plot_dir()
    timeline_plot.plot_combined_timelines_block_batch(
        plot_path,
        layers_cal,
        layers_bubble,
        layers_weight_update,
        actual_time=actual_time,
    )
    # 9. Energy & Cycle Calculation
    pipeline_access = [
        engine.analyzer.input_output_gen(path) for path in output_path_pipeline
    ]
    inputs = [d[0]["inputs"] for d in pipeline_access]
    outputs = [d[0]["outputs"] for d in pipeline_access]

    energy_pipeline = []

    for i in range(len(engine.layers)):
        cim_utilized = engine.analyzer.extract_cim_utilized_instances(
            output_path_pipeline[i]
        )
        cim_write = engine.analyzer.extract_cim_write_energy(output_path_pipeline[i])
        vec_access = engine.analyzer.extract_vector_access_by_module(
            output_path_pipeline[i], "random_fill", "cim_unit"
        )

        e_pipe = engine.analyzer.get_total_energy(
            output_path_pipeline[i] + "/timeloop-mapper.stats.txt"
        ) - (cim_utilized * cim_write * vec_access / 1e6)
        energy_pipeline.append(e_pipe)


    # Scale by head_num for attention layers
    for i in Mh_attention:
        idx = engine.layers.index(i)
        weights[idx] *= head_num
        energy_pipeline[idx] *= head_num

    cycle_pipeline = actual_time
    energy_const = 112.54 / 8 / 1e6

    idx_Q = engine.layers.index("Q")
    idx_A = engine.layers.index("A")
    idx_FFN2 = engine.layers.index("FFN2")

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


def run_multi_layer_transformer_batch(engine, batch_size=None, batch=True):
    """
    Executes multi-layer Transformer pipeline scheduling with batch processing.
    Handles fine-grained weight update overlapping, block-by-block dependencies,
    and calculates cycles/energy with or without pipeline parallelism.
    """
    batch_size = batch_size or engine.model.get("batch_size", 1)
    head_num = engine.model.get("head_num", 1)
    block = engine.model.get("block", 1)
    output_path_pipeline, output_path_no_pipeline = engine._generate_output_paths()
    metrics = engine._collect_layer_metrics(
        include_kernel_in_weights=False,
        clip_min_tiles=False,
    )
    workload = metrics.workloads
    compute = metrics.compute
    weight_access = metrics.weight_access
    min_tile_allocation = metrics.min_tiles

    logging.info(f"Minimum tile allocation per layer: {min_tile_allocation}")

    # Compute max blocks dynamically based on available tiles
    max_block_tiles = sum(min_tile_allocation) + (head_num - 1) * (
        min_tile_allocation[engine.layers.index("A")]
        + min_tile_allocation[engine.layers.index("Z0")]
    )
    max_block = (
        math.floor(engine.hw["tile_num"] / max_block_tiles)
        if max_block_tiles > 0
        else 1
    )

    allocation_dict = allocation_utils.greedy_tile_allocation(
        engine.layers,
        weight_access,
        min_tile_allocation,
        engine.hw["tile_num"],
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
    alpha = engine._bo_value("alpha", 0.2)

    selected_indices = []
    while i < len(engine.layers):
        lb = round(allocation[i] * 0.5) if round(allocation[i] * 0.5) != 0 else 1
        ub = min(allocation[i] + engine.hw["tile_num"] * alpha, allocation[i] * 3)
        bound.append((max(min_tile_allocation[i], lb), ub))
        selected_indices.append(i)
        if engine.layers[i] in FNN[1:]:
            i += 2
        elif engine.layers[i] in Projection:
            i += 3
        else:
            i += 1

    candidate_domains = engine._build_candidate_domains(bound, selected_indices)

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
    evaluator = FitnessEvaluator(engine)
    callback = evaluator.transformer_multi_batch_fitness_callback
    optimizer = engine._build_optimizer(
        bounds=bound,
        evaluate_callback=callback,
        alpha=alpha,
        transformer=True,
        multi_layer=True,
        batch=batch,
        max_block=getattr(engine, "max_block", 1),
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
    engine._run_parallel_mapping(tile_allocation)

    pipeline_access = [
        engine.analyzer.input_output_gen(path) for path in output_path_pipeline
    ]
    inputs = [d[0]["inputs"] for d in pipeline_access]
    outputs = [d[0]["outputs"] for d in pipeline_access]
    weights = [d[0]["weights"] for d in pipeline_access]

    # 5. Pipeline State Variables
    start_time = 0
    A_start_time = 0
    width = 256 * 8 * 2

    Projection_dataspace = {key: [] for key in Projection}
    end_time = {key: 0 for key in engine.layers}
    start_time_dict = {key: 0 for key in engine.layers}
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
                        weight_access[engine.layers.index("Q")] / width
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
                        engine.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[engine.layers.index("Q")],
                            workload[engine.layers.index("Q")],
                            start_time=Q_start_time,
                        )
                    )
                    current_batch_cal.append((Q_start_time, last_t))
                    current_batch_bubble.append({})
                    time_scale.append(cur_ts)
                    end_time["Q"] = last_t

                    # 1.2 K Weight update
                    weight_update_cost = (
                        weight_access[engine.layers.index("K")] / width
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
                        engine.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[engine.layers.index("K")],
                            workload[engine.layers.index("K")],
                            start_time=K_start_time,
                        )
                    )
                    current_batch_cal.append((K_start_time, last_t))
                    current_batch_bubble.append({})
                    time_scale.append(cur_ts)
                    end_time["K"] = last_t

                    # 1.3 V Weight update
                    weight_update_cost = (
                        weight_access[engine.layers.index("V")] / width
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
                        engine.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[engine.layers.index("V")],
                            workload[engine.layers.index("V")],
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
                            engine.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[engine.layers.index(i)],
                                workload[engine.layers.index(i)],
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
                            engine.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[engine.layers.index(i)],
                                workload[engine.layers.index(i)],
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
                    ) = engine.pipeline_analyzer.pipeline_analysis(
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
                        weight_access[engine.layers.index("Z1")] / width
                    )
                    current_batch_weight.append(
                        (weight_update_start, weight_update_cost)
                    )
                    weight_update_start += weight_update_cost
                    pipeline_weight_update_cost += weight_update_cost
                    start_time = max(weight_update_start, start_time)

                    _, cur_ts, _, _, FFN_dataspace["Z1"] = (
                        engine.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[engine.layers.index("Z1")],
                            workload[engine.layers.index("Z1")],
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
                    ) = engine.pipeline_analyzer.pipeline_analysis(
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
                        weight_access[engine.layers.index("FFN1")] / width
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
                            (2 * inputs[engine.layers.index("FFN1")])
                            * 112.54
                            / 8
                            / 1e6
                        )
                        pipeline_feature += (
                            (2 * inputs[engine.layers.index("FFN1")])
                            * 112.54
                            / 8
                            / 1e6
                        )

                    _, cur_ts, _, _, FFN_dataspace["FFN1"] = (
                        engine.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[engine.layers.index("FFN1")],
                            workload[engine.layers.index("FFN1")],
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
                    ) = engine.pipeline_analyzer.pipeline_analysis(
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
                        weight_access[engine.layers.index("FFN2")] / width
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
                            (2 * inputs[engine.layers.index("FFN2")])
                            * 112.54
                            / 8
                            / 1e6
                        )
                        pipeline_feature += (
                            (2 * inputs[engine.layers.index("FFN2")])
                            * 112.54
                            / 8
                            / 1e6
                        )

                    _, cur_ts, _, _, FFN_dataspace["FFN2"] = (
                        engine.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[engine.layers.index("FFN2")],
                            workload[engine.layers.index("FFN2")],
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
                    ) = engine.pipeline_analyzer.pipeline_analysis(
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
                            weight_access[engine.layers.index(proj_key)] / width
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
                            engine.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[engine.layers.index(proj_key)],
                                workload[engine.layers.index(proj_key)],
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
                        ) = engine.pipeline_analyzer.pipeline_analysis(
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
                                engine.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[engine.layers.index(i)],
                                    workload[engine.layers.index(i)],
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
                                engine.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[engine.layers.index(i)],
                                    workload[engine.layers.index(i)],
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
                    ) = engine.pipeline_analyzer.pipeline_analysis(
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
                            engine.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[engine.layers.index(f_key)],
                                workload[engine.layers.index(f_key)],
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
                        ) = engine.pipeline_analyzer.pipeline_analysis(
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
                                (2 * inputs[engine.layers.index(f_key)])
                                * 112.54
                                / 8
                                / 1e6
                            )
                            pipeline_feature += (
                                (2 * inputs[engine.layers.index(f_key)])
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
                            engine.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[engine.layers.index(proj_key)],
                                workload[engine.layers.index(proj_key)],
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
                        ) = engine.pipeline_analyzer.pipeline_analysis(
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
                            ) = engine.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[engine.layers.index(i)],
                                workload[engine.layers.index(i)],
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
                            ) = engine.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[engine.layers.index(i)],
                                workload[engine.layers.index(i)],
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
                    ) = engine.pipeline_analyzer.pipeline_analysis(
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
                        weight_access[engine.layers.index("Z1")] / width
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
                        engine.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[engine.layers.index("Z1")],
                            workload[engine.layers.index("Z1")],
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
                    ) = engine.pipeline_analyzer.pipeline_analysis(
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
                        weight_access[engine.layers.index("FFN1")] / width
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
                        engine.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[engine.layers.index("FFN1")],
                            workload[engine.layers.index("FFN1")],
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
                    ) = engine.pipeline_analyzer.pipeline_analysis(
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
                        weight_access[engine.layers.index("FFN2")] / width
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
                        engine.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[engine.layers.index("FFN2")],
                            workload[engine.layers.index("FFN2")],
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
                    ) = engine.pipeline_analyzer.pipeline_analysis(
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
                            ) = engine.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[engine.layers.index(i)],
                                workload[engine.layers.index(i)],
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
                            ) = engine.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[engine.layers.index(i)],
                                workload[engine.layers.index(i)],
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
                    ) = engine.pipeline_analyzer.pipeline_analysis(
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
                        engine.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[engine.layers.index("Z1")],
                            workload[engine.layers.index("Z1")],
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
                    ) = engine.pipeline_analyzer.pipeline_analysis(
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
                        engine.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[engine.layers.index("FFN1")],
                            workload[engine.layers.index("FFN1")],
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
                    ) = engine.pipeline_analyzer.pipeline_analysis(
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
                        engine.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[engine.layers.index("FFN2")],
                            workload[engine.layers.index("FFN2")],
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
                    ) = engine.pipeline_analyzer.pipeline_analysis(
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
        idx_Q = engine.layers.index("Q")
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
    plot_path = engine._get_plot_dir()
    timeline_plot.plot_combined_timelines_block_batch(
        plot_path,
        layers_cal,
        layers_bubble,
        layers_weight_update,
        actual_time=actual_time,
    )

    cal_time, weight_time, overlap_time = allocation_utils.compute_time_statistics(
        layers_cal, layers_weight_update
    )
    logging.info(
        f"Cal-only Time: {cal_time} | Weight-only Time: {weight_time} | Overlap Time: {overlap_time}"
    )

    # 8. Un-pipelined baseline calculation
    no_pipeline_access = [
        engine.analyzer.input_output_gen(path) for path in output_path_no_pipeline
    ]
    inputs_no_pipeline = [d[0]["inputs"] for d in no_pipeline_access]
    outputs_no_pipeline = [d[0]["outputs"] for d in no_pipeline_access]
    weights_no_pipeline = [d[0]["weights"] for d in no_pipeline_access]

    write_weight_energy_pipeline = []
    energy_pipeline = []

    for i in range(len(engine.layers)):
        cim_util_pipe = engine.analyzer.extract_cim_utilized_instances(
            output_path_pipeline[i]
        )
        cim_write_pipe = (
            engine.analyzer.extract_cim_write_energy(output_path_pipeline[i]) / 1e6
        )
        energy_pipeline.append(
            engine.analyzer.get_total_energy(
                output_path_pipeline[i] + "/timeloop-mapper.stats.txt"
            )
            - (cim_util_pipe * cim_write_pipe)
        )
        write_weight_energy_pipeline.append(cim_util_pipe * cim_write_pipe)


    for i in Mh_attention:
        idx_i = engine.layers.index(i)
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
    logging.info(f"Overlap cycles (Pipeline) = {overlap_time:.2f}")
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
    engine, batch_size=None, batch=True, grouped_indices=[]
):
    """
    [Ultimate Edition] Multi-Group Transformer Pipeline Evaluation
    Handles dynamic cross-block grouping, topological pipeline tracking,
    and detailed energy/power assessments!
    """
    batch_size = batch_size or engine.model.get("batch_size", 1)
    head_num = engine.model.get("head_num", 1)
    block = engine.model.get("block", 1)
    output_path_pipeline, output_path_no_pipeline = engine._generate_output_paths()
    metrics = engine._collect_layer_metrics(
        include_kernel_in_weights=True,
        clip_min_tiles=False,
        attention_weight_scale=head_num,
    )
    workload = metrics.workloads
    weight_access = metrics.weight_access
    engine.compute = metrics.compute

    # 2. Dynamic 1D Grouping
    hw_total_tiles = engine.hw.get("tile_num", 1344)
    arch_size_bits = (
        hw_total_tiles
        * engine.hw.get("macro_num", 1)
        * engine.hw.get("core_num", 1)
        * engine.hw.get("array_col", 1)
        * engine.hw.get("array_row", 1)
        * engine.hw.get("cim_depth", 1)
    )
    """
    grouped_indices = allocation_utils.capacity_aware_transformer_grouping(
        weights=weight_access, layer_names=engine.layers, arch_size=arch_size_bits, ops_per_block=8
    )
    """
    # 3. Setup Bounds & Alpha Dimension Reduction
    tile_capacity_bits = (
        engine.hw.get("macro_num", 16)
        * engine.hw.get("core_num", 4)
        * engine.hw.get("array_col", 128)
        * engine.hw.get("array_row", 128)
        * engine.hw.get("cim_depth", 1)
    )

    true_min_t = []
    for i, layer in enumerate(engine.layers):
        layer_name = layer.split(".")[0]
        if layer_name in ["A", "Z0"]:
            min_t = max(1, int(math.ceil(weight_access[i] / tile_capacity_bits)))
        else:
            min_t = metrics.min_tiles[i]
        true_min_t.append(min_t)

    allocation_dict = {}

    compute_list = []
    for i, layer in enumerate(engine.layers):
        if hasattr(engine, "compute") and engine.compute:
            compute_list.append(engine.compute[i])
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
            layer_name = engine.layers[layer_i].split(".")[0]
            allocation_dict[layer_name] = true_min_t[layer_i] + extra_tiles[idx]

    bounds = []
    var_map = []
    alpha = engine._bo_value("alpha", 0.2)
    Projection = ["Q", "K", "V"]

    i = 0
    var_idx = 0
    while i < len(engine.layers):
        layer_name = engine.layers[i].split(".")[0]
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
                    i < len(engine.layers)
                    and engine.layers[i].split(".")[0] in Projection
                ):
                    var_map.append(var_idx)
                    i += 1
            elif layer_name in ["FFN1", "FFN2"]:
                while i < len(engine.layers) and engine.layers[i].split(".")[0] in [
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
            j_name = engine.layers[j].split(".")[0]
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
                i < len(engine.layers) and engine.layers[i].split(".")[0] in Projection
            ):
                var_map.append(var_idx)
                i += 1
        elif layer_name in ["FFN1", "FFN2"]:
            while i < len(engine.layers) and engine.layers[i].split(".")[0] in [
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
    evaluator = FitnessEvaluator(engine)

    optimizer = engine._build_optimizer(
        bounds=bounds,
        evaluate_callback=lambda x: evaluator.multi_group_transformer_fitness_callback(
            x, grouped_indices, engine.layers, var_map
        ),
        alpha=alpha,
        tile_num=hw_total_tiles,
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

    default_macros = engine.hw.get(
        "macro_num", 1
    )
    tile_allocation = [1] * len(engine.layers)
    macro_allocation = [default_macros] * len(engine.layers)

    for i, layer in enumerate(engine.layers):
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
    engine._run_parallel_mapping(tile_allocation, macro_num=macro_allocation)

    # Parse inputs/outputs for Energy tracking
    pipeline_access = [
        engine.analyzer.input_output_gen(path) for path in output_path_pipeline
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

    end_time_dict = [{layer: 0 for layer in engine.layers} for _ in range(batch_size)]
    Dataspace = [{key: [] for key in engine.layers} for _ in range(batch_size)]
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

    end_time_dict = [{layer: 0 for layer in engine.layers} for _ in range(batch_size)]
    Dataspace = [{layer: [] for layer in engine.layers} for _ in range(batch_size)]
    batch_end_time = [0 for _ in range(batch_size)]

    hw_available_tiles = engine.hw.get("tile_num", 1344)
    weight_noc_ready_time = 0
    active_ops = (
        []
    )

    def get_req_tiles(l_idx):
        t = tile_allocation[l_idx]
        name = engine.layers[l_idx].split(".")[0]
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
                curr_layer = engine.layers[layer_idx]
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
                        engine.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[layer_idx],
                            workload[layer_idx],
                            start_time=start_time,
                        )
                    )
                    current_stage_ops = [
                        engine.layers[i].split(".")[0] for i in group
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
                        ) = engine.pipeline_analyzer.pipeline_analysis(
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
        idx_Q = engine.layers.index("Q")
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
    plot_path = engine._get_plot_dir()
    timeline_plot.plot_combined_timelines_block_batch(
        plot_path,
        layers_cal,
        layers_bubble,
        layers_weight_update,
        layer_names=engine.layers,
        actual_time=actual_time,
    )

    cal_time, weight_time, overlap_time = allocation_utils.compute_time_statistics(
        layers_cal, layers_weight_update
    )
    logging.info(
        f"Cal-only Time: {cal_time} | Weight-only Time: {weight_time} | Overlap Time: {overlap_time}"
    )
    logging.info(f"🎉 Pipeline Final Latency: {actual_time:.2f} cycles")

    # 8. Un-pipelined baseline calculation
    no_pipeline_access = [
        engine.analyzer.input_output_gen(path) for path in output_path_no_pipeline
    ]
    inputs_no_pipeline = [d[0]["inputs"] for d in no_pipeline_access]
    outputs_no_pipeline = [d[0]["outputs"] for d in no_pipeline_access]
    weights_no_pipeline = [d[0]["weights"] for d in no_pipeline_access]

    write_weight_energy_pipeline = []
    energy_pipeline = []

    for i in range(len(engine.layers)):
        cim_util_pipe = engine.analyzer.extract_cim_utilized_instances(
            output_path_pipeline[i]
        )
        cim_write_pipe = (
            engine.analyzer.extract_cim_write_energy(output_path_pipeline[i]) / 1e6
        )
        energy_pipeline.append(
            engine.analyzer.get_total_energy(
                output_path_pipeline[i] + "/timeloop-mapper.stats.txt"
            )
            - (cim_util_pipe * cim_write_pipe)
        )
        write_weight_energy_pipeline.append(cim_util_pipe * cim_write_pipe)


    Mh_attention = ["A", "Z0"]
    for i in Mh_attention:
        idx_i = engine.layers.index(i)
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
    logging.info(f"Overlap cycles (Pipeline) = {overlap_time:.2f}")
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
