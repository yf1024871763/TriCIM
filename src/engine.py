import os
import math
import logging
from src.pipeline_analyzer import PipelineAnalyzer
from src.ParallelExecutor import ParallelExecutor
from src.Bayes_opt import Bayesian_Optimizer
import src.function as function
import src.plot as plot
from src.fitness import FitnessEvaluator

class TriCIMEngine:
    def __init__(self, config):
        self.config = config
        self.hw = config['hardware']
        self.model = config['model']
        self.paths = config['paths']
        
        self.dnn = self.model['dnn']
        self.layers = [] # Will be populated by pipeline_analyzer
        
        # Initialize the underlying parser and analyzer
        self.pipeline_analyzer = PipelineAnalyzer(config)
        self.layers = self.pipeline_analyzer.layers
        self.analyzer = self.pipeline_analyzer.analyzer # Access to underlying Timeloop analysis
        
    def _generate_output_paths(self):
        """Dynamically generate output paths based on config to avoid hardcoding."""
        output_dir = self.paths['output_root']
        pipeline_paths = [os.path.join(output_dir, f"pipeline_{i}-isaac-{self.dnn}-{self.layers[i]}") for i in range(len(self.layers))]
        origin_paths = [os.path.join(output_dir, f"pipeline_origin-isaac-{self.dnn}-{self.layers[i]}") for i in range(len(self.layers))]
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
            if 'R' in workload[i].keys() or 'S' in workload[i].keys():
                workload_size = workload[i]['C'] * workload[i]['M'] * workload[i]['R'] * workload[i]['S'] * self.hw['precision']
            else:
                workload_size = workload[i]['C'] * workload[i]['M'] * self.hw['precision']
            
            compute.append(workload[i].get('C',1) * workload[i].get('M',1) * workload[i].get('R',1) * workload[i].get('S',1) *
                           workload[i].get('P',1) * workload[i].get('Q',1))
            weights_access.append(workload_size)
                
        # 2. Minimum Tile Allocation calculation
        min_tile_allocation = [function.tile_allocation(
            workload[i], macro_num=self.hw['macro_num'], core_num=self.hw['core_num'],
            array_col=self.hw['array_col'], array_row=self.hw['array_row'], 
            cim_depth=self.hw['cim_depth'], precision=self.hw['precision']
        ) for i in range(len(self.layers))]
        
        logging.info(f"Min tiles required: {min_tile_allocation}")

        # 3. Greedy Allocation
        allocation = function.greedy_tile_allocation(
            self.layers, compute, min_tile_allocation, self.hw['tile_num'], self.model['head_num']
        )
        allocation_list = list(allocation.values())
        logging.info(f"Greedy allocation: {allocation_list}")
        
        # 4. Bayesian Optimization Setup
        bound = []
        for i, j in enumerate(allocation_list):
            lb = round(allocation_list[i] * 0.5) if round(allocation_list[i] * 0.5) != 0 else 1
            if min_tile_allocation[i] < j:
                bound.append((max(min_tile_allocation[i], lb), min(allocation_list[i] + 0.02 * self.hw['tile_num'], allocation_list[i] * 2)))
            else:
                bound.append((allocation_list[i], allocation_list[i] + 1))
        

        evaluator = FitnessEvaluator(self)
        
        callback = evaluator.cnn_fitness_callback 
        
        logging.info("Initializing Bayesian Optimizer for CNN Evaluation...")
        optimizer = Bayesian_Optimizer(
            bounds=bound,
            evaluate_callback=callback,  # 传入回调
            n_calls=100,
            dnn_name=self.dnn,
            alpha=0.2,

            tile_num=self.hw['tile_num'],
            layers=self.layers,
            head_num=self.model.get('head_num', 1),
            block_num=self.model.get('block', 1),
            transformer=False,       
            multi_layer=False,       
            batch=False,             
            max_block=1              
        )
        result, _ = optimizer.run_optimization()
        tile_allocation = result.x_opt.astype(int)
        


        # 5. Parallel Execution
        logging.info("Starting Parallel Execution...")
        executor = ParallelExecutor(
            layer_num=len(self.layers),
            arch_path=self.paths['arch_root'],
            tile_num=tile_allocation,
            layers=self.layers, DNN=self.dnn, MACRO_NAME='isaac_isca_2016'
        )
        executor.run_parallel()
        
        # 6. Pipeline Dependency & Bubble Analysis
        Dataspace = [[] for _ in range(len(self.layers))]
        cal_time_list = []
        bubble_list = []
        time_scale = []
        start_time = 0
        
        for i in range(len(self.layers)):
            _, cur_time_scale, _, last_time, Dataspace[i] = self.pipeline_analyzer.parse_dataspace(
                output_path_pipeline[i], workload[i], start_time=start_time
            )
            time_scale.append(cur_time_scale)
            if i == 0:
                cal_time_list.append((start_time, last_time))
                bubble_list.append({})
                
        for i in range(len(self.layers) - 1):
            logging.info(f"Processing Pipeline stage: {self.layers[i]} -> {self.layers[i+1]}")
            actual_time, _, _, _, _, bubble, cal_time, _, Dataspace[i+1] = self.pipeline_analyzer.pipeline_analysis(
                time_scale[i+1], self.pipeline_analyzer.maxpool[i], self.pipeline_analyzer.fc[i+1],
                Input_dataspace=Dataspace[i], Output_dataspace=Dataspace[i+1], shortcut=self.pipeline_analyzer.shortcut[i+1]
            )
            cal_time_list.append(cal_time)
            bubble_list.append(bubble)

        # Plotting
        plot_dir = os.path.join(self.paths['script_root'], f"ISAAC-{self.dnn}")
        os.makedirs(plot_dir, exist_ok=True)
        plot.plot_bubble(plot_dir, cal_time_list, bubble_list, actual_time=actual_time) 
        
        # 7. Energy & Cycle Calculation
        self._calculate_cnn_energy_metrics(output_path_pipeline, output_path_no_pipeline, actual_time, weights_access)

    def _calculate_cnn_energy_metrics(self, output_path_pipeline, output_path_no_pipeline, actual_time, weights_access):
        """Extracted Energy logic to keep run_cnn_evaluation clean."""
        # ... (这里放你原本 CNN_pipeline_analyzer 最后那 30 行计算 Energy/Utilization 的打印代码)
        # 例如：
        logging.info(f'Cycle with pipeline = {actual_time}')
        pipeline_access = []
        util_no_pipeline = 0
        util_pipeline = 0
        for i in range(len(self.layers)):
            pipeline_access.append(self.analyzer.input_output_gen(output_path_pipeline[i]))
            util_pipeline += self.analyzer.get_utilization(output_path_pipeline[i])
        inputs = [d[0]['inputs'] for d in pipeline_access]
        outputs = [2*d[0]['outputs'] for d in pipeline_access]
        weights = [d[0]['weights'] for d in pipeline_access]

        no_pipeline_access = []
        for i in range(len(self.layers)):
            no_pipeline_access.append(self.analyzer.input_output_gen(output_path_no_pipeline[i]))
        inputs_no_pipeline = [d[0]['inputs'] for d in no_pipeline_access]
        outputs_no_pipeline = [2*d[0]['outputs'] for d in no_pipeline_access]
        weights_no_pipeline = [d[0]['weights'] for d in no_pipeline_access]                                                                                                           


        cycle_no_pipeline = 0
        weight = 0
        for i,j in enumerate(output_path_no_pipeline):
            cycle_no_pipeline += self.analyzer.get_cycle(j)+(weights_access[i])/256/8/16  
            util_no_pipeline += self.analyzer.get_utilization(j)
            #print(self.analyzer.get_cycle(j))
            weight+=(weights_access[i])/256/8/16 
        energy_pipeline = []
        energy_origin = []
        write_weight_energy_no_pipeline=[]
        for i in range(len(self.layers)):
            energy_pipeline.append(self.analyzer.get_total_energy(output_path_pipeline[i]+"/timeloop-mapper.stats.txt")-self.analyzer.extract_cim_utilized_instances(output_path_pipeline[i])*self.analyzer.extract_cim_write_energy(output_path_pipeline[i])*self.analyzer.extract_vector_access_by_module(output_path_pipeline[i],'random_fill','cim_unit')/1e6)
            #print(self.analyzer.get_total_energy(output_path_pipeline[i]+"/timeloop-mapper.stats.txt"),self.analyzer.extract_cim_utilized_instances(output_path_pipeline[i])*self.analyzer.extract_cim_write_energy(output_path_pipeline[i])*self.analyzer.extract_vector_access_by_module(output_path_pipeline[i],'random_fill','cim_unit')/1e6)
            energy_origin.append(self.analyzer.get_total_energy(output_path_no_pipeline[i]+"/timeloop-mapper.stats.txt")-(self.analyzer.extract_cim_utilized_instances(output_path_no_pipeline[i])*self.analyzer.extract_cim_write_energy(output_path_no_pipeline[i])/1e6))
            write_weight_energy_no_pipeline.append(self.analyzer.extract_cim_utilized_instances(output_path_no_pipeline[i])*self.analyzer.extract_cim_write_energy(output_path_no_pipeline[i])/1e6)
        total_energy_pipeline = sum(energy_pipeline) + (inputs[0]+outputs[-1])*112.54/8/1e6

        total_energy_pipeline_origin = sum(energy_origin) + (sum(inputs_no_pipeline)+sum(outputs_no_pipeline)+sum(weights_access))*112.54/8/1e6+sum(write_weight_energy_no_pipeline)
        print('cycle without pipeline = ',cycle_no_pipeline,'Weight_update = ',weight,'cycle with pipeline = ',actual_time)
        print('energy without pipeline = ',total_energy_pipeline_origin,'energy with pipeline = ',total_energy_pipeline)

        print('Pipeline Compute Energy = ',sum(energy_pipeline))
        print('Pipeline Feature Update Energy = ',(inputs[0]+outputs[-1])*112.54/8/1e6)

        print('NoPipeline Compute Energy = ',sum(energy_origin))
        print('NoPipeline Weight Update Energy = ',sum(weights_access)*112.54/8/1e6)
        print('NoPipeline Write Weight Energy = ',sum(write_weight_energy_no_pipeline))
        print('NoPipeline Feature Update Energy = ',(sum(inputs_no_pipeline)+sum(outputs_no_pipeline))*112.54/8/1e6)
        print('No_pipeline Utilization = ',util_no_pipeline/len(self.layers),'Pipeline Utilization = ',util_pipeline/len(self.layers))

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
            
            if 'R' in wl.keys() or 'S' in wl.keys():
                workload_size = wl['C'] * wl['M'] * wl.get('R', 1) * wl.get('S', 1) * self.hw['precision']
            else:
                workload_size = wl['C'] * wl['M'] * self.hw['precision']

            compute.append(wl.get('C', 1) * wl.get('M', 1) * wl.get('R', 1) * wl.get('S', 1) *
                           wl.get('P', 1) * wl.get('Q', 1))
            workload_group.append(workload_size)

        # 2. Calculate minimum tiles required per layer
        for i, layer in enumerate(self.layers):
            tile_alloc = function.tile_allocation(
                workload[i], 
                macro_num=self.hw['macro_num'], 
                core_num=self.hw['core_num'],
                array_col=self.hw['array_col'], 
                array_row=self.hw['array_row'],
                cim_depth=self.hw['cim_depth'], 
                precision=self.hw['precision']
            )
            # Clip min_tiles to the maximum available tiles
            min_tiles.append(min(tile_alloc, self.hw['tile_num']))
            
        logging.info(f"Minimum tiles per layer: {min_tiles}")

        start_time = time.time()        
        
        # 3. Explore allocations using TileAllocator
        allocator = TileAllocator(
            layers=self.layers,
            compute_workloads=compute,
            min_tiles_per_layer=min_tiles,
            max_tiles_per_group=self.hw['tile_num'],
            min_layers_per_group=1,   # Minimum 1 layer per group
            max_layers_per_group=16,  # Maximum 16 layers per group
            workload_shape=workload
        )
        all_allocations = allocator.explore_allocations()
        analysis = allocator.analyze_allocations()
        
        # 4. Search Space Analysis Logging
        N = len(self.layers)
        theoretical_space = 2 ** (N - 1) if N > 0 else 0
        
        logging.info("==================================================")
        logging.info(f"Network Layers (N): {N}")
        logging.info(f"Theoretical exhaustive search space (2^(N-1)): {theoretical_space:,} combinations")
        logging.info("==================================================")
        logging.info(f"Total valid allocations found: {analysis['total_allocations']}")

        # 5. Setup dynamic paths
        savepath = os.path.join(self.paths.get('script_root', os.path.dirname(__file__)), f"ISAAC-{self.dnn}")
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
            tile_num=self.hw.get('tile_num', 1344),
            layers=self.layers,
            head_num=self.model.get('head_num', 1),
            block_num=self.model.get('block', 1),
            transformer=False,
            multi_layer=True,
            batch=(batch_size > 1),
            max_block=getattr(self, 'max_block', 1)
        )
        
        result, step = optimizer.run_optimization()
        
        # 7. Time tracking and return
        end_time = time.time()
        total_seconds = end_time - start_time
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        
        logging.info(f"Bayesian Optimization Total Time: {minutes} min {seconds:.2f} sec")
        
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
        head_num = self.model.get('head_num', 1)
        block = self.model.get('block', 1)
        batch_size = self.model.get('batch_size', 1)
        batch = 1 if batch_size > 1 else 0
        output_path_pipeline, output_path_no_pipeline = self._generate_output_paths()

        # 2. Extract Workload & Compute limits
        workload = []
        compute = []
        weights = []
        for i, layer in enumerate(self.layers):
            wl = self.analyzer.get_workload(layer)
            workload.append(wl)
            
            if 'R' in wl.keys() or 'S' in wl.keys():
                workload_size = wl['C'] * wl['M'] * wl.get('R', 1) * wl.get('S', 1) * self.hw['precision']
            else:
                workload_size = wl['C'] * wl['M'] * self.hw['precision']
                
            compute.append(wl.get('C', 1) * wl.get('M', 1) * wl.get('R', 1) * wl.get('S', 1) *
                           wl.get('P', 1) * wl.get('Q', 1))
            weights.append(wl.get('C', 1) * wl.get('M', 1) * wl.get('R', 1) * wl.get('S', 1) * self.hw['precision'])

        # 3. Minimum Tile Allocation & Greedy Allocation
        min_tile_allocation = []
        for i, layer in enumerate(self.layers):
            min_tiles = function.tile_allocation(
                workload[i], macro_num=self.hw['macro_num'], core_num=self.hw['core_num'],
                array_col=self.hw['array_col'], array_row=self.hw['array_row'], 
                cim_depth=self.hw['cim_depth'], precision=self.hw['precision']
            )
            min_tile_allocation.append(min_tiles)

        logging.info(f"Min tile allocation: {min_tile_allocation}")

        allocation_dict = function.greedy_tile_allocation(
            self.layers, compute, min_tile_allocation, 
            self.hw['tile_num'] / block, head_num, transformer=True
        )
        allocation = [math.ceil(x) for x in allocation_dict.values()]
        logging.info(f"Greedy allocation: {allocation}")

        # 4. Bayesian Optimization Bounds Setup
        Projection = ['Q', 'K', 'V']
        Mh_attention = ['A', 'Z0']
        FNN = ['Z1', 'FFN1', 'FFN2']
        
        bound = []
        i = 0
        alpha = 0.2
        while i < len(self.layers):
            lb = round(allocation[i] * 0.5) if round(allocation[i] * 0.5) != 0 else 1
            bound.append((max(min_tile_allocation[i], lb), min(allocation[i] + alpha * self.hw['tile_num'] / block, allocation[i] * 2)))
            if self.layers[i] in FNN[1:]:
                i += 2
            elif self.layers[i] in Projection:
                i += 3
            else:
                i += 1

        space_dimensions = [int(ub) - int(lb) + 1 for lb, ub in bound]
        total_search_space = math.prod(space_dimensions)

        logging.info(f"Variables dimension (N): {len(bound)}")
        logging.info(f"Options per dimension: {space_dimensions}")
        logging.info(f"🚀 Total Search Space Size: {total_search_space:.2e} ({total_search_space} possible combinations)")

        # 5. Execute Bayesian Optimization (Optional: Enable if needed, currently fast-forwarding to default assignment)
        evaluator = FitnessEvaluator(self)

        callback = evaluator.transformer_fitness_callback

        logging.info("Initializing Bayesian Optimizer for Transformer Evaluation...")
        optimizer = Bayesian_Optimizer(
            bounds=bound,
            evaluate_callback=callback,   # 传入 fitness 中的回调函数
            n_calls=100,
            dnn_name=self.dnn,
            alpha=alpha,
            
            # --- 传入原版的约束控制参数，供 BO 内部判断 ---
            tile_num=self.hw['tile_num'],
            layers=self.layers,
            head_num=head_num,
            block_num=block,
            transformer=True,        
            multi_layer=False,            # 这是基础 Transformer，不是多层混合
            batch=(batch_size > 1),       # 布尔值标志位
            max_block=1                   # 默认 1
        )
        
        # 运行优化
        result, _ = optimizer.run_optimization()
        tile_allocation = result.x_opt.astype(int)
        x0, x1, x3, x6, x7 = tile_allocation

        # Structure variable mapping (QKV share mapping, FFN share mapping)
        tile_allocation = [
            x0,       # x0 (e.g. initial layer)
            x1,       # x1 (Q)
            x1,       # x2 = x1 (K)
            x3,       # x3 (V)
            x3,       # x4 = x3 (A)
            x3,       # x5 = x3 (Z0)
            x6,       # x6 (Z1)
            x7        # x7 (FFN1/2)
        ]

        # 6. Parallel Hardware Simulation
        logging.info("Starting Parallel Execution...")
        executor = ParallelExecutor(
            layer_num=len(self.layers),
            arch_path=self.paths['arch_root'],
            tile_num=tile_allocation,
            layers=self.layers, DNN=self.dnn, MACRO_NAME='isaac_isca_2016'
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
                    start_time = max(end_time['K'], 0)
                    for i in Projection:
                        _, cur_time_scale, _, last_time, Projection_dataspace[i] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index(i)], workload[self.layers.index(i)], start_time=start_time
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
                    start_time = max(batch_start_time[batch], 0) if batch == 0 else max(start_time, end_time['K'])
                    for i in Projection:
                        _, cur_time_scale, _, last_time, Projection_dataspace[i] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index(i)], workload[self.layers.index(i)], start_time=start_time
                        )  
                        actual_time, _, _, _, _, bubble, cal_time, _, Projection_dataspace[i] = self.pipeline_analyzer.pipeline_analysis(
                            cur_time_scale, 0, 0, Input_dataspace=FFN_dataspace['FFN2'][batch],
                            Output_dataspace=Projection_dataspace[i], transformer=True
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
                start_time = max(start_time, end_time['K']) if batch == 0 else max(start_time, end_time['K'], end_time['A'])
                for i in Mh_attention:
                    if i == 'A':
                        _, cur_time_scale, _, last_time, Attention_dataspace[i] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index(i)], workload[self.layers.index(i)], start_time=start_time
                        )  
                        cal_time_list.append((start_time, last_time))
                        bubble_list.append({})
                        current_batch_cal.append((start_time, last_time))
                        current_batch_bubble.append({})
                        current_batch_weight.append([])
                        end_time[i] = last_time
                    else:
                        if batch != 0: start_time = end_time['Z0']
                        _, cur_time_scale, _, last_time, Attention_dataspace[i] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index(i)], workload[self.layers.index(i)], start_time=start_time
                        )  
                    time_scale.append(cur_time_scale)

                actual_time, _, _, _, _, bubble, cal_time, _, Attention_dataspace['Z0'] = self.pipeline_analyzer.pipeline_analysis(
                    time_scale[4], 0, 0, Input_dataspace=Attention_dataspace['A'], Weight_dataspace=Projection_dataspace['V'],
                    Output_dataspace=Attention_dataspace['Z0'], transformer=True, attention=True, transpose=True
                )  
                cal_time_list.append(cal_time)
                bubble_list.append(bubble)
                current_batch_cal.append(cal_time)
                current_batch_bubble.append(bubble)
                current_batch_weight.append([])
                end_time['Z0'] = actual_time

                # --- FFN Phase ---
                if batch != 0: start_time = max(start_time, end_time['Z1'])
                for i in FNN: 
                    _, cur_time_scale, _, last_time, FFN_dataspace[i][batch] = self.pipeline_analyzer.parse_dataspace(
                        output_path_pipeline[self.layers.index(i)], workload[self.layers.index(i)], start_time=start_time
                    )  
                    time_scale.append(cur_time_scale)
                    
                actual_time, _, _, _, _, bubble, cal_time, _, FFN_dataspace['Z1'][batch] = self.pipeline_analyzer.pipeline_analysis(
                    time_scale[5], 0, 0, Input_dataspace=Attention_dataspace['Z0'],
                    Output_dataspace=FFN_dataspace['Z1'][batch], transformer=True, output_projetion=True
                )  
                cal_time_list.append(cal_time)
                bubble_list.append(bubble)
                current_batch_cal.append(cal_time)
                current_batch_bubble.append(bubble)
                current_batch_weight.append([])
                end_time['Z1'] = actual_time

                actual_time, _, _, _, _, bubble, cal_time, _, FFN_dataspace['FFN1'][batch] = self.pipeline_analyzer.pipeline_analysis(
                    time_scale[6], 0, 0, Input_dataspace=FFN_dataspace['Z1'][batch],
                    Output_dataspace=FFN_dataspace['FFN1'][batch], transformer=True
                )  
                cal_time_list.append(cal_time)
                bubble_list.append(bubble)
                current_batch_cal.append(cal_time)
                current_batch_bubble.append(bubble)
                current_batch_weight.append([])
                end_time['FFN1'] = actual_time

                actual_time, _, _, _, _, bubble, cal_time, _, FFN_dataspace['FFN2'][batch] = self.pipeline_analyzer.pipeline_analysis(
                    time_scale[7], 0, 0, Input_dataspace=FFN_dataspace['FFN1'][batch],
                    Output_dataspace=FFN_dataspace['FFN2'][batch], transformer=True
                )  
                cal_time_list.append(cal_time)
                bubble_list.append(bubble) 
                current_batch_cal.append(cal_time)
                current_batch_bubble.append(bubble)
                current_batch_weight.append([])
                end_time['FFN2'] = actual_time

                current_layer_cal.append(current_batch_cal)
                current_layer_bubble.append(current_batch_bubble)
                current_layer_weight.append(current_batch_weight)
                batch_start_time[batch] = start_time
                
            layers_cal.append(current_layer_cal)
            layers_bubble.append(current_layer_bubble)
            layers_weight_update.append(current_layer_weight)

        # 9. Energy & Cycle Calculation
        origin_arch_path = os.path.join(self.paths['arch_root'], "pipeline_origin.yaml")
        self.analyzer.modify_arch_yaml(origin_arch_path, self.hw['tile_num'])

        pipeline_access = [self.analyzer.input_output_gen(path) for path in output_path_pipeline]
        inputs = [d[0]['inputs'] for d in pipeline_access]
        outputs = [d[0]['outputs'] for d in pipeline_access]

        no_pipeline_access = [self.analyzer.input_output_gen(path) for path in output_path_no_pipeline]
        inputs_no_pipeline = [d[0]['inputs'] for d in no_pipeline_access]
        outputs_no_pipeline = [d[0]['outputs'] for d in no_pipeline_access]
        weights_no_pipeline = [d[0]['weights'] for d in no_pipeline_access]

        energy_pipeline = []
        energy_origin = []
        write_weight_energy_no_pipeline = []
        
        for i in range(len(self.layers)):
            cim_utilized = self.analyzer.extract_cim_utilized_instances(output_path_pipeline[i])
            cim_write = self.analyzer.extract_cim_write_energy(output_path_pipeline[i])
            vec_access = self.analyzer.extract_vector_access_by_module(output_path_pipeline[i], 'random_fill', 'cim_unit')
            
            e_pipe = self.analyzer.get_total_energy(output_path_pipeline[i] + "/timeloop-mapper.stats.txt") - (cim_utilized * cim_write * vec_access / 1e6)
            energy_pipeline.append(e_pipe)

            cim_utilized_no = self.analyzer.extract_cim_utilized_instances(output_path_no_pipeline[i])
            cim_write_no = self.analyzer.extract_cim_write_energy(output_path_no_pipeline[i])
            vec_access_no = self.analyzer.extract_vector_access_by_module(output_path_no_pipeline[i], 'random_fill', 'cim_unit')
            
            e_origin = self.analyzer.get_total_energy(output_path_no_pipeline[i] + "/timeloop-mapper.stats.txt") - (cim_utilized_no * cim_write_no * vec_access_no / 1e6)
            energy_origin.append(e_origin)
            
            write_weight_energy_no_pipeline.append(cim_utilized_no * cim_write_no * vec_access_no / 1e6)

        # Scale by head_num for attention layers
        for i in Mh_attention:
            idx = self.layers.index(i)
            inputs_no_pipeline[idx] *= head_num
            outputs_no_pipeline[idx] *= head_num
            weights[idx] *= head_num
            energy_pipeline[idx] *= head_num
            energy_origin[idx] *= head_num
            write_weight_energy_no_pipeline[idx] *= head_num

        cycle_no_pipeline = 0
        weight_update = 0
        for i, path in enumerate(output_path_no_pipeline):
            if i in [self.layers.index('A'), self.layers.index('Z0')]:
                cycle_no_pipeline += (self.analyzer.get_cycle(path)) * head_num + (weights[i]) * head_num / 256 / 8 / 16
                weight_update += (weights[i]) * head_num / 256 / 8 / 16
            else: 
                cycle_no_pipeline += self.analyzer.get_cycle(path) + (weights[i]) / 256 / 8 / 16
                weight_update += (weights[i]) / 256 / 8 / 16

        cycle_pipeline = actual_time
        energy_const = 112.54 / 8 / 1e6
        
        idx_Q = self.layers.index('Q')
        idx_A = self.layers.index('A')
        idx_FFN2 = self.layers.index('FFN2')
        
        total_energy_pipeline = sum(energy_pipeline) + (inputs[idx_Q] + 2 * inputs[idx_A] + 2 * outputs[idx_FFN2]) * energy_const
        total_energy_pipeline_origin = sum(energy_origin) + sum(write_weight_energy_no_pipeline) + (sum(inputs_no_pipeline) + 2 * sum(outputs_no_pipeline) + sum(weights)) * energy_const

        # 10. Summary Logging
        logging.info("================ Evaluation Summary ================")
        logging.info(f"Cycles (No Pipeline): {cycle_no_pipeline * block:.2f} | Weight Update: {weight_update * block:.2f}")
        logging.info(f"Cycles (With Pipeline): {cycle_pipeline:.2f}")
        logging.info(f"Energy (No Pipeline): {total_energy_pipeline_origin * block:.4f} pJ")
        logging.info(f"Energy (With Pipeline): {total_energy_pipeline * block:.4f} pJ")
        logging.info("---------------- Detail ----------------")
        logging.info(f"Pipeline Compute Energy: {sum(energy_pipeline) * block:.4f}")
        logging.info(f"Pipeline Feature Update Energy: {block * (inputs[idx_Q] + 2 * inputs[idx_A] + 2 * outputs[idx_FFN2]) * energy_const:.4f}")
        logging.info(f"No-Pipeline Compute Energy: {sum(energy_origin) * block:.4f}")
        logging.info(f"No-Pipeline Weight Update Energy: {sum(weights) * energy_const * block + sum(write_weight_energy_no_pipeline) * block:.4f}")
        logging.info(f"No-Pipeline Feature Update Energy: {(sum(inputs_no_pipeline) * batch_size + 2 * sum(outputs_no_pipeline) * batch_size) * energy_const * block:.4f}")
        
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

        batch_size = batch_size or self.model.get('batch_size', 1)
        head_num = self.model.get('head_num', 1)
        block = self.model.get('block', 1)
        output_path_pipeline, output_path_no_pipeline = self._generate_output_paths()

        workload = []
        compute = []
        weight_access = []

        # 1. Extract workload constraints
        for i, layer in enumerate(self.layers):
            wl = self.analyzer.get_workload(layer)
            workload.append(wl)
            workload_size = wl['C'] * wl['M'] * self.hw['precision']
            compute.append(wl.get('C', 1) * wl.get('M', 1) * wl.get('R', 1) * wl.get('S', 1) *
                           wl.get('P', 1) * wl.get('Q', 1))
            weight_access.append(workload_size)

        min_tile_allocation = []
        for i, layer in enumerate(self.layers):
            min_tile_allocation.append(function.tile_allocation(
                workload[i], macro_num=self.hw['macro_num'], core_num=self.hw['core_num'],
                array_col=self.hw['array_col'], array_row=self.hw['array_row'], 
                cim_depth=self.hw['cim_depth'], precision=self.hw['precision']
            ))

        logging.info(f"Minimum tile allocation per layer: {min_tile_allocation}")

        # Compute max blocks dynamically based on available tiles
        max_block_tiles = sum(min_tile_allocation) + (head_num - 1) * (min_tile_allocation[self.layers.index('A')] + min_tile_allocation[self.layers.index('Z0')])
        max_block = math.floor(self.hw['tile_num'] / max_block_tiles) if max_block_tiles > 0 else 1

        allocation_dict = function.greedy_tile_allocation(
            self.layers, weight_access, min_tile_allocation,
            self.hw['tile_num'], head_num, transformer=True
        )
        allocation = list(allocation_dict.values())
        logging.info(f"Greedy allocation: {allocation} | Max parallel blocks: {max_block}")

        # 2. Setup Bounds for Bayesian Optimization
        bound = []
        Projection = ['Q', 'K', 'V']
        Mh_attention = ['A', 'Z0']
        FNN = ['Z1', 'FFN1', 'FFN2']
        i = 0
        alpha = 0.2

        while i < len(self.layers):
            lb = round(allocation[i] * 0.5) if round(allocation[i] * 0.5) != 0 else 1
            bound.append((max(min_tile_allocation[i], lb), min(allocation[i] + self.hw['tile_num'] * alpha, allocation[i] * 6)))
            if self.layers[i] in FNN[1:]:
                i += 2
            elif self.layers[i] in Projection:
                i += 3
            else:
                i += 1

        space_dimensions = [int(ub) - int(lb) for lb, ub in bound]
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
            n_calls=100,
            dnn_name=self.dnn,
            alpha=0.35,
            
            # --- 传入原版的约束控制参数 ---
            tile_num=self.hw.get('tile_num', 1344),
            layers=self.layers,
            head_num=self.model.get('head_num', 1),
            block_num=self.model.get('block', 1),
            transformer=True,         # 标记为 Transformer 约束
            multi_layer=True,         # 标记为 Multi-layer 约束
            batch=batch,              # 标记为多 Batch 约束
            max_block=getattr(self, 'max_block', 1) 
        )
        
        result, step = optimizer.run_optimization()
        x0, x1, x3, x6, x7 = result.x_opt.astype(int)
        
        # Override with fallback mapping directly using greedy allocation array
        #x0, x1, x3, x6, x7 = allocation[0], allocation[1], allocation[3], allocation[6], allocation[7]
        tile_allocation = [
            int(math.floor(x0/2)), x1, x1, x3, x3, x3, int(math.floor(x6/2)), x7
        ]
        
        # Override to strict mapping logic per user code
        tile_allocation = allocation

        end_time = time.time()
        logging.info(f"BO Optimization Cost: {int((end_time - real_start_time) // 60)} min {(end_time - real_start_time) % 60:.2f} sec")

        # 4. Parallel Simulation Execution
        executor = ParallelExecutor(
            layer_num=len(self.layers), arch_path=self.paths['arch_root'],
            tile_num=tile_allocation, layers=self.layers, DNN=self.dnn, MACRO_NAME='isaac_isca_2016'
        )
        executor.run_parallel()

        pipeline_access = [self.analyzer.input_output_gen(path) for path in output_path_pipeline]
        inputs = [d[0]['inputs'] for d in pipeline_access]
        outputs = [d[0]['outputs'] for d in pipeline_access]
        weights = [d[0]['weights'] for d in pipeline_access]

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
                    current_batch_cal, current_batch_bubble, current_batch_weight = [], [], []
                    Projection_dataspace = {key: [] for key in Projection}
                    time_scale = []
                    
                    if batch == 0:
                        # 1.1 Q Weight update
                        weight_update_start = max(weight_update_start, end_time['Q'], end_time['K'], end_time['V'])
                        weight_update_cost = weight_access[self.layers.index('Q')] / width
                        pipeline_weight_update_cost += weight_update_cost
                        QKV_weight_update_end = weight_update_start + 3 * weight_update_cost
                        current_batch_weight.append((weight_update_start, weight_update_cost))
                        weight_update_start += weight_update_cost
                        
                        Q_start_time = 3 * weight_update_cost if layer_idx == 0 else max(weight_update_start, end_time['A'], batch_end_time[batch], QKV_weight_update_end)
                        _, cur_ts, _, last_t, Projection_dataspace['Q'] = self.pipeline_analyzer.parse_dataspace(output_path_pipeline[self.layers.index('Q')], workload[self.layers.index('Q')], start_time=Q_start_time)
                        current_batch_cal.append((Q_start_time, last_t))
                        current_batch_bubble.append({})
                        time_scale.append(cur_ts)
                        end_time['Q'] = last_t

                        # 1.2 K Weight update
                        weight_update_cost = weight_access[self.layers.index('K')] / width
                        current_batch_weight.append((weight_update_start, weight_update_cost))
                        weight_update_start += weight_update_cost
                        pipeline_weight_update_cost += weight_update_cost
                        
                        K_start_time = start_time if layer_idx == 0 else max(weight_update_start, end_time['A'], batch_end_time[batch], QKV_weight_update_end)
                        _, cur_ts, _, last_t, Projection_dataspace['K'] = self.pipeline_analyzer.parse_dataspace(output_path_pipeline[self.layers.index('Q')], workload[self.layers.index('Q')], start_time=K_start_time)
                        current_batch_cal.append((K_start_time, last_t))
                        current_batch_bubble.append({})
                        time_scale.append(cur_ts)
                        end_time['K'] = last_t

                        # 1.3 V Weight update
                        weight_update_cost = weight_access[self.layers.index('V')] / width
                        current_batch_weight.append((weight_update_start, weight_update_cost))
                        weight_update_start += weight_update_cost
                        pipeline_weight_update_cost += weight_update_cost
                        
                        V_start_time = start_time if layer_idx == 0 else max(weight_update_start, end_time['Z0'], batch_end_time[batch], QKV_weight_update_end)
                        _, cur_ts, _, last_t, Projection_dataspace['V'] = self.pipeline_analyzer.parse_dataspace(output_path_pipeline[self.layers.index('Q')], workload[self.layers.index('Q')], start_time=V_start_time)
                        current_batch_cal.append((V_start_time, last_t))
                        current_batch_bubble.append({})
                        time_scale.append(cur_ts)
                        end_time['V'] = last_t

                        start_time = max(batch_end_time[batch], last_t)
                        batch_end_time[batch] = last_t
                    else:
                        for i in Projection:
                            _, cur_ts, _, last_t, Projection_dataspace[i] = self.pipeline_analyzer.parse_dataspace(output_path_pipeline[self.layers.index(i)], workload[self.layers.index(i)], start_time=end_time[i])
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
                    current_batch_cal, current_batch_bubble, current_batch_weight = [], [], []
                    time_scale = []
                    
                    if batch == 0:
                        Attention_dataspace = {key: [] for key in Mh_attention}
                        FFN_dataspace = {key: [] for key in FNN}
                        for i in Mh_attention: current_batch_weight.append([])

                        if layer_idx != 0: weight_update_start = max(weight_update_start, end_time['Z1'])
                        start_time = max(batch_end_time[batch], end_time['A'], end_time['V'])

                        for i in Mh_attention:
                            _, cur_ts, _, last_t, Attention_dataspace[i] = self.pipeline_analyzer.parse_dataspace(output_path_pipeline[self.layers.index(i)], workload[self.layers.index(i)], start_time=start_time)
                            if i == 'A':
                                current_batch_cal.append((start_time, last_t))
                                end_time['A'] = last_t
                                start_time_dict['A'] = start_time
                                current_batch_bubble.append({})
                            time_scale.append(cur_ts)

                        actual_time, _, _, _, _, bubble, cal_time, next_st, Attention_dataspace['Z0'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[1], 0, 0, Input_dataspace=Attention_dataspace['A'], Output_dataspace=Attention_dataspace['Z0'], transformer=True)
                        end_time['Z0'] = actual_time
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)
                        start_time = next_st

                        # Z1 FFN setup
                        weight_update_cost = weight_access[self.layers.index('Z1')] / width
                        current_batch_weight.append((weight_update_start, weight_update_cost))
                        weight_update_start += weight_update_cost   
                        pipeline_weight_update_cost += weight_update_cost
                        start_time = max(weight_update_start, start_time)  
                        
                        _, cur_ts, _, _, FFN_dataspace['Z1'] = self.pipeline_analyzer.parse_dataspace(output_path_pipeline[self.layers.index('Z1')], workload[self.layers.index('Z1')], start_time=start_time)
                        time_scale.append(cur_ts)
                        actual_time, _, _, _, _, bubble, cal_time, _, FFN_dataspace['Z1'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[2], 0, 0, Input_dataspace=Attention_dataspace['Z0'], Output_dataspace=FFN_dataspace['Z1'], transformer=True, output_projetion=True)
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)
                        end_time['Z1'] = actual_time

                        # FFN1
                        weight_update_cost = weight_access[self.layers.index('FFN1')] / width
                        weight_update_start = max(weight_update_start, end_time['FFN1'])
                        current_batch_weight.append((weight_update_start, weight_update_cost))
                        weight_update_start += weight_update_cost   
                        pipeline_weight_update_cost += weight_update_cost
                        start_time = max(weight_update_start, start_time) 
                        
                        if start_time > end_time['Z1']:
                            total_energy_pipeline += (2 * inputs[self.layers.index('FFN1')]) * 112.54 / 8 / 1e6
                            pipeline_feature += (2 * inputs[self.layers.index('FFN1')]) * 112.54 / 8 / 1e6
                            
                        _, cur_ts, _, _, FFN_dataspace['FFN1'] = self.pipeline_analyzer.parse_dataspace(output_path_pipeline[self.layers.index('FFN1')], workload[self.layers.index('FFN1')], start_time=start_time)
                        time_scale.append(cur_ts) 
                        actual_time, _, _, _, _, bubble, cal_time, _, FFN_dataspace['FFN1'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[3], 0, 0, Input_dataspace=FFN_dataspace['Z1'], Output_dataspace=FFN_dataspace['FFN1'], transformer=True)
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)
                        end_time['FFN1'] = actual_time

                        # FFN2
                        weight_update_cost = weight_access[self.layers.index('FFN2')] / width
                        weight_update_start = max(weight_update_start, end_time['FFN2'])
                        current_batch_weight.append((weight_update_start, weight_update_cost))
                        weight_update_start += weight_update_cost   
                        pipeline_weight_update_cost += weight_update_cost
                        start_time = max(weight_update_start, start_time)
                        
                        if start_time > end_time['FFN1']:
                            total_energy_pipeline += (2 * inputs[self.layers.index('FFN2')]) * 112.54 / 8 / 1e6
                            pipeline_feature += (2 * inputs[self.layers.index('FFN2')]) * 112.54 / 8 / 1e6
                            
                        _, cur_ts, _, _, FFN_dataspace['FFN2'] = self.pipeline_analyzer.parse_dataspace(output_path_pipeline[self.layers.index('FFN2')], workload[self.layers.index('FFN2')], start_time=start_time)
                        time_scale.append(cur_ts) 
                        actual_time, _, _, _, _, bubble, cal_time, next_st, FFN_dataspace['FFN2'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[4], 0, 0, Input_dataspace=FFN_dataspace['FFN1'], Output_dataspace=FFN_dataspace['FFN2'], transformer=True)
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble) 
                        end_time['FFN2'] = actual_time
                        batch_end_time[batch] = actual_time

                        # Follow-up Q, K, V Projections for batch 0
                        for proj_key in ['Q', 'K', 'V']:
                            weight_update_start = max(weight_update_start, end_time['Q'] if proj_key == 'Q' else start_time, end_time['K'] if proj_key == 'Q' else 0, end_time['V'] if proj_key == 'Q' else 0)
                            weight_update_cost = weight_access[self.layers.index(proj_key)] / width
                            pipeline_weight_update_cost += weight_update_cost
                            current_batch_weight.append((weight_update_start, weight_update_cost))
                            weight_update_start += weight_update_cost
                            
                            start_time = max(weight_update_start, start_time, end_time[proj_key])
                            _, cur_ts, _, _, Projection_dataspace[proj_key] = self.pipeline_analyzer.parse_dataspace(output_path_pipeline[self.layers.index(proj_key)], workload[self.layers.index(proj_key)], start_time=start_time)
                            time_scale.append(cur_ts)
                            actual_time, _, _, _, _, bubble, cal_time, _, Projection_dataspace[proj_key] = self.pipeline_analyzer.pipeline_analysis(
                                time_scale[-1], 0, 0, Input_dataspace=FFN_dataspace['FFN2'], Output_dataspace=Projection_dataspace[proj_key], transformer=True)
                            
                            current_batch_cal.append(cal_time)
                            current_batch_bubble.append(bubble)
                            end_time[proj_key] = actual_time
                            if proj_key == 'V': batch_end_time[batch] = actual_time

                    else:
                        # Batch != 0 logic block
                        Attention_dataspace = {key: [] for key in Mh_attention}
                        FFN_dataspace = {key: [] for key in FNN}
                        start_time = end_time['A']

                        for i in Mh_attention:
                            if i == 'A':
                                _, cur_ts, _, last_t, Attention_dataspace[i] = self.pipeline_analyzer.parse_dataspace(output_path_pipeline[self.layers.index(i)], workload[self.layers.index(i)], start_time=start_time)
                                current_batch_cal.append((start_time, last_t))
                                end_time['A'] = last_t
                                start_time_dict['A'] = start_time
                                current_batch_bubble.append({})
                            else:
                                start_time = end_time['Z0']
                                _, cur_ts, _, last_t, Attention_dataspace[i] = self.pipeline_analyzer.parse_dataspace(output_path_pipeline[self.layers.index(i)], workload[self.layers.index(i)], start_time=start_time)
                            time_scale.append(cur_ts)
                        
                        actual_time, _, _, _, _, bubble, cal_time, next_st, Attention_dataspace['Z0'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[1], 0, 0, Input_dataspace=Attention_dataspace['A'], Output_dataspace=Attention_dataspace['Z0'], transformer=True)
                        end_time['Z0'] = actual_time
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)
                        start_time = next_st

                        # Process FFN pipeline sequence
                        for f_key, input_key in [('Z1', 'Z0'), ('FFN1', 'Z1'), ('FFN2', 'FFN1')]:
                            start_time = end_time[f_key]
                            _, cur_ts, _, _, FFN_dataspace[f_key] = self.pipeline_analyzer.parse_dataspace(output_path_pipeline[self.layers.index(f_key)], workload[self.layers.index(f_key)], start_time=start_time)
                            time_scale.append(cur_ts)
                            
                            actual_time, _, _, _, _, bubble, cal_time, next_st, FFN_dataspace[f_key] = self.pipeline_analyzer.pipeline_analysis(
                                time_scale[-1], 0, 0, Input_dataspace=Attention_dataspace[input_key] if input_key == 'Z0' else FFN_dataspace[input_key], 
                                Output_dataspace=FFN_dataspace[f_key], transformer=True, output_projetion=(f_key=='Z1'))
                            
                            if f_key in ['FFN1', 'FFN2'] and start_time > end_time[input_key]:
                                total_energy_pipeline += (2 * inputs[self.layers.index(f_key)]) * 112.54 / 8 / 1e6
                                pipeline_feature += (2 * inputs[self.layers.index(f_key)]) * 112.54 / 8 / 1e6
                                
                            end_time[f_key] = actual_time
                            current_batch_cal.append(cal_time)
                            current_batch_bubble.append(bubble)
                            
                        batch_end_time[batch] = actual_time
                        
                        # Process QKV mapping for subsequent batches
                        for proj_key in ['Q', 'K', 'V']:
                            current_batch_weight.append([])
                            start_time = max(start_time, end_time[proj_key])
                            _, cur_ts, _, _, Projection_dataspace[proj_key] = self.pipeline_analyzer.parse_dataspace(output_path_pipeline[self.layers.index(proj_key)], workload[self.layers.index(proj_key)], start_time=start_time)
                            time_scale.append(cur_ts)
                            actual_time, _, _, _, _, bubble, cal_time, _, Projection_dataspace[proj_key] = self.pipeline_analyzer.pipeline_analysis(
                                time_scale[-1], 0, 0, Input_dataspace=FFN_dataspace['FFN2'], Output_dataspace=Projection_dataspace[proj_key], transformer=True)
                            
                            current_batch_cal.append(cal_time)
                            current_batch_bubble.append(bubble)
                            end_time[proj_key] = actual_time
                            if proj_key == 'V': batch_end_time[batch] = actual_time

                    current_layer_cal.append(current_batch_cal)
                    current_layer_bubble.append(current_batch_bubble)
                    current_layer_weight.append(current_batch_weight)

                # Append boundary logic block (idx == self.block - 1 replica check executed via inner state updates natively)
                layers_cal.append(current_layer_cal)
                layers_bubble.append(current_layer_bubble)
                layers_weight_update.append(current_layer_weight)
                
            # Aggregate total feature energy
            idx_Q = self.layers.index('Q')
            total_energy_pipeline += (inputs[idx_Q] * batch_size + outputs[idx_Q] * batch_size) * 112.54 / 8 / 1e6
            pipeline_feature += (inputs[idx_Q] * batch_size + outputs[idx_Q] * batch_size) * 112.54 / 8 / 1e6

        # 7. Visualization & Stats Plotting
        plot_path = os.path.join(self.paths['script_root'], f"ISAAC-{self.dnn}")
        os.makedirs(plot_path, exist_ok=True)
        plot.plot_combined_timelines_block_batch(plot_path, layers_cal, layers_bubble, layers_weight_update, actual_time=actual_time)

        cal_time, weight_time, overlap_time = function.compute_time_statistics(layers_cal, layers_weight_update)
        logging.info(f"Cal-only Time: {cal_time} | Weight-only Time: {weight_time} | Overlap Time: {overlap_time}")

        # 8. Un-pipelined baseline calculation
        no_pipeline_access = [self.analyzer.input_output_gen(path) for path in output_path_no_pipeline]
        inputs_no_pipeline = [d[0]['inputs'] for d in no_pipeline_access]
        outputs_no_pipeline = [d[0]['outputs'] for d in no_pipeline_access]
        weights_no_pipeline = [d[0]['weights'] for d in no_pipeline_access]

        write_weight_energy_pipeline = []
        write_weight_energy_no_pipeline = []
        energy_origin = []
        energy_pipeline = []
        
        for i in range(len(self.layers)):
            cim_util_pipe = self.analyzer.extract_cim_utilized_instances(output_path_pipeline[i])
            cim_write_pipe = self.analyzer.extract_cim_write_energy(output_path_pipeline[i]) / 1e6
            energy_pipeline.append(self.analyzer.get_total_energy(output_path_pipeline[i]+"/timeloop-mapper.stats.txt") - (cim_util_pipe * cim_write_pipe))
            write_weight_energy_pipeline.append(cim_util_pipe * cim_write_pipe)

            cim_util_no = self.analyzer.extract_cim_utilized_instances(output_path_no_pipeline[i])
            cim_write_no = self.analyzer.extract_cim_write_energy(output_path_no_pipeline[i]) / 1e6
            energy_origin.append(self.analyzer.get_total_energy(output_path_no_pipeline[i]+"/timeloop-mapper.stats.txt") - (cim_util_no * cim_write_no))
            write_weight_energy_no_pipeline.append(cim_util_no * cim_write_no)

        for i in Mh_attention:
            idx_i = self.layers.index(i)
            inputs_no_pipeline[idx_i] *= head_num
            outputs_no_pipeline[idx_i] *= head_num
            weights_no_pipeline[idx_i] *= head_num * batch_size

            energy_pipeline[idx_i] *= head_num
            energy_origin[idx_i] *= head_num

            write_weight_energy_pipeline[idx_i] *= head_num
            write_weight_energy_no_pipeline[idx_i] *= head_num

        cycle_no_pipeline = 0
        weight_update_no_pipeline = []
        for i, path in enumerate(output_path_no_pipeline):
            weight_val = weights_no_pipeline[i] / width
            weight_update_no_pipeline.append(weight_val)
            if i in [self.layers.index('A'), self.layers.index('Z0')]:
                cycle_no_pipeline += (self.analyzer.get_cycle(path)) * head_num * batch_size + weight_val
            else: 
                cycle_no_pipeline += self.analyzer.get_cycle(path) * batch_size + weight_val
            
        cycle_pipeline = actual_time
        weight_access_energy = pipeline_weight_update_cost * width * 112.54 / 8 / 1e6

        total_energy_pipeline += (sum(energy_pipeline) * batch_size + sum(write_weight_energy_pipeline)) * block + weight_access_energy
        total_energy_pipeline_origin = sum(energy_origin) * batch_size + sum(write_weight_energy_no_pipeline) + ((sum(inputs_no_pipeline) + 2 * sum(outputs_no_pipeline)) * batch_size + sum(weights_no_pipeline)) * 112.54 / 8 / 1e6

        logging.info("================ Final Performance Summary ================")
        logging.info(f"Tile allocation = {tile_allocation}")
        logging.info(f"Cycles without pipeline = {cycle_no_pipeline * block:.2f} | Cycles with pipeline = {cycle_pipeline:.2f}")
        logging.info(f"Weight update cycles (No Pipeline) = {sum(weight_update_no_pipeline) * block:.2f}")
        logging.info(f"Weight update cycles (Pipeline) = {pipeline_weight_update_cost:.2f}")
        logging.info(f"Energy (No Pipeline) = {total_energy_pipeline_origin * block:.4f} pJ | Energy (Pipeline) = {total_energy_pipeline:.4f} pJ")
        logging.info("---------------- Detail ----------------")
        logging.info(f"Pipeline Compute Energy = {sum(energy_pipeline) * batch_size * block:.4f}")
        logging.info(f"Pipeline Feature Update Energy = {pipeline_feature:.4f}")
        logging.info(f"Pipeline Weight Update Energy = {weight_access_energy + sum(write_weight_energy_pipeline) * block:.4f}")
        logging.info(f"NoPipeline Compute Energy = {sum(energy_origin) * batch_size * block:.4f}")
        logging.info(f"NoPipeline Weight Update Energy = {sum(weights_no_pipeline) * 112.54 / 8 / 1e6 * block + sum(write_weight_energy_no_pipeline) * block:.4f}")
        logging.info(f"NoPipeline Feature Update Energy = {(sum(inputs_no_pipeline) * batch_size + 2 * sum(outputs_no_pipeline) * batch_size) * 112.54 / 8 / 1e6 * block:.4f}")
        
        return layers_cal, layers_bubble, layers_weight_update
