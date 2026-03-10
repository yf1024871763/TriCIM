import copy
import math
import logging
from src.ParallelExecutor import ParallelExecutor

class FitnessEvaluator:
    """
    专门用于贝叶斯优化的适应度评估类 (Fitness/Objective Functions)。
    作为引擎 (Engine) 和 优化器 (Optimizer) 之间的桥梁。
    """
    def __init__(self, engine):
        # 传入 engine 实例，直接复用引擎中已经初始化好的硬件参数和分析器
        self.engine = engine
        self.hw = engine.hw
        self.model = engine.model
        self.paths = engine.paths
        self.dnn = engine.dnn
        self.layers = engine.layers
        self.analyzer = engine.analyzer
        self.pipeline_analyzer = engine.pipeline_analyzer

    def get_noc_weight_delay(self, data_size, tiles):
        width = 256 * 4 if self.dnn == 'resnet18' else 256 * 8 * 2
        if tiles <= 1 or data_size == 0:
            return data_size / width
            
        mesh_dim = max(2, math.ceil(math.sqrt(tiles)))
        

        if hasattr(self.engine, 'booksim'):
            noc_result = self.engine.booksim.run_simulation(
                mesh_dim=mesh_dim, traffic_pattern="uniform", injection_rate=0.3 
            )
            base_latency = noc_result.get("base_latency", 0.0)
        else:
            base_latency = 0.0 
            
        serialization_delay = data_size / width
        return serialization_delay + base_latency

    def cnn_fitness_callback(self, individual):
        """
        Evaluate the performance of a given basic CNN pipeline configuration.
        (One search dimension per layer)
        """
        import logging
        from src.ParallelExecutor import ParallelExecutor

        # 1. Decode parameters from BO (Expand into a list of tile numbers per layer)
        if hasattr(individual, 'astype'):
            individual = individual.astype(int).tolist()
        elif hasattr(individual, 'flatten'):
            individual = individual.flatten().astype(int).tolist()
        else:
            individual = [int(x) for x in individual]
            
        #logging.info(f"Evaluating CNN Pipeline Alloc: {individual}")

        # 2. Execute parallel computation (Pass individual directly to tile_num)
        executor = ParallelExecutor(
            layer_num=len(self.layers),
            arch_path=self.paths['arch_root'],
            tile_num=individual,
            layers=self.layers,
            DNN=self.dnn,
            MACRO_NAME='isaac_isca_2016'
        )
        executor.run_parallel()

        # 3. Prepare data flow environment
        workload = [self.analyzer.get_workload(layer) for layer in self.layers]
        output_path_pipeline, _ = self.engine._generate_output_paths()
        
        cal_time_list = []
        bubble_list = []
        time_scale = []
        start_time = 0
        Dataspace = [[] for _ in range(len(self.layers))]
        
        # 4. Parse initial Dataspace for each layer
        for i in range(len(self.layers)):
            _, cur_time_scale, _, last_time, Dataspace[i] = self.pipeline_analyzer.parse_dataspace(
                output_path_pipeline[i], workload[i], start_time=start_time
            ) 
            time_scale.append(cur_time_scale)
            if i == 0:
                cal_time_list.append((start_time, last_time))
                bubble_list.append({})

        # 5. Execute continuous Pipeline dependency analysis
        actual_time = last_time if len(self.layers) == 1 else 0
        for i in range(len(self.layers) - 1):
            logging.debug(f"Processing Pipeline: {self.layers[i]} -> {self.layers[i+1]} | Maxpool: {self.pipeline_analyzer.maxpool[i]} | FC: {self.pipeline_analyzer.fc[i+1]}")
            
            actual_time, _, _, _, _, bubble, cal_time, _, Dataspace[i+1] = self.pipeline_analyzer.pipeline_analysis(
                time_scale[i+1], 
                self.pipeline_analyzer.maxpool[i], 
                self.pipeline_analyzer.fc[i+1],
                Input_dataspace=Dataspace[i],
                Output_dataspace=Dataspace[i+1],
                shortcut=self.pipeline_analyzer.shortcut[i+1]
            )
            cal_time_list.append(cal_time)
            bubble_list.append(bubble)
            
        # 6. Calculate feature and weight energy consumption
        pipeline_access = [self.analyzer.input_output_gen(output_path_pipeline[i]) for i in range(len(self.layers))]
        
        inputs = [d[0]['inputs'] if d and len(d) > 0 else 0 for d in pipeline_access]
        outputs = [2 * d[0]['outputs'] if d and len(d) > 0 else 0 for d in pipeline_access]
        
        energy_pipeline = []
        for i in range(len(self.layers)):
            cim_utilized = self.analyzer.extract_cim_utilized_instances(output_path_pipeline[i])
            cim_write_energy = self.analyzer.extract_cim_write_energy(output_path_pipeline[i])
            vec_access = self.analyzer.extract_vector_access_by_module(output_path_pipeline[i], 'random_fill', 'cim_unit')
            
            base_energy = self.analyzer.get_total_energy(output_path_pipeline[i] + "/timeloop-mapper.stats.txt")
            adjusted_energy = base_energy - (cim_utilized * cim_write_energy * vec_access / 1e6)
            energy_pipeline.append(adjusted_energy)
            
        total_energy_pipeline = sum(energy_pipeline) + (inputs[0] + outputs[-1]) * 112.54 / 8 / 1e6
        
        # 7. Log output results
        punishment = self.hw.get('tile_num', 1344) - sum(individual)
        logging.info("=== Iteration Complete ===")
        logging.info(f"Current Params: {individual}")
        logging.info(f"Current Fitness (Latency): {actual_time}")
        logging.info(f"Total Energy (Evaluated): {total_energy_pipeline:.4f}")

        return actual_time

    def transformer_fitness_callback(self, individual):
        """Evaluate the performance of a given transformer pipeline configuration."""
        import logging
        from src.ParallelExecutor import ParallelExecutor

        # Decode individual parameters
        if hasattr(individual, 'astype'):
            individual = individual.astype(int).tolist()
        elif hasattr(individual, 'flatten'):
            individual = individual.flatten().astype(int).tolist()
        else:
            individual = [int(x) for x in individual]
            
        x0, x1, x3, x6, x7 = individual[:5]

        full_alloc = [
            x0,       # x0
            x1,       # x1
            x1,       # x2 = x1
            x3,       # x3
            x3,       # x4 = x3
            x3,       # x5 = x3
            x6,       # x6
            x7        # x7
        ]
        
        head_num = self.model.get('head_num', 1)
        block_num = self.model.get('block', 1)
        
        # Calculate total tiles for constraint checking
        try:
            a_idx = self.layers.index('A')
            z0_idx = self.layers.index('Z0')
            total_tile = sum(full_alloc) + (head_num - 1) * (full_alloc[a_idx] + full_alloc[z0_idx])
        except ValueError as e:
            logging.warning(f"Layer 'A' or 'Z0' not found in layers list: {e}")
            total_tile = sum(full_alloc)    
        
        # Execute parallel simulation
        executor = ParallelExecutor(
            layer_num=len(self.layers),
            arch_path=self.paths['arch_root'],
            tile_num=full_alloc,
            layers=self.layers,
            DNN=self.dnn,
            MACRO_NAME='isaac_isca_2016'
        )
        executor.run_parallel()
        
        # Initialize pipeline constraints and environment
        workload = [self.analyzer.get_workload(layer) for layer in self.layers]
        Projection = ['Q', 'K', 'V']
        Mh_attention = ['A', 'Z0']
        FNN = ['Z1', 'FFN1', 'FFN2']
        output_path_pipeline, _ = self.engine._generate_output_paths()
        
        start_time = 0
        cal_time_list = []
        bubble_list = []
        A_start_time = 0
        FFN_dataspace = {} 
        actual_time = 0
        
        # Process pipeline across all blocks
        for i in range(block_num):       
            #logging.info(f"Processing Block {i} | Start Time: {start_time}")
            Projection_dataspace = {key: [] for key in Projection}

            # --- Input Projection Phase ---
            if i == 0:
                time_scale = []
                for proj in Projection:
                    idx = self.layers.index(proj)
                    _, cur_ts, _, last_t, Projection_dataspace[proj] = self.pipeline_analyzer.parse_dataspace(
                        output_path_pipeline[idx], workload[idx], start_time=start_time
                    )  
                    cal_time_list.append((start_time, last_t))
                    bubble_list.append({})
                    time_scale.append(cur_ts)
                    A_start_time = max(last_t, A_start_time)
            else:
                time_scale = []
                for proj in Projection:
                    idx = self.layers.index(proj)
                    _, cur_ts, _, last_t, Projection_dataspace[proj] = self.pipeline_analyzer.parse_dataspace(
                        output_path_pipeline[idx], workload[idx], start_time=start_time
                    )  
                    actual_time, _, _, _, _, bubble, cal_time, next_st, Projection_dataspace[proj] = self.pipeline_analyzer.pipeline_analysis(
                        cur_ts, 0, 0, Input_dataspace=FFN_dataspace.get('FFN2', []), 
                        Output_dataspace=Projection_dataspace[proj], transformer=True
                    )  
                    cal_time_list.append(cal_time)
                    bubble_list.append(bubble)
                    time_scale.append(cur_ts)
                    A_start_time = max(A_start_time, actual_time)
                    
            # --- Attention Phase ---
            Attention_dataspace = {key: [[] for _ in range(head_num)] for key in Mh_attention}
            FFN_dataspace = {key: [] for key in FNN}
                
            for attn in Mh_attention:
                idx = self.layers.index(attn)
                for j in range(1):
                    if attn == 'A':
                        _, cur_ts, _, last_t, Attention_dataspace[attn][j] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[idx], workload[idx], start_time=A_start_time
                        )  
                        cal_time_list.append((A_start_time, last_t))
                        bubble_list.append({})
                    else:
                        _, cur_ts, _, last_t, Attention_dataspace[attn][j] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[idx], workload[idx], start_time=A_start_time
                        )  
                time_scale.append(cur_ts)
                
            for j in range(1):
                actual_time, _, _, _, _, bubble, cal_time, next_st, Attention_dataspace['Z0'][j] = self.pipeline_analyzer.pipeline_analysis(
                    time_scale[4], 0, 0, Input_dataspace=Attention_dataspace['A'][j], Weight_dataspace=Projection_dataspace['V'],
                    Output_dataspace=Attention_dataspace['Z0'][j], transformer=True, attention=True, transpose=True
                )  
                cal_time_list.append(cal_time)
                bubble_list.append(bubble)
                
            start_time = next_st
            
            # --- Feed Forward Network Phase ---
            for ffn in FNN: 
                idx = self.layers.index(ffn)
                _, cur_ts, _, last_t, FFN_dataspace[ffn] = self.pipeline_analyzer.parse_dataspace(
                    output_path_pipeline[idx], workload[idx], start_time=start_time
                )  
                time_scale.append(cur_ts)
                
            actual_time, _, _, _, _, bubble, cal_time, next_st, FFN_dataspace['Z1'] = self.pipeline_analyzer.pipeline_analysis(
                time_scale[5], 0, 0, Input_dataspace=Attention_dataspace['Z0'][0], Output_dataspace=FFN_dataspace['Z1'], transformer=True, output_projetion=True
            )  
            cal_time_list.append(cal_time)
            bubble_list.append(bubble)
            
            actual_time, _, _, _, _, bubble, cal_time, next_st, FFN_dataspace['FFN1'] = self.pipeline_analyzer.pipeline_analysis(
                time_scale[6], 0, 0, Input_dataspace=FFN_dataspace['Z1'], Output_dataspace=FFN_dataspace['FFN1'], transformer=True
            )  
            cal_time_list.append(cal_time)
            bubble_list.append(bubble)
            
            actual_time, _, _, _, _, bubble, cal_time, next_st, FFN_dataspace['FFN2'] = self.pipeline_analyzer.pipeline_analysis(
                time_scale[7], 0, 0, Input_dataspace=FFN_dataspace['FFN1'], Output_dataspace=FFN_dataspace['FFN2'], transformer=True
            )  
            cal_time_list.append(cal_time)
            bubble_list.append(bubble) 
            
            start_time = next_st
            
        punishment = ((self.hw['tile_num'] / block_num) - total_tile) * 2000
        
        logging.info(f"Iteration Completed | Current Params: {full_alloc}")
        logging.info(f"Current Fitness: {actual_time}")
        
        return actual_time
    
    def transformer_multi_batch_fitness_callback(self, individual):
        """Evaluate the performance of a given transformer multi-batch pipeline configuration."""
        import logging
        import math
        from src.ParallelExecutor import ParallelExecutor

        output_path_pipeline, _ = self.engine._generate_output_paths()
        
        workload = []
        for layer in self.layers:
            workload.append(self.analyzer.get_workload(layer))
            
        compute = []
        weight_access = []
        for i, layer in enumerate(self.layers):
            workload_size = workload[i].get('C', 1) * workload[i].get('M', 1) * self.hw.get('precision', 16)
            compute.append(
                workload[i].get('C', 1) * workload[i].get('M', 1) * workload[i].get('R', 1) * workload[i].get('S', 1) *
                workload[i].get('P', 1) * workload[i].get('Q', 1)
            )
            weight_access.append(workload_size)
            
        Projection = ['Q', 'K', 'V']
        Mh_attention = ['A', 'Z0']
        FNN = ['Z1', 'FFN1', 'FFN2']

        if hasattr(individual, 'astype'):
            individual = individual.astype(int).tolist()
        elif hasattr(individual, 'flatten'):
            individual = individual.flatten().astype(int).tolist()
        else:
            individual = [int(x) for x in individual]
            
        x0, x1, x3, x6, x7 = individual[:5]

        full_alloc = [
            int(math.floor(x0)),       # x0
            x1,                        # x1
            x1,                        # x2 = x1
            x3,                        # x3
            x3,                        # x4 = x3
            x3,                        # x5 = x3
            int(math.floor(x6)),       # x6
            x7                         # x7
        ]
        
        full_alloc = [max(1, j) for j in full_alloc]
        
        executor = ParallelExecutor(
            layer_num=len(self.layers),
            arch_path=self.paths['arch_root'],
            tile_num=full_alloc,
            layers=self.layers,
            DNN=self.dnn,
            MACRO_NAME='isaac_isca_2016'
        )
        executor.run_parallel()

        pipeline_access = []
        for i in range(len(self.layers)):
            pipeline_access.append(self.analyzer.input_output_gen(output_path_pipeline[i]))
            
        total_energy_pipeline = 0
        inputs = [d[0]['inputs'] for d in pipeline_access]
        outputs = [d[0]['outputs'] for d in pipeline_access]

        start_time = 0
        cal_time_list = []
        bubble_list = []
        time_scale = []
        A_start_time = 0
        
        width = 256 * 2 * 8 * 8 if self.dnn == 'gpt3' else 256 * 2 * 8
        
        Projection_dataspace = {key: [] for key in Projection}
        end_time = {key: 0 for key in self.layers}
        start_time_dict = {key: 0 for key in self.layers}
        weight_update_start = 0
        pipeline_weight_update_cost = 0

        layers_cal = []
        layers_bubble = []
        layers_weight_update = []
        
        batch_size = self.model.get('batch_size', 1)
        block_num = self.model.get('block', 1)
        batch_end_time = [0 for _ in range(batch_size)]

        actual_time = 0

        for idx in range(int(block_num)): 
            layer_idx = idx
            current_layer_cal = []
            current_layer_bubble = []
            current_layer_weight = []
            
            if idx == 0:
                for batch in range(int(batch_size)):
                    current_batch_cal = []   
                    current_batch_bubble = []
                    current_batch_weight = []
                    Projection_dataspace = {key: [] for key in Projection}
                    time_scale = []
                    
                    if batch == 0:
                        # Q
                        weight_update_start = max(weight_update_start, end_time['Q'], end_time['K'], end_time['V'])
                        weight_update_cost = weight_access[self.layers.index('Q')] / width
                        pipeline_weight_update_cost += weight_update_cost
                        QKV_weight_update_end = weight_update_start + 3 * weight_update_cost
                        current_batch_weight.append((weight_update_start, weight_update_cost))  
                        weight_update_start += weight_update_cost
                        
                        if layer_idx == 0:
                            start_time = 3 * weight_update_cost
                            Q_start_time = start_time
                        else:
                            Q_start_time = max(weight_update_start, end_time['A'], batch_end_time[batch], QKV_weight_update_end)
                            
                        _, cur_time_scale, _, last_time, Projection_dataspace['Q'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('Q')], workload[self.layers.index('Q')], start_time=Q_start_time
                        )  
                        current_batch_cal.append((Q_start_time, last_time))  
                        current_batch_bubble.append({})  
                        time_scale.append(cur_time_scale)
                        end_time['Q'] = last_time

                        # K
                        weight_update_cost = weight_access[self.layers.index('K')] / width
                        current_batch_weight.append((weight_update_start, weight_update_cost))  
                        weight_update_start += weight_update_cost
                        pipeline_weight_update_cost += weight_update_cost
                        
                        if layer_idx == 0:
                            K_start_time = start_time
                        else:
                            K_start_time = max(weight_update_start, end_time['A'], batch_end_time[batch], QKV_weight_update_end)
                            
                        _, cur_time_scale, _, last_time, Projection_dataspace['K'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('Q')], workload[self.layers.index('Q')], start_time=K_start_time
                        )  
                        current_batch_cal.append((K_start_time, last_time))  
                        current_batch_bubble.append({})  
                        time_scale.append(cur_time_scale)
                        end_time['K'] = last_time

                        # V
                        weight_update_cost = weight_access[self.layers.index('V')] / width
                        current_batch_weight.append((weight_update_start, weight_update_cost))  
                        weight_update_start += weight_update_cost
                        pipeline_weight_update_cost += weight_update_cost
                        
                        if layer_idx == 0:
                            V_start_time = start_time
                        else:
                            V_start_time = max(weight_update_start, end_time['Z0'], batch_end_time[batch], QKV_weight_update_end)
                            
                        _, cur_time_scale, _, last_time, Projection_dataspace['V'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('Q')], workload[self.layers.index('Q')], start_time=V_start_time
                        )  
                        current_batch_cal.append((V_start_time, last_time))  
                        current_batch_bubble.append({})  
                        time_scale.append(cur_time_scale)
                        end_time['V'] = last_time

                        start_time = max(batch_end_time[batch], last_time)
                        batch_end_time[batch] = last_time
                        
                    else:
                        for i in Projection:
                            _, cur_time_scale, _, last_time, Projection_dataspace[i] = self.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[self.layers.index(i)], workload[self.layers.index(i)], start_time=end_time[i]
                            )  
                            current_batch_cal.append((end_time[i], last_time))  
                            current_batch_bubble.append({})  
                            time_scale.append(cur_time_scale)
                            end_time[i] = last_time
                            batch_end_time[batch] = last_time
                            
                    current_layer_cal.append(current_batch_cal)       
                    current_layer_bubble.append(current_batch_bubble) 
                    current_layer_weight.append(current_batch_weight) 

            else:
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
                            weight_update_start = max(weight_update_start, end_time['Z1'])
                    
                        start_time = max(batch_end_time[batch], end_time['A'], end_time['V'])

                        for i in Mh_attention:
                            if i == 'A':
                                _, cur_time_scale, _, last_time, Attention_dataspace[i] = self.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[self.layers.index(i)], workload[self.layers.index(i)], start_time=start_time
                                )  
                                current_batch_cal.append((start_time, last_time))  
                                end_time['A'] = last_time
                                start_time_dict['A'] = start_time
                                current_batch_bubble.append({})              
                            else:
                                _, cur_time_scale, _, last_time, Attention_dataspace[i] = self.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[self.layers.index(i)], workload[self.layers.index(i)], start_time=start_time
                                )  
                            time_scale.append(cur_time_scale)
                    
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, Attention_dataspace['Z0'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[1], 0, 0, 
                            Input_dataspace=Attention_dataspace['A'],
                            Output_dataspace=Attention_dataspace['Z0'],
                            transformer=True                        
                        )  
                        end_time['Z0'] = actual_time
                        current_batch_cal.append(cal_time)  
                        current_batch_bubble.append(bubble)  
                        start_time = next_start_time

                        weight_update_cost = weight_access[self.layers.index('Z1')] / width
                        current_batch_weight.append((weight_update_start, weight_update_cost))
                        weight_update_start += weight_update_cost   
                        pipeline_weight_update_cost += weight_update_cost
                        start_time = max(weight_update_start, start_time)  
                        
                        _, cur_time_scale, _, last_time, FFN_dataspace['Z1'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('Z1')], workload[self.layers.index('Z1')], start_time=start_time
                        )  

                        time_scale.append(cur_time_scale) 
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, FFN_dataspace['Z1'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[2], 0, 0, Input_dataspace=Attention_dataspace['Z0'],
                            Output_dataspace=FFN_dataspace['Z1'], transformer=True, output_projetion=True
                        )  
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)
                        end_time['Z1'] = actual_time

                        weight_update_cost = weight_access[self.layers.index('FFN1')] / width
                        weight_update_start = max(weight_update_start, end_time['FFN1'])
                        current_batch_weight.append((weight_update_start, weight_update_cost))
                        weight_update_start += weight_update_cost   
                        pipeline_weight_update_cost += weight_update_cost
                        start_time = max(weight_update_start, start_time) 

                        _, cur_time_scale, _, last_time, FFN_dataspace['FFN1'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('FFN1')], workload[self.layers.index('FFN1')], start_time=start_time
                        )  
                        
                        time_scale.append(cur_time_scale) 
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, FFN_dataspace['FFN1'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[3], 0, 0, Input_dataspace=FFN_dataspace['Z1'],
                            Output_dataspace=FFN_dataspace['FFN1'], transformer=True
                        )  
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)
                        end_time['FFN1'] = actual_time

                        weight_update_cost = weight_access[self.layers.index('FFN2')] / width
                        weight_update_start = max(weight_update_start, end_time['FFN2'])
                        current_batch_weight.append((weight_update_start, weight_update_cost))
                        weight_update_start += weight_update_cost   
                        pipeline_weight_update_cost += weight_update_cost
                        start_time = max(weight_update_start, start_time) 

                        _, cur_time_scale, _, last_time, FFN_dataspace['FFN2'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('FFN2')], workload[self.layers.index('FFN2')], start_time=start_time
                        )  
                        time_scale.append(cur_time_scale) 
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, FFN_dataspace['FFN2'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[4], 0, 0, Input_dataspace=FFN_dataspace['FFN1'],
                            Output_dataspace=FFN_dataspace['FFN2'], transformer=True
                        )  
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble) 
                        end_time['FFN2'] = actual_time

                        batch_end_time[batch] = actual_time

                        weight_update_start = max(weight_update_start, end_time['Q'], end_time['K'], end_time['V'])
                        weight_update_cost = weight_access[self.layers.index('Q')] / width
                        pipeline_weight_update_cost += weight_update_cost
                        current_batch_weight.append((weight_update_start, weight_update_cost))  
                        weight_update_start += weight_update_cost
                        
                        if layer_idx == 0:
                            start_time = 3 * weight_update_cost
                            Q_start_time = start_time
                        else:
                            Q_start_time = max(weight_update_start, start_time, end_time['Q'])
                            
                        _, cur_time_scale, _, last_time, Projection_dataspace['Q'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('Q')], workload[self.layers.index('Q')], start_time=Q_start_time
                        )  

                        time_scale.append(cur_time_scale)
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, Projection_dataspace['Q'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[5], 0, 0, Input_dataspace=FFN_dataspace['FFN2'], Output_dataspace=Projection_dataspace['Q'], transformer=True
                        )  
                        current_batch_cal.append(cal_time)  
                        current_batch_bubble.append(bubble)  
                        end_time['Q'] = actual_time
                        
                        current_batch_weight.append([])  
                        start_time = max(start_time, end_time['K'])
                        _, cur_time_scale, _, last_time, Projection_dataspace['K'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('K')], workload[self.layers.index('K')], start_time=start_time
                        )  

                        time_scale.append(cur_time_scale)
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, Projection_dataspace['K'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[6], 0, 0, Input_dataspace=FFN_dataspace['FFN2'], Output_dataspace=Projection_dataspace['K'], transformer=True
                        )  
                        current_batch_cal.append(cal_time)  
                        current_batch_bubble.append(bubble)  
                        end_time['K'] = actual_time
                        
                        current_batch_weight.append([])  
                        start_time = max(start_time, end_time['V'])
                        _, cur_time_scale, _, last_time, Projection_dataspace['V'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('V')], workload[self.layers.index('V')], start_time=start_time
                        )  

                        time_scale.append(cur_time_scale)
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, Projection_dataspace['V'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[7], 0, 0, Input_dataspace=FFN_dataspace['FFN2'], Output_dataspace=Projection_dataspace['V'], transformer=True
                        )  
                        current_batch_cal.append(cal_time)  
                        current_batch_bubble.append(bubble)  
                        end_time['V'] = actual_time
                        batch_end_time[batch] = actual_time

                    else:
                        Attention_dataspace = {key: [] for key in Mh_attention}
                        FFN_dataspace = {key: [] for key in FNN}
                        start_time = end_time['A']

                        for i in Mh_attention:
                            if i == 'A':
                                _, cur_time_scale, _, last_time, Attention_dataspace[i] = self.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[self.layers.index(i)], workload[self.layers.index(i)], start_time=start_time
                                )  
                                current_batch_cal.append((start_time, last_time))  
                                end_time['A'] = last_time
                                start_time_dict['A'] = start_time
                                current_batch_bubble.append({})              
                            else:
                                start_time = end_time['Z0']
                                _, cur_time_scale, _, last_time, Attention_dataspace[i] = self.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[self.layers.index(i)], workload[self.layers.index(i)], start_time=start_time
                                )  
                            time_scale.append(cur_time_scale)
                        
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, Attention_dataspace['Z0'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[1], 0, 0, 
                            Input_dataspace=Attention_dataspace['A'],
                            Output_dataspace=Attention_dataspace['Z0'],
                            transformer=True
                        )  
                        end_time['Z0'] = actual_time
                        current_batch_cal.append(cal_time)  
                        current_batch_bubble.append(bubble)  
                        start_time = next_start_time

                        start_time = end_time['Z1']
                        _, cur_time_scale, _, last_time, FFN_dataspace['Z1'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('Z1')], workload[self.layers.index('Z1')], start_time=start_time
                        )  
                        time_scale.append(cur_time_scale)
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, FFN_dataspace['Z1'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[2], 0, 0, 
                            Input_dataspace=Attention_dataspace['Z0'],
                            Output_dataspace=FFN_dataspace['Z1'],
                            transformer=True, output_projetion=True
                        )  
                        end_time['Z1'] = actual_time
                        current_batch_cal.append(cal_time)  
                        current_batch_bubble.append(bubble)  

                        start_time = end_time['FFN1']
                        _, cur_time_scale, _, last_time, FFN_dataspace['FFN1'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('FFN1')], workload[self.layers.index('FFN1')], start_time=start_time
                        )  
                        time_scale.append(cur_time_scale)
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, FFN_dataspace['FFN1'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[3], 0, 0, 
                            Input_dataspace=FFN_dataspace['Z1'],
                            Output_dataspace=FFN_dataspace['FFN1'],
                            transformer=True
                        )  
                        end_time['FFN1'] = actual_time
                        current_batch_cal.append(cal_time)  
                        current_batch_bubble.append(bubble)  

                        start_time = end_time['FFN2']
                        _, cur_time_scale, _, last_time, FFN_dataspace['FFN2'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('FFN2')], workload[self.layers.index('FFN2')], start_time=start_time
                        )  
                        time_scale.append(cur_time_scale)
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, FFN_dataspace['FFN2'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[4], 0, 0, 
                            Input_dataspace=FFN_dataspace['FFN1'],
                            Output_dataspace=FFN_dataspace['FFN2'],
                            transformer=True
                        )  
                        end_time['FFN2'] = actual_time
                        current_batch_cal.append(cal_time)  
                        current_batch_bubble.append(bubble)  

                        start_time = max(start_time, end_time['Q'])
                        batch_end_time[batch] = actual_time

                        current_batch_weight.append([])  
                        _, cur_time_scale, _, last_time, Projection_dataspace['Q'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('Q')], workload[self.layers.index('Q')], start_time=start_time
                        )  
                        time_scale.append(cur_time_scale)
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, Projection_dataspace['Q'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[5], 0, 0, Input_dataspace=FFN_dataspace['FFN2'], Output_dataspace=Projection_dataspace['Q'], transformer=True
                        )  
                        current_batch_cal.append(cal_time)  
                        current_batch_bubble.append(bubble)  
                        end_time['Q'] = actual_time
                        
                        current_batch_weight.append([])  
                        start_time = max(start_time, end_time['K'])
                        _, cur_time_scale, _, last_time, Projection_dataspace['K'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('K')], workload[self.layers.index('K')], start_time=start_time
                        )  
                        time_scale.append(cur_time_scale)
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, Projection_dataspace['K'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[6], 0, 0, Input_dataspace=FFN_dataspace['FFN2'], Output_dataspace=Projection_dataspace['K'], transformer=True
                        )  
                        current_batch_cal.append(cal_time)  
                        current_batch_bubble.append(bubble)  
                        end_time['K'] = actual_time
                        
                        current_batch_weight.append([])  
                        start_time = max(start_time, end_time['V'])
                        _, cur_time_scale, _, last_time, Projection_dataspace['V'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('V')], workload[self.layers.index('V')], start_time=start_time
                        )  
                        time_scale.append(cur_time_scale)
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, Projection_dataspace['V'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[7], 0, 0, Input_dataspace=FFN_dataspace['FFN2'], Output_dataspace=Projection_dataspace['V'], transformer=True
                        )  
                        current_batch_cal.append(cal_time)  
                        current_batch_bubble.append(bubble)  
                        end_time['V'] = actual_time
                        batch_end_time[batch] = actual_time
                        
                    current_layer_cal.append(current_batch_cal)       
                    current_layer_bubble.append(current_batch_bubble) 
                    current_layer_weight.append(current_batch_weight) 
                    
            layers_cal.append(current_layer_cal)
            layers_bubble.append(current_layer_bubble)
            layers_weight_update.append(current_layer_weight)
            
            if idx == block_num - 1:
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
                            weight_update_start = max(weight_update_start, end_time['Z1'])
                    
                        start_time = max(batch_end_time[batch], end_time['A'], end_time['V'])
                        for i in Mh_attention:
                            if i == 'A':
                                _, cur_time_scale, _, last_time, Attention_dataspace[i] = self.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[self.layers.index(i)], workload[self.layers.index(i)], start_time=start_time
                                )  
                                current_batch_cal.append((start_time, last_time))  
                                end_time['A'] = last_time
                                start_time_dict['A'] = start_time
                                current_batch_bubble.append({})              
                            else:
                                _, cur_time_scale, _, last_time, Attention_dataspace[i] = self.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[self.layers.index(i)], workload[self.layers.index(i)], start_time=start_time
                                )  
                            time_scale.append(cur_time_scale)
                    
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, Attention_dataspace['Z0'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[1], 0, 0, 
                            Input_dataspace=Attention_dataspace['A'],
                            Output_dataspace=Attention_dataspace['Z0'],
                            transformer=True
                        )  
                        end_time['Z0'] = actual_time
                        current_batch_cal.append(cal_time)  
                        current_batch_bubble.append(bubble)  
                        start_time = next_start_time

                        weight_update_cost = weight_access[self.layers.index('Z1')] / width
                        weight_update_start = max(weight_update_start, end_time['Z1'])
                        current_batch_weight.append((weight_update_start, weight_update_cost))
                        weight_update_start += weight_update_cost   
                        pipeline_weight_update_cost += weight_update_cost
                        start_time = max(weight_update_start, start_time, end_time['Z1'])  
                        
                        _, cur_time_scale, _, last_time, FFN_dataspace['Z1'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('Z1')], workload[self.layers.index('Z1')], start_time=start_time
                        )  
                        time_scale.append(cur_time_scale) 
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, FFN_dataspace['Z1'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[2], 0, 0, Input_dataspace=Attention_dataspace['Z0'],
                            Output_dataspace=FFN_dataspace['Z1'], transformer=True, output_projetion=True
                        )  
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)
                        end_time['Z1'] = actual_time

                        weight_update_cost = weight_access[self.layers.index('FFN1')] / width
                        weight_update_start = max(weight_update_start, end_time['FFN1'])
                        current_batch_weight.append((weight_update_start, weight_update_cost))
                        weight_update_start += weight_update_cost   
                        pipeline_weight_update_cost += weight_update_cost
                        start_time = max(weight_update_start, start_time, end_time['FFN1']) 

                        _, cur_time_scale, _, last_time, FFN_dataspace['FFN1'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('FFN1')], workload[self.layers.index('FFN1')], start_time=start_time
                        )  
                        time_scale.append(cur_time_scale) 
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, FFN_dataspace['FFN1'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[3], 0, 0, Input_dataspace=FFN_dataspace['Z1'],
                            Output_dataspace=FFN_dataspace['FFN1'], transformer=True
                        )  
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble)
                        end_time['FFN1'] = actual_time

                        weight_update_cost = weight_access[self.layers.index('FFN2')] / width
                        current_batch_weight.append((weight_update_start, weight_update_cost))
                        weight_update_start = max(weight_update_start, end_time['FFN2'])
                        weight_update_start += weight_update_cost   
                        pipeline_weight_update_cost += weight_update_cost
                        start_time = max(weight_update_start, start_time, end_time['FFN2']) 

                        _, cur_time_scale, _, last_time, FFN_dataspace['FFN2'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('FFN2')], workload[self.layers.index('FFN2')], start_time=start_time
                        )  
                        time_scale.append(cur_time_scale) 
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, FFN_dataspace['FFN2'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[4], 0, 0, Input_dataspace=FFN_dataspace['FFN1'],
                            Output_dataspace=FFN_dataspace['FFN2'], transformer=True
                        )  
                        current_batch_cal.append(cal_time)
                        current_batch_bubble.append(bubble) 
                        end_time['FFN2'] = actual_time

                        start_time = next_start_time
                        batch_end_time[batch] = actual_time
                    else:
                        Attention_dataspace = {key: [] for key in Mh_attention}
                        FFN_dataspace = {key: [] for key in FNN}
                        start_time = end_time['A']

                        for i in Mh_attention:
                            if i == 'A':
                                _, cur_time_scale, _, last_time, Attention_dataspace[i] = self.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[self.layers.index(i)], workload[self.layers.index(i)], start_time=start_time
                                )  
                                current_batch_cal.append((start_time, last_time))  
                                end_time['A'] = last_time
                                start_time_dict['A'] = start_time
                                current_batch_bubble.append({})              
                            else:
                                start_time = end_time['Z0']
                                _, cur_time_scale, _, last_time, Attention_dataspace[i] = self.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[self.layers.index(i)], workload[self.layers.index(i)], start_time=start_time
                                )  
                            time_scale.append(cur_time_scale)
                        
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, Attention_dataspace['Z0'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[1], 0, 0, 
                            Input_dataspace=Attention_dataspace['A'],
                            Output_dataspace=Attention_dataspace['Z0'],
                            transformer=True
                        )  
                        end_time['Z0'] = actual_time
                        current_batch_cal.append(cal_time)  
                        current_batch_bubble.append(bubble)  
                        start_time = next_start_time

                        start_time = end_time['Z1']
                        _, cur_time_scale, _, last_time, FFN_dataspace['Z1'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('Z1')], workload[self.layers.index('Z1')], start_time=start_time
                        )  
                        time_scale.append(cur_time_scale)
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, FFN_dataspace['Z1'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[2], 0, 0, 
                            Input_dataspace=Attention_dataspace['Z0'],
                            Output_dataspace=FFN_dataspace['Z1'],
                            transformer=True, output_projetion=True
                        )  
                        end_time['Z1'] = actual_time
                        current_batch_cal.append(cal_time)  
                        current_batch_bubble.append(bubble)  

                        start_time = end_time['FFN1']
                        _, cur_time_scale, _, last_time, FFN_dataspace['FFN1'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('FFN1')], workload[self.layers.index('FFN1')], start_time=start_time
                        )  
                        time_scale.append(cur_time_scale)
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, FFN_dataspace['FFN1'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[3], 0, 0, 
                            Input_dataspace=FFN_dataspace['Z1'],
                            Output_dataspace=FFN_dataspace['FFN1'],
                            transformer=True
                        )  
                        end_time['FFN1'] = actual_time
                        current_batch_cal.append(cal_time)  
                        current_batch_bubble.append(bubble)  

                        start_time = end_time['FFN2']
                        _, cur_time_scale, _, last_time, FFN_dataspace['FFN2'] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[self.layers.index('FFN2')], workload[self.layers.index('FFN2')], start_time=start_time
                        )  
                        time_scale.append(cur_time_scale)
                        actual_time, _, _, _, _, bubble, cal_time, next_start_time, FFN_dataspace['FFN2'] = self.pipeline_analyzer.pipeline_analysis(
                            time_scale[4], 0, 0, 
                            Input_dataspace=FFN_dataspace['FFN1'],
                            Output_dataspace=FFN_dataspace['FFN2'],
                            transformer=True
                        )  
                        end_time['FFN2'] = actual_time
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

        logging.info("=== Iteration Complete ===")
        logging.info(f"Current Parameters: {full_alloc}")
        logging.info(f"Current Fitness: {actual_time}")
        
        return actual_time
    def cnn_multi_layer_fitness_callback(self, individual):
        """Evaluate the performance of a given CNN multi-layer pipeline configuration."""
        import logging
        import math
        import copy
        from src.ParallelExecutor import ParallelExecutor

        alloc = self.engine.all_allocations[int(individual[0])]
        workload_group = []
        tile_allocation = []
        
        for group in alloc:
            layers_in_group = group[0]
            alloc_dict = group[1]
            workload_group.append(layers_in_group)
            
            for layer in layers_in_group:
                if alloc_dict[layer] > self.hw['tile_num']:
                    alloc_dict[layer] = self.hw['tile_num']
                tile_allocation.append(alloc_dict[layer])
                
        logging.info(f"Evaluating Index= {individual[0]} | Groups: {workload_group} | Tiles: {tile_allocation}")
        
        weight_access_group = []
        workload_size = []
        workload = []
        
        for layer in self.layers:
            workload.append(self.analyzer.get_workload(layer))
            
        for i, layer in enumerate(self.layers):
            if 'R' in workload[i].keys() or 'S' in workload[i].keys():
                workload_size.append(workload[i]['C'] * workload[i]['M'] * workload[i]['R'] * workload[i]['S'] * self.hw.get('precision', 16))
            else:
                workload_size.append(workload[i]['C'] * workload[i]['M'] * self.hw.get('precision', 16))
        
        for k in workload_group:
            weight_access = 0
            for layer in k:
                weight_access += workload_size[self.layers.index(layer)]
            weight_access_group.append(weight_access)

        executor = ParallelExecutor(
            layer_num=len(tile_allocation),
            arch_path=self.paths['arch_root'],
            tile_num=tile_allocation,
            layers=self.layers,
            DNN=self.dnn,
            MACRO_NAME='isaac_isca_2016'
        )   
        executor.run_parallel()         
        
        start_time = 0
        max_cost_time = 0
        actual_compute_time = []
        factor = []
        
        def get_noc_weight_delay(data_size, tiles):
            width = 256 * 4 if self.dnn == 'resnet18' else 256 * 8 * 2
            if tiles <= 1 or data_size == 0:
                return data_size / width
            
            mesh_dim = max(2, math.ceil(math.sqrt(tiles)))
            
            if hasattr(self.engine, 'booksim'):
                noc_result = self.engine.booksim.run_simulation(
                    mesh_dim=mesh_dim, 
                    traffic_pattern="uniform", 
                    injection_rate=0.3 
                )
                base_latency = noc_result.get("base_latency", 0.0)
            else:
                base_latency = 0.0
                
            serialization_delay = data_size / width
            real_delay = serialization_delay + base_latency
            return real_delay

        output_path_pipeline, _ = self.engine._generate_output_paths()
        
        batch_size = int(self.model.get('batch_size', 1))
        end_time_dict = {key: 0 for key in self.layers}
        start_time_dict = [{key: 0 for key in self.layers} for _ in range(batch_size)]  
        Dataspace = [{key: [] for key in self.layers} for _ in range(batch_size)]

        current_layer_cal = [[] for _ in range(len(self.layers))]
        current_layer_bubble = [[] for _ in range(len(self.layers))]
        current_layer_weight = [[] for _ in range(len(self.layers))]
        
        weight_update_start = 0
        available_tile = self.hw['tile_num']
        last_time_dict = {key: 0 for key in self.layers}

        for k, layers in enumerate(workload_group):  
            for batch in range(batch_size):
                current_batch_cal = []   
                current_batch_bubble = []
                current_batch_weight = []  
                time_scale = []

                if batch == 0: 
                    if len(layers) == 1:
                        last_group = workload_group[k-1] if k > 0 else []
                        if last_group:
                            weight_update_start = end_time_dict[last_group[-1]]
                        
                        idx = self.layers.index(layers[0])
                        group_tiles = tile_allocation[idx]
                        real_delay = get_noc_weight_delay(weight_access_group[k], group_tiles)
                        
                        current_batch_weight.append([(weight_update_start, real_delay)])
                        weight_update_start += real_delay
                        
                        start_time = max(weight_update_start, end_time_dict.get(self.layers[idx-1], 0) if idx > 0 else 0)
                        _, cur_time_scale, _, last_time, Dataspace[batch][layers[0]] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[idx], workload[idx], self.pipeline_analyzer.fc[idx], start_time
                        )  
                        
                        start_time_dict[batch][layers[0]] = end_time_dict.get(layers[0], 0) 
                        current_batch_cal.append((start_time, last_time))
                        current_batch_bubble.append({})
                        last_time_dict[layers[0]] = last_time
                        start_time = last_time
                        end_time_dict[layers[0]] = last_time
                        weight_update_start = start_time
                        if last_time > max_cost_time: 
                            max_cost_time = last_time
                    else:
                        if k == 0:
                            for i, j in enumerate(layers):     
                                idx = self.layers.index(j)
                                layer_tiles = tile_allocation[idx]
                                real_delay = get_noc_weight_delay(workload_size[idx], layer_tiles)
                                
                                current_batch_weight.append([(weight_update_start, real_delay)])
                                weight_update_start += real_delay
                                start_time = weight_update_start
                                
                                _, cur_time_scale, factor_cur, last_time, Dataspace[batch][j] = self.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[idx], workload[idx], fc=self.pipeline_analyzer.fc[idx], start_time=start_time
                                ) 
                                last_time_dict[j] = last_time
                                if i == 0:
                                    current_batch_bubble.append({})
                                    current_batch_cal.append((start_time, last_time))
                                    end_time_dict[layers[i]] = last_time 
                                time_scale.append(cur_time_scale)
                                factor.append(factor_cur)
                                
                            for i, j in enumerate(layers):  
                                idx = self.layers.index(j)
                                if idx == 0: continue
                                maxpooling = self.pipeline_analyzer.maxpool[idx-1]
                                actual_time, _, _, _, _, bubble, cal_time, _, Dataspace[batch][layers[i]] = self.pipeline_analyzer.pipeline_analysis(
                                    time_scale[i], maxpooling, self.pipeline_analyzer.fc[idx],
                                    Input_dataspace=copy.deepcopy(Dataspace[batch][layers[i-1]]), 
                                    Output_dataspace=copy.deepcopy(Dataspace[batch][layers[i]]), 
                                    shortcut=self.pipeline_analyzer.shortcut[idx]
                                )
                                
                                if actual_time > max_cost_time: 
                                    max_cost_time = actual_time
                                
                                current_batch_cal.append(cal_time)
                                current_batch_bubble.append(bubble)
                                actual_compute_time.append(actual_time)
                                current_batch_weight.append([])
                                end_time_dict[layers[i]] = actual_time
                            start_time = actual_time
                        else:
                            last_group = workload_group[k-1]
                            last_layer_tile = [tile_allocation[self.layers.index(x)] for x in last_group]
                            available_tile = self.hw['tile_num'] - sum(last_layer_tile)
                            current_layer_tile = [tile_allocation[self.layers.index(x)] for x in layers]
                            layer_p = 0 
                            
                            for i, j in enumerate(layers):  
                                if available_tile < 0: available_tile = 0
                                while available_tile < current_layer_tile[i]:
                                    if layer_p == len(last_group) - 1:
                                        available_tile = current_layer_tile[i]
                                        break
                                    weight_update_start = max(weight_update_start, end_time_dict[last_group[layer_p]])
                                    available_tile += last_layer_tile[layer_p]
                                    layer_p += 1
                                available_tile -= current_layer_tile[i]  
                                
                                idx = self.layers.index(j)
                                layer_tiles = tile_allocation[idx]
                                real_delay = get_noc_weight_delay(workload_size[idx], layer_tiles)
                                
                                current_batch_weight.append([(weight_update_start, real_delay)])
                                weight_update_start += real_delay
                                start_time = weight_update_start
                                
                                _, cur_time_scale, factor_cur, last_time, Dataspace[batch][j] = self.pipeline_analyzer.parse_dataspace(
                                    output_path_pipeline[idx], workload[idx], fc=self.pipeline_analyzer.fc[idx], start_time=start_time
                                ) 
                                start_time_dict[batch][layers[i]] = start_time
                                last_time_dict[layers[i]] = last_time
                                if idx == 0:
                                    current_batch_bubble.append({})
                                    current_batch_cal.append((start_time, last_time))
                                    end_time_dict[layers[i]] = last_time 
                                time_scale.append(cur_time_scale)
                                factor.append(factor_cur)

                            for i, j in enumerate(layers):  
                                idx = self.layers.index(j)
                                if idx == 0: continue
                                maxpooling = self.pipeline_analyzer.maxpool[idx-1]
                                actual_time, _, _, _, _, bubble, cal_time, _, Dataspace[batch][layers[i]] = self.pipeline_analyzer.pipeline_analysis(
                                    time_scale[i], maxpooling, self.pipeline_analyzer.fc[idx],
                                    Input_dataspace=copy.deepcopy(Dataspace[batch][self.layers[idx-1]]), 
                                    Output_dataspace=copy.deepcopy(Dataspace[batch][layers[i]]), 
                                    shortcut=self.pipeline_analyzer.shortcut[idx]
                                )
                                if actual_time > max_cost_time: 
                                    max_cost_time = actual_time
                                
                                current_batch_cal.append(cal_time)
                                current_batch_bubble.append(bubble)
                                actual_compute_time.append(actual_time)
                                current_batch_weight.append([])
                                end_time_dict[layers[i]] = actual_time
                            start_time = actual_time

                else:
                    if len(layers) == 1:
                        idx = self.layers.index(layers[0])
                        current_batch_weight.append([])
                        start_time = max(start_time_dict[batch].get(self.layers[idx-1], 0) if idx > 0 else 0, end_time_dict[self.layers[idx]])
                        
                        _, _, _, last_time, Dataspace[batch][layers[0]] = self.pipeline_analyzer.parse_dataspace(
                            output_path_pipeline[idx], workload[idx], self.pipeline_analyzer.fc[idx], start_time
                        )  
                        start_time_dict[batch][layers[0]] = end_time_dict.get(layers[0], 0)
                        current_batch_cal.append((start_time, last_time))
                        current_batch_bubble.append({})
                        last_time_dict[layers[0]] = last_time
                        start_time = last_time
                        end_time_dict[layers[0]] = last_time
   
                        if last_time > max_cost_time: 
                            max_cost_time = last_time
                    else:
                        for i, j in enumerate(layers):                                          
                            idx = self.layers.index(j)
                            current_batch_weight.append([])
                            start_time = max(start_time_dict[batch].get(self.layers[idx-1], 0) if idx > 0 else 0, end_time_dict[self.layers[idx]])
                            
                            _, cur_time_scale, factor_cur, last_time, Dataspace[batch][j] = self.pipeline_analyzer.parse_dataspace(
                                output_path_pipeline[idx], workload[idx], fc=self.pipeline_analyzer.fc[idx], start_time=start_time
                            ) 
                            start_time_dict[batch][layers[i]] = start_time
                            last_time_dict[layers[i]] = last_time
                            if idx == 0:
                                current_batch_bubble.append({})
                                current_batch_cal.append((start_time, last_time))
                                end_time_dict[layers[i]] = last_time 
                            time_scale.append(cur_time_scale)
                            factor.append(factor_cur)
                                
                        for i, j in enumerate(layers):  
                            idx = self.layers.index(j)
                            if idx == 0: continue
                            maxpooling = self.pipeline_analyzer.maxpool[idx-1]
                            actual_time, _, _, _, _, bubble, cal_time, _, Dataspace[batch][layers[i]] = self.pipeline_analyzer.pipeline_analysis(
                                time_scale[i], maxpooling, self.pipeline_analyzer.fc[idx],
                                Input_dataspace=copy.deepcopy(Dataspace[batch][self.layers[idx-1]]), 
                                Output_dataspace=copy.deepcopy(Dataspace[batch][layers[i]]), 
                                shortcut=self.pipeline_analyzer.shortcut[idx]
                            )
                            if actual_time > max_cost_time: 
                                max_cost_time = actual_time
                            
                            current_batch_cal.append(cal_time)
                            current_batch_bubble.append(bubble)
                            actual_compute_time.append(actual_time)
                            current_batch_weight.append([])
                            end_time_dict[layers[i]] = actual_time
                        start_time = actual_time

                for i, layer in enumerate(layers):
                    idx = self.layers.index(layer)  
                    current_layer_cal[idx].append(current_batch_cal[i])
                    current_layer_bubble[idx].append(current_batch_bubble[i])
                    current_layer_weight[idx].append(current_batch_weight[i])  
                    
        dataspace_results = [self.analyzer.input_output_gen(output_path_pipeline[i]) for i in range(len(self.layers))]
        inputs = [d[0]['inputs'] if d and len(d) > 0 else 0 for d in dataspace_results]
        outputs = [d[0]['outputs'] if d and len(d) > 0 else 0 for d in dataspace_results]
        
        pipeline_access = 0 
        for k in workload_group:
            idx_head = self.layers.index(k[0])
            idx_tail = self.layers.index(k[-1])
            pipeline_access += inputs[idx_head] + outputs[idx_tail]
            
        energy_pipeline = [
            self.analyzer.get_total_energy(output_path_pipeline[i] + "/timeloop-mapper.stats.txt") * batch_size 
            for i in range(len(self.layers))
        ]
        total_energy_pipeline = sum(energy_pipeline) + (sum(weight_access_group) + pipeline_access * batch_size) * 112.54 / 8 / 1e6
        
        logging.info("=== Evaluation Completed ===")
        logging.info(f"Current Parameters: {individual}")
        logging.info(f"Current Fitness (Latency including NoC penalty): {max_cost_time}")
        logging.info(f"Total Energy Assessment: {total_energy_pipeline}")
        
        return max_cost_time
