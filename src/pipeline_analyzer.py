import re
import os
import math
import logging
import pandas as pd
from itertools import permutations
from src.analyzer import analysis

# Configure logging for open-source standard
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class PipelineAnalyzer:
    """
    Core engine for parsing Timeloop/Cimloop dataspaces, calculating strides,
    and performing pipeline dependency and bubble analysis.
    """
    def __init__(self, config):
        self.config = config
        self.dnn_name = config.get('dnn', 'resnet18')
        self.hw = config.get('hardware', {})
        self.DNN = self.dnn_name    
        self.hardware_level = self.hw.get('hardware_level', 1)
        if self.dnn_name == "resnet18":
            self.hardware_level = 4
        self.precision = self.hw.get('precision', 16)
        
        # Internal state tracking
        self.transition_positions = []
        self.change_index = []
        self.change_index_group = []
        self.factor = []
        self.compute_num = 0
        self.factor_ignore = []
        self.time_scale = 1
        self.per_access_energy = 3.37  # glb pJ
        
        # Initialize workload analyzer
        self.analyzer = analysis(self.dnn_name, config=self.config)
        
        # Dynamically load layers from the configured workloads path
        workload_dir = os.path.join(self.config['paths']['workload_root'], self.dnn_name)
        if os.path.exists(workload_dir):
            self.layers = sorted([f.split('.')[0] for f in os.listdir(workload_dir) if f != "index.yaml" and f.endswith(".yaml")])
        else:
            logging.warning(f"Workload directory not found: {workload_dir}")
            self.layers = []

        self.transformer = config.get('transformer', False)
        self.head_num = config.get('head_num', 1)
        self.block = config.get('block', 1)
        self.shortcut = [0 for _ in range(len(self.layers))]
        
        self.dim = ['C', 'M', 'P', 'Q', 'R', 'S', 'N']
        
        # Network specific shortcut and pooling rules
        if self.dnn_name == "resnet18":
            self.maxpool = [2, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 7]
            self.fc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            if len(self.shortcut) > 17:
                self.shortcut[7] = 1
                self.shortcut[12] = 1
                self.shortcut[17] = 1
        elif self.dnn_name == "vgg16":
            self.maxpool = [0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0]
            self.fc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

    def parse_and_sort(self, lines):
        def get_key(line):
            parts = line.strip().split('/')[1:]
            nums = [int(p) for p in parts if p != '']
            return (len(nums), nums)
        return sorted(lines, key=get_key)

    def find_transition_positions(self, data):
        if isinstance(data, list):
            data = "\n".join(data)
        t_paths = re.findall(r't/[\d/]+', data)
        parsed_paths = [list(map(int, path.strip('/').split('/')[1:])) for path in t_paths]
        
        change_positions = []
        index = []
        for i in range(0, len(parsed_paths), 2):
            p1 = parsed_paths[i]
            if i + 1 < len(parsed_paths):
                p2 = parsed_paths[i+1]
                min_len = min(len(p1), len(p2))
                change_positions.append([])
                for idx in range(min_len):
                    if p1[idx] != p2[idx]:
                        if idx not in self.change_index_group:
                            self.change_index_group.append(idx)  
                        change_positions.append(idx)
                        index.append(self.change_index_group.index(idx))
                        index.append(self.change_index_group.index(idx))
        return change_positions, index

    def parse_tensor_range(self, line, keyword="Inputs"):
        pattern = fr"{keyword}:\s*\{{\s*\[([\d,\s]+):([\d,\s]+)\]"
        match = re.search(pattern, line)
        if not match:
            return None
        start = [int(x.strip()) for x in match.group(1).split(",")]
        end   = [int(x.strip()) for x in match.group(2).split(",")]
        shape = [e - s for s, e in zip(start, end)]
        return {"start": start, "end": end, "shape": shape}
    
    def get_timestamp(self, timestamp):
        match = re.search(r'Time = (\d+)', timestamp)
        return int(match.group(1)) if match else 0
    
    def get_timestamp_list(self, timestamp):
        return [list(map(int, path.strip('/').split('/')[1:])) for path in timestamp]
        
    def find_closest_multiple(self, target, factor):
        if factor == 0:
            raise ValueError("Factor cannot be zero.")
        quotient = target // factor
        multiple1 = quotient * factor
        multiple2 = (quotient + 1) * factor
        if abs(target - multiple1) <= abs(target - multiple2):
            return multiple1
        return multiple2

    def get_spatial_factors(self, file_path):
        lines = []
        map_path = os.path.join(file_path, "timeloop-mapper.map.txt")
        if not os.path.exists(map_path):
            logging.warning(f"Map file not found: {map_path}")
            return {i: 1 for i in self.dim}

        with open(map_path, "r") as f:
            lines = f.readlines()
            
        indices = {}
        pattern = r'for (\w+) in \[(\d+):(\d+)(?:,(\d+))?\) \((Spatial-[XY])\)'
        for line in lines:
            match = re.search(pattern, line)
            if match:
                variable = match.group(1)
                upper_bound = int(match.group(3))
                if variable in indices:
                    indices[variable] *= upper_bound
                else:
                    indices[variable] = upper_bound
        
        for i in self.dim:
            if i not in indices:
                indices[i] = 1
        return indices

    def get_temporal_factors(self, file_path: str):
        """
        Reads the txt file and extracts the temporal loops under each module.
        Returns [(level, loop_var, loop_len), ...]
        """
        map_path = os.path.join(file_path, "timeloop-mapper.map.txt")
        if not os.path.exists(map_path):
            return []

        with open(map_path, "r") as f:
            lines = f.readlines()

        result = []
        module_pattern = re.compile(r'^\s*([\w_]+)\s*\[.*\]')
        loop_pattern = re.compile(r'^\s*\|\s*for\s+(\w+)\s+in\s+\[\s*0\s*:(\d+)(?:,\d+)?\s*\)\s*(?!.*Spatial)')

        current_level = -1
        for line in lines:
            module_match = module_pattern.match(line)
            if module_match:
                current_level += 1
                continue

            loop_match = loop_pattern.match(line)
            if loop_match:
                loop_var = loop_match.group(1)
                loop_len = int(loop_match.group(2))
                result.append((current_level, loop_var, loop_len))
        return result

    def find_next_factor(self, factor, position, dim, flag=1):
        count = 0
        result = []
        for sub_list in factor:
            if sub_list[1] == dim:
                if flag == 1:
                    if count < position:
                        result.append(sub_list[2])   
                        count += 1            
                        continue
                elif flag == 0:
                    if count > position:
                        result.append(sub_list[2])
                        count += 1 
                        continue
            count += 1
        return result if result else [1]

    def find_tiling_factor_base(self, tilling_loop_value, position, dim, flag):
        count = 0
        ret = []
        for value in tilling_loop_value:
            if value[0] == dim:
                if flag == 0:
                    if count < position:
                        ret.append(value[1])
                else:
                    if count > position:
                        ret.append(value[1])
            count += 1
        return ret if ret else [0]

    def nearest_power_of_two(self, n):
        exponent = math.log2(n)
        lower_power = 2 ** math.floor(exponent)
        upper_power = 2 ** math.ceil(exponent)
        return lower_power if abs(n - lower_power) < abs(n - upper_power) else upper_power

    def find_imperfect_factor(self, factor, spatial_factor, workload):
        imperfect_factor = []
        for key, value in workload.items():
            if key not in self.dim: continue
            temporal_factor = 1
            for item in factor:
                if key in item:
                    temporal_factor *= item[2]
            factor_temp = temporal_factor * spatial_factor[key]
            if factor_temp != value:
                if temporal_factor == 1:
                    spatial_factor[key] = value
                else:
                    imperfect_factor.append(key)
        return imperfect_factor

    def parse_stride(self, data, factor, spatial_factor, fc, workload, start_time):
        """
        [Extracted from processor] 
        Complex state machine for parsing loop strides based on tiling factors.
        """
        new_data = []
        factor_ = [[] for _ in range(self.hardware_level)]
        dimension = []
        max_actual_time = 1

        imperfect_factor = self.find_imperfect_factor(factor, spatial_factor, workload)

        for x in factor:
            if x[0] < self.hardware_level:
                factor_[x[0]].append(x)
                dimension.append(x[1])

        loop_num = 0
        psum_index = []
        psum_flag = False
        
        for x in range(self.hardware_level):          
            for y in factor_[x]:
                max_actual_time *= y[2] 
                if y[1] == 'C':
                    psum_flag = True
                    psum_index.append(loop_num)
                loop_num += 1
                
        t = 0
        cur_tiling_factor_index = [0 for _ in range(loop_num)]
        pre_tiling_factor_index = [0 for _ in range(loop_num)]
        tiling_factor_value = [[key, 0] for key in dimension]
        
        imperfect_factor_dict = {k: [] for k in imperfect_factor}
        if imperfect_factor:
            for imp_factor in imperfect_factor:
                for index, d in enumerate(tiling_factor_value):
                    if imp_factor in d:
                        imperfect_factor_dict[imp_factor].append(index)
                        
        input_start = {'N':0, 'C':0, 'P':0, 'Q':0, 'X':0, 'G':0} 
        input_end   = {'N':1, 'C':0, 'P':0, 'Q':0, 'X':16, 'G':1} 
        output_start = {'N':0, 'M':0, 'Q':0, 'P':0, 'Z':0, 'G':0} 
        output_end   = {'N':1, 'M':0, 'Q':0, 'P':0, 'Z':16, 'G':1} 
        weight_start = {'C':0, 'M':0, 'R':0, 'S':0, 'Y':0, 'G':0} 
        weight_end   = {'C':0, 'M':0, 'R':0, 'S':0, 'Y':16, 'G':1} 

        spatial_factor['R'] = workload.get('R', 1)
        spatial_factor['S'] = workload.get('S', 1)
        
        for i in range(0, max_actual_time): 
            t_tmp = t // self.time_scale
            for j in range(loop_num):
                product = 1
                for k in range(loop_num):
                    if k > j:
                        product *= factor[k][2]
                cur_tiling_factor_index[j] = t_tmp // product
                if cur_tiling_factor_index[j] > 0:
                    t_tmp = t_tmp % (product * cur_tiling_factor_index[j])
            
            factor_diff = [n - c for n, c in zip(cur_tiling_factor_index, pre_tiling_factor_index)]
            for k in range(loop_num):
                if factor_diff[k] < 0:
                    factor_diff[k] = 1
                    
            pre_tiling_factor_index = cur_tiling_factor_index.copy()
            output_psum = psum_flag
            num = sum(x == 1 for x in factor_diff)
            positions = loop_num - num
            
            if(i==0):
                if(fc==0):   
                    input_end['C'] = spatial_factor['C'] 
                else:
                    input_end['C'] = spatial_factor['C']
                input_end['P'] = spatial_factor['R'] + spatial_factor["P"]-1
                input_end['Q'] = spatial_factor['S'] + spatial_factor["Q"]-1
                output_end['M'] = spatial_factor['M']
                output_end['Q'] = spatial_factor["Q"]
                output_end['P'] = spatial_factor["P"]
                
                weight_end['C']=spatial_factor['C']
                weight_end['M']=spatial_factor['M']
                weight_end['R']=spatial_factor['R']
                weight_end['S']=spatial_factor['S']
                if(psum_flag):   
                    if (psum_index[0] <= positions):

                        if tiling_factor_value[psum_index[0]][1] != factor_[0][psum_index[0]][2]-1:
                            output_psum = True
            else :
                last_iteration = True
                if(num==1):
                    tiling_factor_value[positions][1]  +=1
                    if(factor[positions][1] == "Q"):
                        input_start ["Q"] += spatial_factor["Q"]
                        #input_end ["Q"] = input_start["Q"] + spatial_factor["S"]-1+spatial_factor["Q"]
                        output_start["Q"] += spatial_factor["Q"]
                        #output_end ["Q"] += spatial_factor["Q"]
                        if(factor[positions][1] in imperfect_factor_dict):
                            for i in range(len(imperfect_factor_dict[factor[positions][1]])):
                                last_iteration &=  tiling_factor_value[imperfect_factor_dict[factor[positions][1]][i]][1] == factor[imperfect_factor_dict[factor[positions][1]][i]][2]-1
                        else : last_iteration = False
                        if(last_iteration):
                            output_end["Q"]=workload["Q"]
                            input_end ["Q"] = workload["Q"]+spatial_factor["S"]-1
                        else:
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1
                        
                    elif (factor[positions][1] == "P"):
                            input_start ["P"] += spatial_factor["P"]
                            #input_end ["P"] = input_start["P"] + spatial_factor["R"]-1+spatial_factor["P"]
                            output_start["P"] += spatial_factor["P"]
                            #output_end ["P"] += spatial_factor["P"]
                            
                            if(factor[positions][1] in imperfect_factor_dict):
                                for i in range(len(imperfect_factor_dict[factor[positions][1]])):
                                    last_iteration &=  tiling_factor_value[imperfect_factor_dict[factor[positions][1]][i]][1] == factor[imperfect_factor_dict[factor[positions][1]][i]][2]-1
                            else : last_iteration = False
                            if(last_iteration):
                                output_end["P"]=workload["P"]
                                input_end ["P"] = workload["P"]+spatial_factor["R"]-1
                            else:
                                output_end["P"] =  output_start["P"]+ spatial_factor["P"]
                                input_end ["P"] =  input_start["P"] + spatial_factor["P"]+spatial_factor["R"]-1
                    elif (factor[positions][1] == "C"):
                        input_start["C"] += spatial_factor["C"]
    

                        weight_start["C"] += spatial_factor["C"]

                        if(factor[positions][1] in imperfect_factor_dict):
                                for i in range(len(imperfect_factor_dict[factor[positions][1]])):
                                    last_iteration &=  tiling_factor_value[imperfect_factor_dict[factor[positions][1]][i]][1] == factor[imperfect_factor_dict[factor[positions][1]][i]][2]-1
                        else : last_iteration = False
                        if(last_iteration):
                                input_end ["C"] = workload["C"]
                                weight_end ["C"] = workload["C"]
                        else:
                                input_end ["C"] =  input_start["C"] + spatial_factor["C"]
                                weight_end ["C"] = weight_end["C"]+ spatial_factor["C"]
                    elif(factor[positions][1] == "M"):
                        output_start["M"] += spatial_factor["M"]
                        output_end["M"] += spatial_factor["M"]

                        weight_start["M"] += spatial_factor["M"]
                        weight_end["M"] += spatial_factor["M"]
                elif num ==2 :
                    tiling_factor_value[positions][1] +=1
                    tiling_factor_value[positions+1][1]  = 0
                    if(factor[positions][1] == "P"):     
                        next_factor = self.find_next_factor(factor,positions,"P",0)
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"P",0)
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"P",1)
                        input_start ["P"] = base_up[0]*spatial_factor["P"]*next_factor[0]*factor[positions][2] + spatial_factor["P"]*tiling_factor_value[positions][1]*next_factor[0]  #FIXME
                        output_start["P"] =base_up[0]*spatial_factor["P"]*next_factor[0]*factor[positions][2] + spatial_factor["P"]*tiling_factor_value[positions][1]*next_factor[0]  #FIXME
                        if(factor[positions][1] in imperfect_factor_dict):
                            for i in range(len(imperfect_factor_dict[factor[positions][1]])):
                                last_iteration &=  tiling_factor_value[imperfect_factor_dict[factor[positions][1]][i]][1] == factor[imperfect_factor_dict[factor[positions][1]][i]][2]-1
                        else : last_iteration = False
                        if(last_iteration):
                            output_end["P"]=workload["P"]
                            input_end ["P"] = workload["P"]+spatial_factor["R"]-1
                        else:
                            output_end["P"] =  output_start["P"]+ spatial_factor["P"]
                            input_end ["P"] =  input_start["P"] + spatial_factor["P"]+spatial_factor["R"]-1
                        
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",0)
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",1)
                        next_factor = self.find_next_factor(factor,positions,"Q",0)
                        input_start ["Q"] = base_up[0]*spatial_factor["Q"]*next_factor[0] + base_down[0]*spatial_factor["Q"]
                        input_end ["Q"] = input_start ["Q"]+spatial_factor['S'] + spatial_factor["Q"]-1
                        output_start["Q"] = base_up[0]*spatial_factor["Q"]*next_factor[0] + base_down[0]*spatial_factor["Q"]
                        output_end["Q"] = output_start["Q"] + spatial_factor["Q"]

                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"M",0)
                        output_start["M"] = base_up[0]*spatial_factor["M"]
                        output_end["M"] = output_start["M"] + spatial_factor["M"]
                        weight_start["M"] = base_up[0]*spatial_factor["M"]
                        weight_end["M"] = weight_start["M"] + spatial_factor["M"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"C",0)
                        input_start ["C"] = base_up[0]*spatial_factor["C"]
                        input_end ["C"] = input_start["C"]+spatial_factor["C"]
                        weight_start["C"] = base_up[0]*spatial_factor["C"]
                        weight_end["C"] = weight_start["C"] + spatial_factor["C"]

                        if("C" in imperfect_factor_dict):
                                for i in range(len(imperfect_factor_dict["C"])):
                                    last_iteration =  tiling_factor_value[imperfect_factor_dict["C"][i]][1] == factor[imperfect_factor_dict["C"][i]][2]-1
                        else : last_iteration = False
                        
                        if(last_iteration):
                                input_end ["C"] = workload["C"]
                                weight_end ["C"] = workload["C"]
                        else:
                                input_end ["C"] =  input_start["C"] + spatial_factor["C"]
                                weight_end ["C"] = weight_end["C"]+ spatial_factor["C"]


                    elif(factor[positions][1] == "Q"):  # travel Q
                        next_factor = self.find_next_factor(factor,positions,"Q",0)
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",0)
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",1)

                        input_start ["Q"] =  base_up[0]*spatial_factor["Q"]*next_factor[0]+ spatial_factor["Q"]*tiling_factor_value[positions][1]              
                        output_start["Q"] =base_up[0]*spatial_factor["Q"]*next_factor[0]+spatial_factor["Q"]*tiling_factor_value[positions][1]
                        if(factor[positions][1] in imperfect_factor_dict):

                            for i in range(len(imperfect_factor_dict[factor[positions][1]])):
                                last_iteration &=  tiling_factor_value[imperfect_factor_dict[factor[positions][1]][i]][1] == factor[imperfect_factor_dict[factor[positions][1]][i]][2]-1
                        else : last_iteration = False
                        if(last_iteration):
                            output_end["Q"]=workload["Q"]
                            input_end ["Q"] = workload["Q"]+spatial_factor["S"]-1
                        else:
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1

                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"P",0)
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"P",1)
                        next_factor = self.find_next_factor(factor,positions,"P",0)
                        input_start ["P"] = base_up[0]*spatial_factor["P"]*next_factor[0] + base_down[0]*spatial_factor["P"] #init P
                        input_end ["P"] = input_start ["P"]+spatial_factor['R'] + spatial_factor["P"]-1
                        output_start["P"] = base_up[0]*spatial_factor["P"]*next_factor[0] + base_down[0]*spatial_factor["P"] 
                        output_end["P"] = output_start["P"] + spatial_factor["P"]

                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"M",0)
                        output_start["M"] = base_up[0]*spatial_factor["M"]
                        output_end["M"] = output_start["M"] + spatial_factor["M"]
                        weight_start["M"] = base_up[0]*spatial_factor["M"]
                        weight_end["M"] = weight_start["M"] + spatial_factor["M"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"C",0)
                        input_start ["C"] = base_up[0]*spatial_factor["C"]
                        input_end ["C"] = input_start["C"]+spatial_factor["C"]
                        weight_start["C"] = base_up[0]*spatial_factor["C"]
                        weight_end["C"] = weight_start["C"] + spatial_factor["C"]

                        if("C" in imperfect_factor_dict):
                                for i in range(len(imperfect_factor_dict["C"])):
                                    last_iteration =  tiling_factor_value[imperfect_factor_dict["C"][i]][1] == factor[imperfect_factor_dict["C"][i]][2]-1
                        else : last_iteration = False
                        
                        if(last_iteration):
                                input_end ["C"] = workload["C"]
                                weight_end ["C"] = workload["C"]
                        else:
                                input_end ["C"] =  input_start["C"] + spatial_factor["C"]
                                weight_end ["C"] = weight_end["C"]+ spatial_factor["C"]
                    elif (factor[positions][1] == "C"):
                        next_factor_down = self.find_next_factor(factor,positions,"Q",0)
                        next_factor_up = self.find_next_factor(factor,positions,"Q",1)
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",0)
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",1)
                        if(len(base_up)==1):
                            input_start ["Q"] =  base_up[0]*spatial_factor["Q"] *next_factor_down[0]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1
                            output_start["Q"] =base_up[0]*spatial_factor["Q"] *next_factor_down[0]
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]
                        else:
                            input_start ["Q"] = base_up[0]*spatial_factor["Q"]*next_factor_up[1] + base_up[1]*spatial_factor["Q"]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1
                            output_start["Q"] =base_up[0]*spatial_factor["Q"]*next_factor_up[1] + base_up[1]*spatial_factor["Q"]
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]
                       
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"P",0)
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"P",1)
                        next_factor_down = self.find_next_factor(factor,positions,"P",0)
                        next_factor_up = self.find_next_factor(factor,positions,"P",1)
                        if(len(base_up)==1):
                            input_start ["P"] = base_up[0]*spatial_factor["P"]*next_factor_down[0]
                            input_end ["P"] = input_start ["P"]+spatial_factor['R'] + spatial_factor["P"]-1
                            output_start["P"] = base_up[0]*spatial_factor["P"]*next_factor_down[0]
                            output_end["P"] = output_start["P"] + spatial_factor["P"]
                        else:
                            input_start ["P"] = base_up[0]*spatial_factor["P"]*next_factor_up[1] + base_up[1]*spatial_factor["P"]
                            input_end ["P"] =  input_start["P"] + spatial_factor["P"]+spatial_factor["R"]-1
                            output_start["P"] =base_up[0]*spatial_factor["P"]*next_factor_up[1] + base_up[1]*spatial_factor["P"]
                            output_end["P"] = output_start["P"]+ spatial_factor["P"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"M",0)
                        next_factor = self.find_next_factor(factor,positions,"M",0)

                        output_start["M"] = base_up[0]*spatial_factor["M"]*next_factor[0]
                        output_end["M"] = output_start["M"] + spatial_factor["M"]

                        if(factor[positions][1] in imperfect_factor_dict):
                                for i in range(len(imperfect_factor_dict[factor[positions][1]])):
                                    last_iteration &=  tiling_factor_value[imperfect_factor_dict[factor[positions][1]][i]][1] == factor[imperfect_factor_dict[factor[positions][1]][i]][2]-1
                        else : last_iteration = False
                        input_start ["C"] = spatial_factor["C"]*tiling_factor_value[positions][1]
                        weight_start ["C"] = spatial_factor["C"]*tiling_factor_value[positions][1]
                        if(last_iteration):
                                input_end ["C"] = workload["C"]
                                weight_end ["C"] = workload["C"]
                        else:
                                input_end ["C"] =  input_start["C"] + spatial_factor["C"]
                                weight_end ["C"] = weight_end["C"]+ spatial_factor["C"]



                    elif(factor[positions][1] == "M"):  
                        next_factor_down = self.find_next_factor(factor,positions,"Q",0)
                        next_factor_up = self.find_next_factor(factor,positions,"Q",1)
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",0)
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",1)
                        if(len(base_up)==1):
                            input_start ["Q"] =  base_up[0]*spatial_factor["Q"] *next_factor_down[0]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1
                            output_start["Q"] =base_up[0]*spatial_factor["Q"] *next_factor_down[0]
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]
                        else:
                            input_start ["Q"] = base_up[0]*spatial_factor["Q"]*next_factor_up[1] + base_up[1]*spatial_factor["Q"]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1
                            output_start["Q"] =base_up[0]*spatial_factor["Q"]*next_factor_up[1] + base_up[1]*spatial_factor["Q"]
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]
                       
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"P",0)
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"P",1)
                        next_factor_down = self.find_next_factor(factor,positions,"P",0)
                        next_factor_up = self.find_next_factor(factor,positions,"P",1)
                        if(len(base_up)==1):
                            input_start ["P"] = base_up[0]*spatial_factor["P"]*next_factor_down[0]
                            input_end ["P"] = input_start ["P"]+spatial_factor['R'] + spatial_factor["P"]-1
                            output_start["P"] = base_up[0]*spatial_factor["P"]*next_factor_down[0]
                            output_end["P"] = output_start["P"] + spatial_factor["P"]
                        else:
                            input_start ["P"] = base_up[0]*spatial_factor["P"]*next_factor_up[1] + base_up[1]*spatial_factor["P"]
                            input_end ["P"] =  input_start["P"] + spatial_factor["P"]+spatial_factor["R"]-1
                            output_start["P"] =base_up[0]*spatial_factor["P"]*next_factor_up[1] + base_up[1]*spatial_factor["P"]
                            output_end["P"] = output_start["P"]+ spatial_factor["P"]
                        

                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"C",0)
                        next_factor = self.find_next_factor(factor,positions,"C",0)
                        input_start ["C"] = base_up[0]*spatial_factor["C"]*next_factor[0]  #init M
                        input_end ["C"] = input_start ["C"]+spatial_factor['C'] 
                        
                        weight_start["C"] = base_up[0]*spatial_factor["C"]*next_factor[0]
                        weight_end["C"] = weight_start ["C"]+spatial_factor['C']

                        output_start ["M"] = spatial_factor["M"]*tiling_factor_value[positions][1]
                        output_end ["M"] = output_start["M"]+spatial_factor["M"]    
                        
                        
                        weight_start ["M"] = spatial_factor["M"]*tiling_factor_value[positions][1]
                        weight_end ["M"] = weight_start["M"]+spatial_factor["M"]  
                        
                elif num ==3 :
                    tiling_factor_value[positions][1]  +=1
                    tiling_factor_value[positions+1][1]  =0
                    tiling_factor_value[positions+2][1]  =0
                    if(factor[positions][1] == "P"):
                        next_factor = self.find_next_factor(factor,positions,"P",0)
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"P",0)
                        input_start["P"] = base_up[0]*next_factor[0]*spatial_factor["P"] + spatial_factor["P"]*tiling_factor_value[positions][1] *next_factor[0]
                        #input_end ["P"] = input_start["P"] + spatial_factor["P"]+ spatial_factor['R'] -1
                        output_start["P"] = base_up[0]*next_factor[0]*spatial_factor["P"] + spatial_factor["P"]*tiling_factor_value[positions][1] *next_factor[0]
                        #output_end["P"] = output_start["P"] + spatial_factor["P"]
                        if(factor[positions][1] in imperfect_factor_dict):
                            for i in range(len(imperfect_factor_dict[factor[positions][1]])):
                                last_iteration &=  tiling_factor_value[imperfect_factor_dict[factor[positions][1]][i]][1] == factor[imperfect_factor_dict[factor[positions][1]][i]][2]-1
                        else : last_iteration = False
                        if(last_iteration):
                            output_end["P"]=workload["P"]
                            input_end ["P"] = workload["P"]+spatial_factor["R"]-1
                        else:
                            output_end["P"] = output_start["P"]+ spatial_factor["P"]
                            input_end ["P"] =  input_start["P"] + spatial_factor["P"]+spatial_factor["R"]-1
                        next_factor_down = self.find_next_factor(factor,positions,"Q",0)
                        next_factor_up = self.find_next_factor(factor,positions,"Q",1)
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",0)
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",1)
                        if(len(base_up)==1):
                            input_start ["Q"] =  base_up[0]*spatial_factor["Q"] *next_factor_down[0]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1
                            output_start["Q"] =base_up[0]*spatial_factor["Q"] *next_factor_down[0]
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]
                        else:
                            input_start ["Q"] = base_up[0]*spatial_factor["Q"]*next_factor_up[1] + base_up[1]*spatial_factor["Q"]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1
                            output_start["Q"] =base_up[0]*spatial_factor["Q"]*next_factor_up[1] + base_up[1]*spatial_factor["Q"]
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]

                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"M",0)
                        output_start["M"] = base_up[0]*spatial_factor["M"]
                        output_end["M"] = output_start["M"] + spatial_factor["M"]
                        weight_start["M"] = base_up[0]*spatial_factor["M"]
                        weight_end["M"] = weight_start["M"] + spatial_factor["M"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"C",0)
                        input_start ["C"] = base_up[0]*spatial_factor["C"]
                        input_end ["C"] = input_start["C"]+spatial_factor["C"]
                        weight_start["C"] = base_up[0]*spatial_factor["C"]
                        weight_end["C"] = weight_start["C"] + spatial_factor["C"]

                        if("C" in imperfect_factor_dict):
                                for i in range(len(imperfect_factor_dict["C"])):
                                    last_iteration =  tiling_factor_value[imperfect_factor_dict["C"][i]][1] == factor[imperfect_factor_dict["C"][i]][2]-1
                        else : last_iteration = False
                        
                        if(last_iteration):
                                input_end ["C"] = workload["C"]
                                weight_end ["C"] = workload["C"]
                        else:
                                input_end ["C"] =  input_start["C"] + spatial_factor["C"]
                                weight_end ["C"] = weight_end["C"]+ spatial_factor["C"]
                    elif(factor[positions][1] == "Q"):
                        next_factor = self.find_next_factor(factor,positions,"Q",0)
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",0)
                        input_start["Q"] =  base_up[0]*next_factor[0]*spatial_factor["Q"] + spatial_factor["Q"]*tiling_factor_value[positions][1] 
                        #input_end ["Q"] = input_start["Q"] + spatial_factor["Q"]+ spatial_factor['S'] -1
                        output_start["Q"] = base_up[0]*next_factor[0]*spatial_factor["Q"] + spatial_factor["Q"]*tiling_factor_value[positions][1] 
                       # output_end["Q"] = output_start["Q"] + spatial_factor["Q"]
                        if(factor[positions][1] in imperfect_factor_dict):
                            for i in range(len(imperfect_factor_dict[factor[positions][1]])):
                                last_iteration &=  tiling_factor_value[imperfect_factor_dict[factor[positions][1]][i]][1] == factor[imperfect_factor_dict[factor[positions][1]][i]][2]-1
                        else : last_iteration = False
                        if(last_iteration):
                            output_end["Q"]=workload["Q"]
                            input_end ["Q"] = workload["Q"]+spatial_factor["S"]-1
                        else:
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"P",0)
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"P",1)
                        next_factor_down = self.find_next_factor(factor,positions,"P",0)
                        next_factor_up = self.find_next_factor(factor,positions,"P",1)
                        if(len(base_up)==1):
                            input_start ["P"] = base_up[0]*spatial_factor["P"]*next_factor_down[0]
                            input_end ["P"] = input_start ["P"]+spatial_factor['R'] + spatial_factor["P"]-1
                            output_start["P"] = base_up[0]*spatial_factor["P"]*next_factor_down[0]
                            output_end["P"] = output_start["P"] + spatial_factor["P"]
                        else:
                            input_start ["P"] = base_up[0]*spatial_factor["P"]*next_factor_up[1] + base_up[1]*spatial_factor["P"]
                            input_end ["P"] =  input_start["P"] + spatial_factor["P"]+spatial_factor["R"]-1
                            output_start["P"] =base_up[0]*spatial_factor["P"]*next_factor_up[1] + base_up[1]*spatial_factor["P"]
                            output_end["P"] = output_start["P"]+ spatial_factor["P"]
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"M",0)
                        output_start["M"] = base_up[0]*spatial_factor["M"]
                        output_end["M"] = output_start["M"] + spatial_factor["M"]
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"C",0)
                        input_start ["C"] = base_up[0]*spatial_factor["C"]
                        input_end ["C"] = input_start["C"]+spatial_factor["C"]
                       
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"M",0)
                        output_start["M"] = base_up[0]*spatial_factor["M"]
                        output_end["M"] = output_start["M"] + spatial_factor["M"]
                        weight_start["M"] = base_up[0]*spatial_factor["M"]
                        weight_end["M"] = weight_start["M"] + spatial_factor["M"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"C",0)
                        input_start ["C"] = base_up[0]*spatial_factor["C"]
                        input_end ["C"] = input_start["C"]+spatial_factor["C"]
                        weight_start["C"] = base_up[0]*spatial_factor["C"]
                        weight_end["C"] = weight_start["C"] + spatial_factor["C"]

                        if("C" in imperfect_factor_dict):
                                for i in range(len(imperfect_factor_dict["C"])):
                                    last_iteration =  tiling_factor_value[imperfect_factor_dict["C"][i]][1] == factor[imperfect_factor_dict["C"][i]][2]-1
                        else : last_iteration = False
                        
                        if(last_iteration):
                                input_end ["C"] = workload["C"]
                                weight_end ["C"] = workload["C"]
                        else:
                                input_end ["C"] =  input_start["C"] + spatial_factor["C"]
                                weight_end ["C"] = weight_end["C"]+ spatial_factor["C"]
                    elif (factor[positions][1] == "C"):
                        next_factor_down = self.find_next_factor(factor,positions,"Q",0)
                        next_factor_up = self.find_next_factor(factor,positions,"Q",1)
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",0)
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",1)
                        if(len(base_up)==1):
                            input_start ["Q"] =  base_up[0]*spatial_factor["Q"] *next_factor_down[0]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1
                            output_start["Q"] =base_up[0]*spatial_factor["Q"] *next_factor_down[0]
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]
                        else:
                            input_start ["Q"] = base_up[0]*spatial_factor["Q"]*next_factor_up[1] + base_up[1]*spatial_factor["Q"]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1
                            output_start["Q"] =base_up[0]*spatial_factor["Q"]*next_factor_up[1] + base_up[1]*spatial_factor["Q"]
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]
                       
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"P",0)
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"P",1)
                        next_factor_down = self.find_next_factor(factor,positions,"P",0)
                        next_factor_up = self.find_next_factor(factor,positions,"P",1)
                        if(len(base_up)==1):
                            input_start ["P"] = base_up[0]*spatial_factor["P"]*next_factor_down[0]
                            input_end ["P"] = input_start ["P"]+spatial_factor['R'] + spatial_factor["P"]-1
                            output_start["P"] = base_up[0]*spatial_factor["P"]*next_factor_down[0]
                            output_end["P"] = output_start["P"] + spatial_factor["P"]
                        else:
                            input_start ["P"] = base_up[0]*spatial_factor["P"]*next_factor_up[1] + base_up[1]*spatial_factor["P"]
                            input_end ["P"] =  input_start["P"] + spatial_factor["P"]+spatial_factor["R"]-1
                            output_start["P"] =base_up[0]*spatial_factor["P"]*next_factor_up[1] + base_up[1]*spatial_factor["P"]
                            output_end["P"] = output_start["P"]+ spatial_factor["P"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"M",0)
                        next_factor = self.find_next_factor(factor,positions,"M",0)
                        output_start["M"] = base_up[0]*spatial_factor["M"]*next_factor[0]
                        output_end["M"] = output_start["M"] + spatial_factor["M"]

                        if(factor[positions][1] in imperfect_factor_dict):
                                for i in range(len(imperfect_factor_dict[factor[positions][1]])):
                                    last_iteration &=  tiling_factor_value[imperfect_factor_dict[factor[positions][1]][i]][1] == factor[imperfect_factor_dict[factor[positions][1]][i]][2]-1
                        else : last_iteration = False
                        input_start ["C"] = spatial_factor["C"]*tiling_factor_value[positions][1]
                        weight_start ["C"] = spatial_factor["C"]*tiling_factor_value[positions][1]
                        if(last_iteration):
                                input_end ["C"] = workload["C"]
                                weight_end ["C"] = workload["C"]
                        else:
                                input_end ["C"] =  input_start["C"] + spatial_factor["C"]
                                weight_end ["C"] = weight_end["C"]+ spatial_factor["C"]

                    elif(factor[positions][1] == "M"):  
                        next_factor_down = self.find_next_factor(factor,positions,"Q",0)
                        next_factor_up = self.find_next_factor(factor,positions,"Q",1)
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",0)
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",1)
                        if(len(base_up)==1):
                            input_start ["Q"] =  base_up[0]*spatial_factor["Q"] *next_factor_down[0]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1
                            output_start["Q"] =base_up[0]*spatial_factor["Q"] *next_factor_down[0]
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]
                        else:
                            input_start ["Q"] = base_up[0]*spatial_factor["Q"]*next_factor_up[1] + base_up[1]*spatial_factor["Q"]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1
                            output_start["Q"] =base_up[0]*spatial_factor["Q"]*next_factor_up[1] + base_up[1]*spatial_factor["Q"]
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]
                       
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"P",0)
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"P",1)
                        next_factor_down = self.find_next_factor(factor,positions,"P",0)
                        next_factor_up = self.find_next_factor(factor,positions,"P",1)
                        if(len(base_up)==1):
                            input_start ["P"] = base_up[0]*spatial_factor["P"]*next_factor_down[0]
                            input_end ["P"] = input_start ["P"]+spatial_factor['R'] + spatial_factor["P"]-1
                            output_start["P"] = base_up[0]*spatial_factor["P"]*next_factor_down[0]
                            output_end["P"] = output_start["P"] + spatial_factor["P"]
                        else:
                            input_start ["P"] = base_up[0]*spatial_factor["P"]*next_factor_up[1] + base_up[1]*spatial_factor["P"]
                            input_end ["P"] =  input_start["P"] + spatial_factor["P"]+spatial_factor["R"]-1
                            output_start["P"] =base_up[0]*spatial_factor["P"]*next_factor_up[1] + base_up[1]*spatial_factor["P"]
                            output_end["P"] = output_start["P"]+ spatial_factor["P"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"M",0)
                        next_factor = self.find_next_factor(factor,positions,"M",0)

                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"C",0)
                        next_factor = self.find_next_factor(factor,positions,"C",0)
                        input_start ["C"] = base_up[0]*spatial_factor["C"]*next_factor[0]  #init M
                        input_end ["C"] = input_start ["C"]+spatial_factor['C'] 

                        weight_start ["C"] = base_up[0]*spatial_factor["C"]*next_factor[0]  #init M
                        weight_end ["C"] = weight_start ["C"]+spatial_factor['C']

                        output_start ["M"] = spatial_factor["M"]*tiling_factor_value[positions][1]
                        output_end ["M"] = output_start["M"]+spatial_factor["M"]       
                        
                        weight_start ["M"] = spatial_factor["M"]*tiling_factor_value[positions][1]
                        weight_end ["M"] = weight_start["M"]+spatial_factor["M"]  

                 
                elif num ==4 :
                    tiling_factor_value[positions][1]  +=1
                    tiling_factor_value[positions+1][1]  =0
                    tiling_factor_value[positions+2][1]  =0
                    tiling_factor_value[positions+3][1]  =0
                    if(factor[positions][1] == "P"):
                        next_factor = self.find_next_factor(factor,positions,"P",0)
                        input_start["P"] = next_factor[0]*spatial_factor["P"]*tiling_factor_value[positions][1] 
                        #input_end ["P"] = input_start["P"] + spatial_factor["P"]+ spatial_factor['R'] -1
                        output_start["P"] = next_factor[0]*spatial_factor["P"]*tiling_factor_value[positions][1]
                        #output_end["P"] = output_start["P"] + spatial_factor["P"]
                        if(factor[positions][1] in imperfect_factor_dict):
                            for i in range(len(imperfect_factor_dict[factor[positions][1]])):
                                last_iteration &=  tiling_factor_value[imperfect_factor_dict[factor[positions][1]][i]][1] == factor[imperfect_factor_dict[factor[positions][1]][i]][2]-1
                        else : last_iteration = False
                        if(last_iteration):
                            output_end["P"]=workload["P"]
                            input_end ["P"] = workload["P"]+spatial_factor["R"]-1
                        else:
                            output_end["P"] = output_start["P"]+ spatial_factor["P"]
                            input_end ["P"] =  input_start["P"] + spatial_factor["P"]+spatial_factor["R"]-1
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",1)
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",0)
                        next_factor = self.find_next_factor(factor,positions,"Q",0)
                        input_start["Q"] =next_factor[0]*spatial_factor["Q"]*base_up[0] + base_down[0] *spatial_factor["Q"]
                        input_end["Q"] = input_start["Q"]+ spatial_factor["Q"] + spatial_factor["S"]-1
                        output_start["Q"] = next_factor[0]*spatial_factor["Q"]*base_up[0] + base_down[0]*spatial_factor["Q"]
                        output_end["Q"] =output_start["Q"]+ spatial_factor["Q"]

                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"M",0)
                        output_start["M"] = base_up[0]*spatial_factor["M"]
                        output_end["M"] = output_start["M"] + spatial_factor["M"]
                        weight_start["M"] = base_up[0]*spatial_factor["M"]
                        weight_end["M"] = weight_start["M"] + spatial_factor["M"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"C",0)
                        input_start ["C"] = base_up[0]*spatial_factor["C"]
                        input_end ["C"] = input_start["C"]+spatial_factor["C"]
                        weight_start["C"] = base_up[0]*spatial_factor["C"]
                        weight_end["C"] = weight_start["C"] + spatial_factor["C"]
                    elif(factor[positions][1] == "Q"):
                        next_factor = self.find_next_factor(factor,positions,"Q",0)
                        input_start["Q"] = next_factor[0]*spatial_factor["Q"]*tiling_factor_value[positions][1] 
                        #input_end ["Q"] = input_start["Q"] + spatial_factor["Q"]+ spatial_factor['S'] -1
                        output_start["Q"] = next_factor[0]*spatial_factor["Q"]*tiling_factor_value[positions][1]
                        #output_end["Q"] = output_start["Q"] + spatial_factor["Q"]
                        if(factor[positions][1] in imperfect_factor_dict):
                            for i in range(len(imperfect_factor_dict[factor[positions][1]])):
                                last_iteration &=  tiling_factor_value[imperfect_factor_dict[factor[positions][1]][i]][1] == factor[imperfect_factor_dict[factor[positions][1]][i]][2]-1
                        else : last_iteration = False
                        if(last_iteration):
                            output_end["Q"]=workload["Q"]
                            input_end ["Q"] = workload["Q"]+spatial_factor["S"]-1
                        else:
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"P",1)
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"P",0)
                        next_factor = self.find_next_factor(factor,positions,"P",0)
                        input_start["P"] =next_factor[0]*spatial_factor["P"]*base_up[0] + base_down[0] *spatial_factor["P"]
                        input_end["P"] = input_start["P"]+ spatial_factor["P"] + spatial_factor["R"]-1
                        output_start["P"] = next_factor[0]*spatial_factor["P"]*base_up[0] + base_down[0]*spatial_factor["P"]
                        output_end["P"] =output_start["P"]+ spatial_factor["P"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"M",0)
                        output_start["M"] = base_up[0]*spatial_factor["M"]
                        output_end["M"] = output_start["M"] + spatial_factor["M"]
                        weight_start["M"] = base_up[0]*spatial_factor["M"]
                        weight_end["M"] = weight_start["M"] + spatial_factor["M"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"C",0)
                        input_start ["C"] = base_up[0]*spatial_factor["C"]
                        input_end ["C"] = input_start["C"]+spatial_factor["C"]
                        weight_start["C"] = base_up[0]*spatial_factor["C"]
                        weight_end["C"] = weight_start["C"] + spatial_factor["C"]
                    elif (factor[positions][1] == "C"):
                        next_factor = self.find_next_factor(factor,positions,"Q")
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",0)
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",1)
                        if(len(base_up)==1):
                            input_start ["Q"] =  base_up[0]*spatial_factor["Q"]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1
                            output_start["Q"] =base_up[0]*spatial_factor["Q"]
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]
                        else:
                            input_start ["Q"] = base_up[0]*spatial_factor["Q"]*base_up[1]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1
                            output_start["Q"] =base_up[0]*spatial_factor["Q"]*base_up[1]
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"P",0)
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"P",1)
                        next_factor = self.find_next_factor(factor,positions,"P")
                        if(len(base_up)==1):
                            input_start ["P"] = base_up[0]*spatial_factor["P"] #init P
                            input_end ["P"] = input_start ["P"]+spatial_factor['R'] + spatial_factor["P"]-1
                            output_start["P"] = base_up[0]*spatial_factor["P"]
                            output_end["P"] = output_start["P"] + spatial_factor["P"]
                            
                        else:
                            input_start ["P"] = base_up[0]*spatial_factor["P"]*base_up[1]
                            input_end ["P"] =  input_start["P"] + spatial_factor["P"]+spatial_factor["R"]-1
                            output_start["P"] =base_up[0]*spatial_factor["P"]*base_up[1]
                            output_end["P"] = output_start["P"]+ spatial_factor["P"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"M",0)
                        next_factor = self.find_next_factor(factor,positions,"M")

                        output_start["M"] = base_up[0]*spatial_factor["M"]*next_factor[0]
                        output_end["M"] = output_start["M"] + spatial_factor["M"]

                        weight_start["M"] = base_up[0]*spatial_factor["M"]*next_factor[0]
                        weight_end["M"] = weight_start["M"] + spatial_factor["M"]

                        if(factor[positions][1] in imperfect_factor_dict):
                                for i in range(len(imperfect_factor_dict[factor[positions][1]])):
                                    last_iteration &=  tiling_factor_value[imperfect_factor_dict[factor[positions][1]][i]][1] == factor[imperfect_factor_dict[factor[positions][1]][i]][2]-1
                        else : last_iteration = False
                        input_start ["C"] = spatial_factor["C"]*tiling_factor_value[positions][1]
                        weight_start ["C"] = spatial_factor["C"]*tiling_factor_value[positions][1]
                        if(last_iteration):
                                input_end ["C"] = workload["C"]
                                weight_end ["C"] = workload["C"]
                        else:
                                input_end ["C"] =  input_start["C"] + spatial_factor["C"]
                                weight_end ["C"] = weight_end["C"]+ spatial_factor["C"]
                        
                    elif(factor[positions][1] == "M"):  
                        next_factor = self.find_next_factor(factor,positions,"Q")
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",0)
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",1)
                        if(len(base_up)==1):
                            input_start ["Q"] =  base_up[0]*spatial_factor["Q"]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1
                            output_start["Q"] =base_up[0]*spatial_factor["Q"]
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]
                        else:
                            input_start ["Q"] = base_up[0]*spatial_factor["Q"]*base_up[1]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1
                            output_start["Q"] =base_up[0]*spatial_factor["Q"]*base_up[1]
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"P",0)
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"P",1)
                        next_factor = self.find_next_factor(factor,positions,"P")
                        if(len(base_up)==1):
                            input_start ["P"] = base_up[0]*spatial_factor["P"] #init P
                            input_end ["P"] = input_start ["P"]+spatial_factor['R'] + spatial_factor["P"]-1
                            output_start["P"] = base_up[0]*spatial_factor["P"]
                            output_end["P"] = output_start["P"] + spatial_factor["P"]
                            
                        else:
                            input_start ["P"] = base_up[0]*spatial_factor["P"]*base_up[1]
                            input_end ["P"] =  input_start["P"] + spatial_factor["P"]+spatial_factor["R"]-1
                            output_start["P"] =base_up[0]*spatial_factor["P"]*base_up[1]
                            output_end["P"] = output_start["P"]+ spatial_factor["P"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"C",0)
                        next_factor = self.find_next_factor(factor,positions,"C")
                        input_start ["C"] = base_up[0]*spatial_factor["C"]*next_factor[0]  #init M
                        input_end ["C"] = input_start ["C"]+spatial_factor['C'] 

                        weight_start ["C"] = base_up[0]*spatial_factor["C"]*next_factor[0]  #init M
                        weight_end ["C"] = weight_start ["C"]+spatial_factor['C']

                        output_start ["M"] = spatial_factor["M"]*tiling_factor_value[positions][1]
                        output_end ["M"] = output_start["M"]+spatial_factor["M"]    
                        
                        weight_start ["M"] = spatial_factor["M"]*tiling_factor_value[positions][1]
                        weight_end ["M"] = weight_start["M"]+spatial_factor["M"]  
                elif num==5:   
                    tiling_factor_value[positions][1] +=1
                    tiling_factor_value[positions+1][1]=0
                    tiling_factor_value[positions+2][1] =0
                    tiling_factor_value[positions+3][1] =0  
                    tiling_factor_value[positions+4][1] =0  

                    if(factor[positions][1] == "P"):
                        next_factor = self.find_next_factor(factor,positions,"P",0)
                        input_start["P"] = next_factor[0] *spatial_factor["P"]*tiling_factor_value[positions][1]
                        #input_end ["P"] = input_start["P"] + spatial_factor["P"]+ spatial_factor['R'] -1
                        output_start["P"] = next_factor[0] *spatial_factor["P"]*tiling_factor_value[positions][1]
                       # output_end["P"] = output_start["P"] + spatial_factor["P"]
                        for i in range(len(imperfect_factor_dict[factor[positions][0][1]])):
                            last_iteration &=  tiling_factor_value[imperfect_factor_dict[factor[positions][0][1]][i]][1] == factor[imperfect_factor_dict[factor[positions][0][1]][i]][0][2]-1
                        if(last_iteration):
                            output_end["P"]=workload["P"]
                            input_end ["P"] = workload["P"]+spatial_factor["R"]-1
                        else:
                            output_end["P"] = output_start["P"]+ spatial_factor["P"]
                            input_end ["P"] =  input_start["P"] + spatial_factor["P"]+spatial_factor["R"]-1
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"P",1)
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"P",0)
                        next_factor = self.find_next_factor(factor,positions,"Q")

                        input_start["Q"] = next_factor[0]*spatial_factor["Q"]*base_up[0] + base_down[0]*spatial_factor["Q"]
                        input_end["Q"] = input_start["Q"]+ spatial_factor["Q"] + spatial_factor["S"]-1
                        output_start["Q"] = next_factor[0]*spatial_factor["Q"]*base_up[0] +base_down[0]*spatial_factor["Q"]
                        output_end["Q"] =output_start["Q"]+ spatial_factor["Q"]

                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"M",0)
                        output_start["M"] = base_up[0]*spatial_factor["M"]
                        output_end["M"] = output_start["M"] + spatial_factor["M"]
                        weight_start["M"] = base_up[0]*spatial_factor["M"]
                        weight_end["M"] = weight_start["M"] + spatial_factor["M"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"C",0)
                        input_start ["C"] = base_up[0]*spatial_factor["C"]
                        input_end ["C"] = input_start["C"]+spatial_factor["C"]
                        weight_start["C"] = base_up[0]*spatial_factor["C"]
                        weight_end["C"] = weight_start["C"] + spatial_factor["C"]
                    elif(factor[positions][1] == "Q"):
                        next_factor = self.find_next_factor(factor,positions,"Q",0)

                        input_start["Q"] = next_factor[0]*spatial_factor["Q"]*tiling_factor_value[positions][1] 
                        #input_end ["Q"] = input_start["Q"] + spatial_factor["Q"]+ spatial_factor['S'] -1
                        output_start["Q"] = next_factor[0]*spatial_factor["Q"]*tiling_factor_value[positions][1]
                        #output_end["Q"] = output_start["Q"] + spatial_factor["Q"]
                        for i in range(len(imperfect_factor_dict[factor[positions][0][1]])):
                            last_iteration &=  tiling_factor_value[imperfect_factor_dict[factor[positions][0][1]][i]][1] == factor[imperfect_factor_dict[factor[positions][0][1]][i]][0][2]-1
                        if(last_iteration):
                            output_end["Q"]=workload["Q"]
                            input_end ["Q"] = workload["Q"]+spatial_factor["S"]-1
                        else:
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"P",1)
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"P",0)
                        next_factor = self.find_next_factor(factor,positions,"P",0)
                        input_start["P"] =next_factor[0]*spatial_factor["P"]*base_up[0] + base_down[0] *spatial_factor["P"]
                        input_end["P"] = input_start["P"]+ spatial_factor["P"] + spatial_factor["R"]-1
                        output_start["P"] = next_factor[0]*spatial_factor["P"]*base_up[0] + base_down[0]*spatial_factor["P"]
                        output_end["P"] =output_start["P"]+ spatial_factor["P"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"M",0)
                        output_start["M"] = base_up[0]*spatial_factor["M"]
                        output_end["M"] = output_start["M"] + spatial_factor["M"]
                        weight_start["M"] = base_up[0]*spatial_factor["M"]
                        weight_end["M"] = weight_start["M"] + spatial_factor["M"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"C",0)
                        input_start ["C"] = base_up[0]*spatial_factor["C"]
                        input_end ["C"] = input_start["C"]+spatial_factor["C"]
                        weight_start["C"] = base_up[0]*spatial_factor["C"]
                        weight_end["C"] = weight_start["C"] + spatial_factor["C"]
                elif num==6:   
                    tiling_factor_value[positions][1] +=1
                    tiling_factor_value[positions+1][1] =0
                    tiling_factor_value[positions+2][1] =0
                    tiling_factor_value[positions+3][1] =0  
                    tiling_factor_value[positions+4][1] =0 
                    tiling_factor_value[positions+5][1] =0
                    if(factor[positions][1] == "P"):
                        next_factor = self.find_next_factor(factor,positions,"P",0)
                        input_start["P"] = next_factor[0] *spatial_factor["P"]*tiling_factor_value[positions][1]
                        #input_end ["P"] = input_start["P"] + spatial_factor["P"]+ spatial_factor['R'] -1
                        output_start["P"] = next_factor[0] *spatial_factor["P"]*tiling_factor_value[positions][1]
                        #output_end["P"] = output_start["P"] + spatial_factor["P"]
                        for i in range(len(imperfect_factor_dict[factor[positions][0][1]])):
                            last_iteration &=  tiling_factor_value[imperfect_factor_dict[factor[positions][0][1]][i]][1] == factor[imperfect_factor_dict[factor[positions][0][1]][i]][0][2]-1
                        if(last_iteration):
                            output_end["P"]=workload["P"]
                            input_end ["P"] = workload["P"]+spatial_factor["R"]-1
                        else:
                            output_end["P"] = output_start["P"]+ spatial_factor["P"]
                            input_end ["P"] =  input_start["P"] + spatial_factor["P"]+spatial_factor["R"]-1
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",1)
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"Q",0)
                        next_factor = self.find_next_factor(factor,positions,"Q")

                        input_start["Q"] = next_factor[0]*spatial_factor["Q"]*base_up[0] + base_down[0]*spatial_factor["Q"]
                        input_end["Q"] = input_start["Q"]+ spatial_factor["Q"] + spatial_factor["S"]-1
                        output_start["Q"] = next_factor[0]*spatial_factor["Q"]*base_up[0] +base_down[0]*spatial_factor["Q"]
                        output_end["Q"] =output_start["Q"]+ spatial_factor["Q"]

                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"M",0)
                        output_start["M"] = base_up[0]*spatial_factor["M"]
                        output_end["M"] = output_start["M"] + spatial_factor["M"]
                        weight_start["M"] = base_up[0]*spatial_factor["M"]
                        weight_end["M"] = weight_start["M"] + spatial_factor["M"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"C",0)
                        input_start ["C"] = base_up[0]*spatial_factor["C"]
                        input_end ["C"] = input_start["C"]+spatial_factor["C"]
                        weight_start["C"] = base_up[0]*spatial_factor["C"]
                        weight_end["C"] = weight_start["C"] + spatial_factor["C"]
                    elif(factor[positions][1] == "Q"):
                        next_factor = self.find_next_factor(factor,positions,"Q",0)

                        input_start["Q"] = next_factor[0]*spatial_factor["Q"]*tiling_factor_value[positions][1] 
                        #input_end ["Q"] = input_start["Q"] + spatial_factor["Q"]+ spatial_factor['S'] -1
                        output_start["Q"] = next_factor[0]*spatial_factor["Q"]*tiling_factor_value[positions][1]
                       # output_end["Q"] = output_start["Q"] + spatial_factor["Q"]
                        for i in range(len(imperfect_factor_dict[factor[positions][0][1]])):
                            last_iteration &=  tiling_factor_value[imperfect_factor_dict[factor[positions][0][1]][i]][1] == factor[imperfect_factor_dict[factor[positions][0][1]][i]][0][2]-1
                        if(last_iteration):
                            output_end["Q"]=workload["Q"]
                            input_end ["Q"] = workload["Q"]+spatial_factor["S"]-1
                        else:
                            output_end["Q"] = output_start["Q"]+ spatial_factor["Q"]
                            input_end ["Q"] =  input_start["Q"] + spatial_factor["Q"]+spatial_factor["S"]-1
                        base_down = self.find_tiling_factor_base(tiling_factor_value,positions,"P",1)
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"P",0)
                        next_factor = self.find_next_factor(factor,positions,"P",0)
                        input_start["P"] =next_factor[0]*spatial_factor["P"]*base_up[0] + base_down[0] *spatial_factor["P"]
                        input_end["P"] = input_start["P"]+ spatial_factor["P"] + spatial_factor["R"]-1
                        output_start["P"] = next_factor[0]*spatial_factor["P"]*base_up[0] + base_down[0]*spatial_factor["P"]
                        output_end["P"] =output_start["P"]+ spatial_factor["P"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"M",0)
                        output_start["M"] = base_up[0]*spatial_factor["M"]
                        output_end["M"] = output_start["M"] + spatial_factor["M"]
                        weight_start["M"] = base_up[0]*spatial_factor["M"]
                        weight_end["M"] = weight_start["M"] + spatial_factor["M"]
                        
                        base_up = self.find_tiling_factor_base(tiling_factor_value,positions,"C",0)
                        input_start ["C"] = base_up[0]*spatial_factor["C"]
                        input_end ["C"] = input_start["C"]+spatial_factor["C"]
                        weight_start["C"] = base_up[0]*spatial_factor["C"]
                        weight_end["C"] = weight_start["C"] + spatial_factor["C"]
                
                if(psum_flag):   
                    if (psum_index[0] <= positions):
                        if tiling_factor_value[psum_index[0]][1] == factor_[0][psum_index[0]][2]-1:
                            output_psum = False
            
            output_start_list = [value for value in output_start.values()]
            output_end_list =   [value for value in output_end.values()]
            input_start_list = [value for value in input_start.values()]
            input_end_list =   [value for value in input_end.values()]
            weight_start_list = [value for value in weight_start.values()]
            weight_end_list =   [value for value in weight_end.values()]
            
            formatted_str_output = f"Outputs: {{ [{','.join(map(str, output_start_list))}:{','.join(map(str, output_end_list))}] }}"
            formatted_str_input = f"Inputs: {{ [{','.join(map(str, input_start_list))}:{','.join(map(str, input_end_list))}] }}"
            formatted_str_weight = f"Weights: {{ [{','.join(map(str, weight_start_list))}:{','.join(map(str, weight_end_list))}] }}"
            
            t += self.time_scale
            if psum_flag and output_psum:
                formatted_str = f"Time = {t+start_time} {formatted_str_weight} {formatted_str_input} {formatted_str_output} psum\n"
            else:
                formatted_str = f"Time = {t+start_time} {formatted_str_weight} {formatted_str_input} {formatted_str_output}\n"
            new_data.append(formatted_str)
            
        return new_data, t + start_time

    def parse_dataspace(self, path, workload, fc=0, start_time=0):
        """
        Parses the dataspace and calculates time scale and compute bounds.
        """
        key = []
        result = self.parse_and_sort(key)
        factor = self.get_temporal_factors(path)

        factor_ignore = []
        new_factor = []
        for i in factor:
            if i[0] >= self.hardware_level:
                factor_ignore.append(i)
            else:
                new_factor.append(i)
                
        factor = new_factor
        spatial_factor = self.get_spatial_factors(path)
        self.time_scale = 1
        compute_num = 1
        
        for f in factor_ignore:
            if f[1] != 'X':
                spatial_factor[f[1]] *= f[2]
            else:
                compute_num *= f[2]
            self.time_scale *= f[2]
        
        self.factor_ignore = factor_ignore
        for i in spatial_factor.values():
            compute_num *= i
        
        new_data, last_time = self.parse_stride(result, factor, spatial_factor, fc, workload, start_time)
        
        # Write parsed output back
        with open(os.path.join(path, "parsed_dataspace.txt"), 'w') as f:
            for lines in new_data:
                f.writelines(lines)
                
        return compute_num, self.time_scale, factor, last_time, new_data
        
    def cal_dataspace_total_size(self, line):
        output_dataspace = self.parse_tensor_range(line, "Outputs")
        input_dataspace = self.parse_tensor_range(line, "Inputs")

        output_size = 1
        input_size = 1
        for output in output_dataspace["end"]:
            output_size *= output
        for input in input_dataspace["end"]:
            input_size *= input
        return input_size, output_size
    
    def actual_time_analysis(self, pre_time, time, start_time, timestamp, dataspace, time_scale):
        pattern = r'pre timstamp:(\d+(?:\.\d+)?) cur timestamp:(\d+(?:\.\d+)?)'
        t = start_time
        max_pre = 0 
        new_timestamp = []
        bubble = {}
        new_start_time = start_time
        
        for i in range(len(timestamp)):
            match = re.search(pattern, timestamp[i])
            if not match: continue
            
            pre_timestamp = int(match.group(1))
            cur_timestamp = int(match.group(2))
            
            if pre_timestamp > max_pre:
                max_pre = pre_timestamp
                
            t += 1
            ready = False
            while True:
                if pre_timestamp < t:
                    ready = True
                    t += time_scale
                    new_start_time = t
                    break
                else:
                    bubble[new_start_time] = bubble.get(new_start_time, 0) + 1
                    t += 1
            if ready:
                new_timestamp.append(t)
                
        return t, new_timestamp, 0, bubble, (new_timestamp[0]-time_scale, new_timestamp[-1]) if new_timestamp else (0, 0)

    def pipeline_analysis(self, time_scale, maxpool, fc, pre_data_space_path=None, data_space_path=None, weight_dataspace_path=None, transformer=False, attention=False, transpose=False, Input_dataspace=[], Output_dataspace=[], Weight_dataspace=[], head=1, output_projetion=False, shortcut=0):
        """
        Calculates pipeline overlaps and dependencies between two layers to resolve bubble metrics.
        """
        pre_layer_dataspace = []
        cur_dataspace = []
        
        if Input_dataspace:
            pre_layer_dataspace = Input_dataspace
        elif pre_data_space_path:
            with open(os.path.join(pre_data_space_path, "parsed_dataspace.txt"), 'r') as f:
                pre_layer_dataspace = f.readlines()
                
        if Output_dataspace:
            cur_dataspace = Output_dataspace
        elif data_space_path:
            with open(os.path.join(data_space_path, "parsed_dataspace.txt"), 'r') as f:
                cur_dataspace = f.readlines()

        if attention:
            weight_dataspace = Weight_dataspace
            if not weight_dataspace and weight_dataspace_path:
                with open(os.path.join(weight_dataspace_path, "parsed_dataspace.txt"), 'r') as f:
                    weight_dataspace = f.readlines()

        timestamp_set = []
        new_dataspace = []
        start_time = 0
        cur_start_time = self.get_timestamp(cur_dataspace[0]) if cur_dataspace else 0
        first_match = False
        num = 0

        for idx, dataspace in enumerate(cur_dataspace):
            Input = self.parse_tensor_range(dataspace, "Inputs")
            Weight = self.parse_tensor_range(dataspace, "Weights")
            if not Input: continue
            
            Input["end"][2], Input["end"][3] = Input["end"][3], Input["end"][2]
            

            if transformer:
                if attention:
                    timestamp_output = 0
                    timestamp_weight = 0 
                    
                    for i, pre_dataspace in enumerate(weight_dataspace):
                        Pre_Output  = self.parse_tensor_range(pre_dataspace,"Outputs")
                        if transpose:
                            Weight_match = Weight['end'][0]<=Pre_Output['end'][3] and  Weight['end'][1]*head<=Pre_Output['end'][1]
                        else:
                            Weight_match = Weight['end'][0]*head<=Pre_Output['end'][1] and  Weight['end'][1]<=Pre_Output['end'][3]
                        if('psum'in pre_dataspace):
                            continue
                        if i + 1 < len(pre_layer_dataspace):
                            next_pre_dataspace = pre_layer_dataspace[i + 1]
                            next_Output = self.parse_tensor_range(next_pre_dataspace, "Outputs")
                            partial_result = all (x2 == x1 for x1, x2 in zip(next_Output["end"], Pre_Output["end"]))
                            if(partial_result):
                                continue                     
                        if( Weight_match):
                                num+=1
                                timestamp_weight = self.get_timestamp(pre_dataspace)
                                break
                    num = 0
                    for i, pre_dataspace in enumerate(pre_layer_dataspace):
                        Pre_Output  = self.parse_tensor_range(pre_dataspace,"Outputs")
                        Input_match = Input['end'][1]*head<=Pre_Output['end'][1] and  Input['end'][3]<=Pre_Output['end'][3]
                        if('psum'in pre_dataspace):
                            continue
                        if i + 1 < len(pre_layer_dataspace):
                            next_pre_dataspace = pre_layer_dataspace[i + 1]
                            next_Output = self.parse_tensor_range(next_pre_dataspace, "Outputs")
                            partial_result = all (x2 == x1 for x1, x2 in zip(next_Output["end"], Pre_Output["end"]))
                            if(partial_result):
                                continue
                        
                        if( Input_match ):
                                num+=1
                                timestamp_output = self.get_timestamp(pre_dataspace)
                                break
                    
                    if(Weight_match and Input_match):
                        timestamp_set.append(f"pre timstamp:{max(timestamp_output,timestamp_weight)} cur timestamp:{self.get_timestamp(dataspace)}")
                        new_dataspace.append(dataspace)   
                        if first_match == False:
                            first_match = True
                            start_time = max(timestamp_output,timestamp_weight)
                            
                else:
                    for i, pre_dataspace in enumerate(pre_layer_dataspace):
                        Pre_Output  = self.parse_tensor_range(pre_dataspace,"Outputs")
                        end_result = all (x2 >= x1 for x1, x2 in zip(Input["end"], Pre_Output["end"]))
                        if output_projetion:
                            end_result = Input['end'][1]*head<=Pre_Output['end'][1]*self.head_num and  Input['end'][3]<=Pre_Output['end'][3]
                        if('psum'in pre_dataspace):
                            continue
                        if i + 1 < len(pre_layer_dataspace):
                            next_pre_dataspace = pre_layer_dataspace[i + 1]
                            next_Output = self.parse_tensor_range(next_pre_dataspace, "Outputs")
                            partial_result = all (x2 == x1 for x1, x2 in zip(next_Output["end"], Pre_Output["end"]))
                            if(partial_result):
                                continue
                        
                        if( end_result ):
                                if num ==0: start_time = self.get_timestamp(pre_dataspace)
                                num+=1
                                new_dataspace.append(dataspace)
                                timestamp_set.append(f"pre timstamp:{self.get_timestamp(pre_dataspace)} cur timestamp:{self.get_timestamp(dataspace)}")
                                break
            else:
                if fc == 0:
                    if(maxpool):
                            Input["end"][2] = (Input["end"][2] -2)*2 
                            Input["end"][3] = (Input["end"][3] -2)*2 

                    for pre_dataspace in pre_layer_dataspace:
                        Pre_Output  = self.parse_tensor_range(pre_dataspace,"Outputs")
                        if maxpool == 0:
                            if shortcut ==0:
                                Pre_Output["end"][2]+= 2
                                Pre_Output["end"][3]+= 2

                        if('psum'in pre_dataspace):
                            continue

                        end_result = all (x2 >= x1 for x1, x2 in zip(Input["end"], Pre_Output["end"]))
                        if( end_result):
                            if num ==0: start_time = self.get_timestamp(pre_dataspace)
                            num+=1
                            new_dataspace.append(dataspace)
                            timestamp_set.append(f"pre timstamp:{self.get_timestamp(pre_dataspace)} cur timestamp:{self.get_timestamp(dataspace)}")
                            break
                else :
                    if(maxpool):
                        for pre_dataspace in pre_layer_dataspace:
                            Pre_Output  = self.parse_tensor_range(pre_dataspace,"Outputs")
                            if('psum'in pre_dataspace):
                                continue
                            if(self.DNN == "resnet18"):
                                Flatten_output = Pre_Output["end"][1]*(Pre_Output["end"][2]//maxpool)*(Pre_Output["end"][3]//maxpool )
                                end_result = Flatten_output >= Input["end"][1]
                                if( end_result):
                                    if num ==0: start_time = self.get_timestamp(pre_dataspace)
                                    num+=1
                                    new_dataspace.append(dataspace)
                                    timestamp_set.append(f"pre timstamp:{self.get_timestamp(pre_dataspace)} cur timestamp:{self.get_timestamp(dataspace)}")
                                    break
                            else:
                                Flatten_output = Pre_Output["end"][1]*(Pre_Output["end"][2]//maxpool)*(Pre_Output["end"][3]//maxpool )
                                end_result = Flatten_output >= Input["end"][1]
                                if( end_result):
                                    if num ==0: start_time = self.get_timestamp(pre_dataspace)
                                    num+=1
                                    new_dataspace.append(dataspace)
                                    timestamp_set.append(f"pre timstamp:{self.get_timestamp(pre_dataspace)} cur timestamp:{self.get_timestamp(dataspace)}")
                                    break
                    else:
                        pre_dataspace = pre_layer_dataspace[-1]
                        Pre_Output  = self.parse_tensor_range(pre_dataspace,"Outputs")
                        end_result = Pre_Output["end"][1] >= Input["end"][1]
                        start_time = self.get_timestamp(pre_dataspace)
                        num+=1
                        new_dataspace.append(dataspace)
                        timestamp_set.append(f"pre timstamp:{self.get_timestamp(pre_dataspace)} cur timestamp:{self.get_timestamp(dataspace)}")

        # Analyze actual runtime considering dependencies
        if timestamp_set:
            timestamp_set.sort(key=lambda x: int(x.split('pre timstamp:')[1].split(' ')[0]))
            sorted_cur_timestamps = [int(x.split('cur timestamp:')[1]) for x in timestamp_set]
            
            timestamp_to_dataspace = {}
            for line in new_dataspace:
                start_idx = line.find("Time = ") + len("Time = ")
                end_idx = line.find(" ", start_idx)
                if end_idx == -1: end_idx = len(line)
                ts = int(float(line[start_idx:end_idx]))
                timestamp_to_dataspace[ts] = line

            sorted_dataspace = [timestamp_to_dataspace[ts] for ts in sorted_cur_timestamps if ts in timestamp_to_dataspace]
            start_time = max(cur_start_time, start_time)
            
            actual_time, new_timestamp, time_stride, bubble, cal_time = self.actual_time_analysis(
                len(pre_layer_dataspace), len(cur_dataspace), start_time, timestamp_set, sorted_dataspace, time_scale
            )
            
            # Update timestamps in memory and overwrite parsed files if path provided
            new_start_time = new_timestamp[0] if new_timestamp else 0
            for i in range(len(new_timestamp)):
                if new_dataspace[i].startswith('Time ='):
                    equal_index = new_dataspace[i].find('=')
                    space_index = new_dataspace[i].find(' ', equal_index)
                    if space_index == -1:
                        new_dataspace[i] = f"Time = {new_timestamp[i]}\n"
                    else:
                        remaining = new_dataspace[i][space_index:]
                        new_dataspace[i] = f"Time = {new_timestamp[i]}{remaining}"

            if data_space_path:
                with open(os.path.join(data_space_path, "parsed_dataspace.txt"), 'w') as file:
                    file.writelines(new_dataspace)

            return actual_time, len(pre_layer_dataspace), len(cur_dataspace), start_time, time_stride, bubble, cal_time, new_start_time, new_dataspace
        return 0, 0, 0, 0, 0, {}, (0, 0), 0, []

    def Datamovement_analysis(self, pre_data_space_path, data_space_path):
        """Analyzes fusion opportunities and actual data movement."""
        pre_layer_dataspace = []
        cur_dataspace = []

        with open(pre_data_space_path+"/parsed_dataspace.txt",'r')as f:
            for lines in f:
                pre_layer_dataspace.append(lines)
        with open(data_space_path+"/parsed_dataspace.txt",'r')as f:
            for lines in f:
                cur_dataspace.append(lines)
  
        cur_dataspace_indx = 0
        reuse_data = []
        psum_access = 0
        psum_output = []
        reuse_Input = []
        no_reuse_Output = 0
        for i,pre_dataspace in enumerate(pre_layer_dataspace):
            Output = self.parse_tensor_range(pre_dataspace,"Outputs")
            
            next_input_req = self.parse_tensor_range(cur_dataspace[cur_dataspace_indx],"Inputs")
            if('psum'in pre_dataspace):
                next_Output = self.parse_tensor_range(pre_layer_dataspace[i+1],"Outputs")
                if next_Output == Output:continue
                elif('psum' in pre_layer_dataspace[i+1]):
                    #psum_access+=math.prod(Output['shape'])
                    psum_output.append(Output)
                    continue
            if(next_input_req['end'][1]<=Output['end'][1] and next_input_req['end'][2]<=Output['end'][3] 
               and next_input_req['start'][1]>=Output['start'][1] and next_input_req['start'][2]>=Output['start'][3] ):
                if(Output not in psum_output):
                    reuse_data.append([next_input_req,Output])
                    reuse_Input.append(next_input_req)
                else:
                    reuse_data.append([next_input_req,0])
                    reuse_Input.append(next_input_req)
                cur_dataspace_indx +=1
                while(self.parse_tensor_range(cur_dataspace[cur_dataspace_indx],"Inputs") == next_input_req):
                    cur_dataspace_indx +=1
                    if(cur_dataspace_indx == len(cur_dataspace)):
                        break
        while (cur_dataspace_indx != len(cur_dataspace)):
                if(self.parse_tensor_range(cur_dataspace[cur_dataspace_indx],"Inputs") in reuse_Input):
                    no_reuse_Output += math.prod(self.parse_tensor_range(cur_dataspace[cur_dataspace_indx],"Inputs")['shape'])
                cur_dataspace_indx +=1
        Input_fusion_reuse = 0
        Output_fusion_reuse = 0
        for x in reuse_data :
            Input_fusion_reuse += math.prod(x[0]['shape'])
            if x[1] ==0 :continue
            Output_fusion_reuse += math.prod(x[1]['shape'])*2
        if Output_fusion_reuse >=no_reuse_Output:
            Output_fusion_reuse -=no_reuse_Output*2
        print("Input reuse with fusion = ",Input_fusion_reuse," Output reuse with fusion = ",Output_fusion_reuse,psum_access)
        #pre_access=plot.extract_component_scalar_reads(pre_data_space_path+"/timeloop-mapper.stats.txt","dummy_top")
        #cur_access=plot.extract_component_scalar_reads(data_space_path+"/timeloop-mapper.stats.txt","dummy_top")
        Original_access = pre_access.copy()
        Original_access['Weights']+=cur_access['Weights']
        Original_access['Inputs']+=cur_access['Inputs']
        Original_access['Outputs']+=cur_access['Outputs']
        Fusion_access = pre_access.copy()
        Fusion_access['Outputs'] +=cur_access['Outputs']-Output_fusion_reuse
        Fusion_access['Inputs'] +=cur_access['Inputs']-Input_fusion_reuse
        Fusion_access['Weights'] += cur_access['Weights']
        #total_access = pre_access+cur_access-Input_fusion_reuse-Output_fusion_reuse
        print("Originnal Reuse = ",sum(Original_access.values())," Fusion Reuse = ",sum(Fusion_access.values()))
        return sum(Original_access.values()),sum(Fusion_access.values()),Original_access,Fusion_access


    def get_max_group_len(self, start_layer=0):
        """Calculates maximum sequence block grouping based on available hardware capacity."""
        workload = []
        layers = self.layers[start_layer:]
        for layer in self.layers:
            workload.append(self.analyzer.get_workload(layer))
            
        arch_size = self.hw.get('tile_num') * self.hw.get('macro_num') * self.hw.get('core_num') * \
                    self.hw.get('array_col') * self.hw.get('array_row') * self.hw.get('cim_depth')
                    
        workload_group = []
        workload_sub_group = []
        
        for i, layer in enumerate(layers):
            idx = self.layers.index(layer)
            if 'R' in workload[idx].keys() or 'S' in workload[idx].keys():
                workload_size = workload[idx]['C'] * workload[idx]['M'] * workload[idx]['R'] * workload[idx]['S'] * self.precision                 
            else:
                workload_size = workload[idx]['C'] * workload[idx]['M'] * self.precision
                
            if workload_size <= arch_size:
                arch_size -= workload_size
                workload_sub_group.append(layer)
            else: 
                if workload_sub_group:
                    workload_group.append(workload_sub_group)
                    workload_sub_group = []
                arch_size = self.hw.get('tile_num') * self.hw.get('macro_num') * self.hw.get('core_num') * \
                            self.hw.get('array_col') * self.hw.get('array_row') * self.hw.get('cim_depth')
                            
                if workload_size <= arch_size:
                    arch_size -= workload_size
                    workload_sub_group.append(layer)
                else:
                    workload_group.append([layer])

            if idx == len(self.layers) - 1:
                if workload_sub_group:
                    workload_group.append(workload_sub_group)
                else:
                    workload_group.append([layer])

        max_len = max([len(item) for item in workload_group]) if workload_group else 0
        return max_len
