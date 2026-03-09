import re
import os
import math
import yaml
import logging


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class analysis:
    """
    Core parser for Timeloop/Cimloop output dataspace and stats.
    """
    def __init__(self, dnn, config=None):
        self.DNN = dnn
        self.config = config or {}
        

        self.paths = self.config.get('paths', {})
        self.workload_dir = os.path.join(self.paths.get('workload_root', './workloads'), self.DNN)
        self.output_dir = self.paths.get('output_root', './outputs')
        self.arch_dir = self.paths.get('arch_root', './arch')
        
        self.tile_num = []
        self.layer_num = 0
        self.weight_num = []
        

        if os.path.exists(self.workload_dir):
            layers = [f.split('.')[0] for f in os.listdir(self.workload_dir) 
                      if f != "index.yaml" and f.endswith(".yaml")]
            self.layers = sorted(layers)
        else:
            logging.warning(f"Workload directory not found: {self.workload_dir}")
            self.layers = []
         
    def cal_tile_num(self):
        self.layer_num = len(self.layers)
        tile_num = [] 
        for layer in self.layers:
            layer_path = os.path.join(self.workload_dir, f"{layer}.yaml")
            with open(layer_path, 'r') as file:
                instance_line = None
                for line in file:
                    if 'instance' in line:
                        instance_line = line
                        break
                if instance_line:
                    instance_line = instance_line.strip()
                    instance_data = instance_line.split(': ', 1)[1]
                    pattern = r'(\w+):\s*(\d+)'
                    matches = re.findall(pattern, instance_data)
                    result = {key: int(value) for key, value in matches}
                    
                    if result.get('R') is not None:
                        requirement_row = result['C'] * result['R'] * result['S']
                        requirement_col = result['M'] * 8
                    else:
                        requirement_row = result['C']
                        requirement_col = result['M'] * 8
                    
                    row = 8 * 128
                    col = 12 * 128
                    num = max(math.ceil(requirement_row/row), math.ceil(requirement_col/col))
                    num = self.next_power_of_two(num)
                    tile_num.append(num)
                    
        self.tile_num = tile_num

    def cal_weight_num(self):
        self.layer_num = len(self.layers)
        self.weight_num = []
        for layer in self.layers:
            result = self.get_workload(layer)
            if result.get('R') is not None:   
                self.weight_num.append(int(result['C'] * result['R'] * result['S'] * result['M'] * 16))
            else:
                self.weight_num.append(int(result['C'] * result['M'] * 16))

    def get_workload(self, layer):
        layer_path = os.path.join(self.workload_dir, f"{layer}.yaml")
        result = {}
        if not os.path.exists(layer_path):
            logging.warning(f"Workload file not found: {layer_path}")
            return result
            
        with open(layer_path, 'r') as file:
            instance_line = None
            for line in file:
                if 'instance' in line:
                    instance_line = line
                    break
            if instance_line:
                instance_line = instance_line.strip()
                instance_data = instance_line.split(': ', 1)[1]
                pattern = r'(\w+):\s*(\d+)'
                matches = re.findall(pattern, instance_data)
                result = {key: int(value) for key, value in matches}  
        return result

    def next_power_of_two(self, n):
        return 1 if n == 0 else 2 ** math.ceil(math.log2(n))

    def modify_arch_yaml(self, file_path, tilenum):
        if not os.path.exists(file_path):
            logging.error(f"Cannot modify yaml, file not found: {file_path}")
            return
            
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if 'spatial: {meshX:' in line:
                indent = line[:line.find('spatial:')]
                lines[i] = f"{indent}spatial: {{meshX: {tilenum}}}\n"
                break

        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)
        logging.debug(f"Modified meshX to {tilenum} in {file_path}")

    def get_total_energy(self, file_path):
        if not os.path.exists(file_path):
            return 0.0
        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(r'Energy:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*uJ', line)
                if match:
                    return float(match.group(1))
        return 0.0

    def cal_energy(self):   
        energy_origin = []
        energy_pipeline = []
        for layer in self.layers:
            pipe_stats = os.path.join(self.output_dir, f"pipeline-isaac-{self.DNN}-{layer}", "timeloop-mapper.stats.txt")
            orig_stats = os.path.join(self.output_dir, f"pipeline_origin-isaac-{self.DNN}-{layer}", "timeloop-mapper.stats.txt")
            energy_pipeline.append(self.get_total_energy(pipe_stats))
            energy_origin.append(self.get_total_energy(orig_stats))
        return energy_pipeline, energy_origin

    def input_output_gen(self, folder_path):
        results = []
        if not os.path.exists(folder_path):
            return results

        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    levels = re.split(r'Level \d+\n[-=]+', content)

                    for level in levels:
                        if "=== dummy_top ===" in level:
                            inputs_match = re.search(r"Inputs:.*?Address generations \(per-cluster\)\s*:\s*([\d.eE+-]+)", level, re.DOTALL)
                            outputs_match = re.search(r"Outputs:.*?Address generations \(per-cluster\)\s*:\s*([\d.eE+-]+)", level, re.DOTALL)
                            weights_match = re.search(r"Weights:.*?Address generations \(per-cluster\)\s*:\s*([\d.eE+-]+)", level, re.DOTALL)
                            
                            results.append({
                                'inputs': float(inputs_match.group(1)) if inputs_match else None,
                                'outputs': float(outputs_match.group(1)) if outputs_match else None,
                                'weights': float(weights_match.group(1)) if weights_match else None
                            })
                            break
        return results

    def cal_input_output(self):
        dataspace = []
        for layer in self.layers:
            folder = os.path.join(self.output_dir, f"pipeline_origin-isaac-{self.DNN}-{layer}")
            dataspace.append(self.input_output_gen(folder))
        return dataspace

    def cal_pipeline_input_output(self):
        if not self.layers: return 0, 0
        first_folder = os.path.join(self.output_dir, f"pipeline-isaac-{self.DNN}-{self.layers[0]}")
        last_folder = os.path.join(self.output_dir, f"pipeline-isaac-{self.DNN}-{self.layers[-1]}")
        first = self.input_output_gen(first_folder)
        last = self.input_output_gen(last_folder)
        
        inputs = first[0]['inputs'] if first else 0
        outputs = last[0]['outputs'] if last else 0
        return inputs, outputs

    def get_energy_by_component(self, file_path, component_name):
        computes = None
        component_energy_per_compute = None
        stats_txt = os.path.join(file_path, "timeloop-mapper.stats.txt")

        if not os.path.exists(stats_txt): return 0.0

        with open(stats_txt, 'r') as file:
            for line in file:
                if "Computes" in line:
                    match = re.search(r"Computes\s*=\s*(\d+)", line)
                    if match: computes = int(match.group(1))
                
                match = re.match(rf"\s*{re.escape(component_name)}\s*=\s*([0-9eE\+\-\.]+)", line)
                if match:
                    component_energy_per_compute = float(match.group(1))

        if computes is None or component_energy_per_compute is None:
            return 0.0

        total_energy_fJ = computes * component_energy_per_compute
        return total_energy_fJ / 10e9  # uJ

    def get_cycle(self, file_path):
        stats_txt = os.path.join(file_path, "timeloop-mapper.stats.txt")
        if not os.path.exists(stats_txt):
            return 0
            
        with open(stats_txt, 'r') as file:
            for line in file:
                match = re.search(r"Cycles:\s*(\d+)", line)
                if match:
                    return int(match.group(1))
        return 0

    def get_utilization(self, file_path):
        stats_txt = os.path.join(file_path, "timeloop-mapper.stats.txt")
        if not os.path.exists(stats_txt):
            return 0.0
            
        with open(stats_txt, 'r') as file:
            for line in file:
                match = re.search(r"Utilization:\s*([0-9.]+)%", line)
                if match:
                    return float(match.group(1))
        return 0.0

    def extract_cim_write_energy(self, file_path):
        yaml_path = os.path.join(file_path, "timeloop-mapper.ERT.yaml")
        try:
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)
            
            if 'ERT' not in data or 'tables' not in data['ERT']:
                return 0.0
            
            for table in data['ERT']['tables']:
                if 'name' in table and table['name'].startswith('system_top_level.cim_unit'):
                    if 'actions' in table:
                        for action in table['actions']:
                            if 'name' in action and action['name'] == 'write' and 'energy' in action:
                                return float(action['energy'])
            return 0.0
        except Exception:
            return 0.0

    def extract_cim_utilized_instances(self, file_path):
        stats_txt = os.path.join(file_path, "timeloop-mapper.stats.txt")
        try:
            with open(stats_txt, 'r') as file:
                lines = file.readlines()
                
            in_cim_section, in_stats_section, in_weights_section = False, False, False
            for line in lines:
                line = line.strip()
                if line == "=== cim_unit ===":
                    in_cim_section = True; in_stats_section = False; in_weights_section = False
                    continue
                if in_cim_section and line.startswith("=== ") and line.endswith(" ==="):
                    in_cim_section = False
                    continue
                if in_cim_section and line == "STATS":
                    in_stats_section = True
                    continue
                if in_stats_section and line == "Weights:":
                    in_weights_section = True
                    continue
                if in_weights_section and line.startswith("Utilized instances (max)"):
                    return int(line.split(":")[1].strip())
            return 0
        except Exception:
            return 0

    def extract_vector_access_by_module(self, file_path, target_op, module_name="cim_unit"):
        stats_txt = os.path.join(file_path, "timeloop-mapper.stats.txt")
        try:
            with open(stats_txt, 'r') as file:
                content = file.read()
            
            module_pattern = re.compile(rf'===\s+{re.escape(module_name)}\s+===(.*?)(?=\n===|\nLevel|\Z)', re.DOTALL | re.IGNORECASE)
            module_match = module_pattern.search(content)
            
            if not module_match: return 0.0
            
            vector_pattern = re.compile(rf'vector access\s+:\s+([\d\.e\+\-]+)\s+op_name:\s*{re.escape(target_op)}', re.IGNORECASE)
            vec_match = vector_pattern.search(module_match.group(1))
            
            if vec_match:
                return float(vec_match.group(1))
            return 0.0
        except Exception:
            return 0.0