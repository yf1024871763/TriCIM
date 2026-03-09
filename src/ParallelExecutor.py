import concurrent.futures
import os
import shutil
import tempfile
import logging


class ParallelExecutor:
    def __init__(self, layer_num, arch_path, tile_num, layers, DNN, MACRO_NAME):
        self.layer_num = layer_num
        self.arch_path = arch_path
        self.tile_num = tile_num
        self.layers = layers
        self.DNN = DNN
        self.MACRO_NAME = MACRO_NAME
        
        # 创建临时工作目录
        self.temp_dir = tempfile.mkdtemp(prefix="parallel_work_")
        #logging.info(f"Created parallel temp workspace: {self.temp_dir}")
    
    def _worker(self, i):
        """单个任务的执行函数"""
        import logging
        
        # === [新增] 给子进程戴上“消音器”，屏蔽底层 Accelergy/Cimloop 的刷屏 ===
        # 1. 把根日志器调高到 WARNING
        logging.getLogger().setLevel(logging.WARNING)
        import utils as utl
        # 2. 如果底层库偷偷建了独立的名字，把它们全抓出来静音
        for logger_name in logging.root.manager.loggerDict:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        task_arch_path = os.path.join(self.arch_path, f"pipeline_{i}.yaml")
        target_dir = os.path.dirname(task_arch_path)
        os.makedirs(target_dir, exist_ok=True)
        
        base_pipeline = os.path.join(self.arch_path, "pipeline.yaml")
        shutil.copy(base_pipeline, task_arch_path)
        
        try:
            # 修改配置文件
            self.modify_arch_yaml(task_arch_path, self.tile_num[i])
            
            # 执行任务
            result = utl.quick_run(
                chip=f"pipeline_{i}",
                macro=self.MACRO_NAME,
                tile="isaac",
                dnn=self.DNN,
                layer=self.layers[i]     
            )
            
            # 运行完毕后清理副本
            if os.path.exists(task_arch_path):
                os.remove(task_arch_path)
            return True
        
        except Exception as e:
            logging.error(f"Task {i} failed: {str(e)}")
            return None
    
    def modify_arch_yaml(self, file_path, tilenum):
        """动态修改 YAML 配置文件中的 meshX"""
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if 'spatial: {meshX:' in line:
                indent = line[:line.find('spatial:')]
                lines[i] = f"{indent}spatial: {{meshX: {tilenum}}}\n"
                break

        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)

    def run_parallel(self, max_workers=None):
        """并行执行所有任务"""
        if max_workers is None:
            max_workers = min(os.cpu_count(), self.layer_num)

        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._worker, i): i 
                for i in range(self.layer_num)
            }
            
            for future in concurrent.futures.as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    result = future.result()
                    results.append((i, result))
                except Exception as e:
                    logging.error(f"Task {i} exception: {str(e)}")
                    results.append((i, None))
        
        # 按原始顺序排序结果
        results.sort(key=lambda x: x[0])
        return [r for i, r in results]
    
    def cleanup(self):
        """清理临时工作目录"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            #logging.info(f"Cleaned up temp directory: {self.temp_dir}")