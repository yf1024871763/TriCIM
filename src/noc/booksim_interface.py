import os
import re
import tempfile
import subprocess
import time


class BookSimInterface:
    def __init__(self, booksim_binary_path: str):
        self.booksim_binary = booksim_binary_path
        self.cache = {}

    def run_simulation(
        self, mesh_dim: int, traffic_pattern: str, injection_rate: float
    ) -> dict:

        STANDARD_PACKET_SIZE = 16

        cache_key = (mesh_dim, traffic_pattern, injection_rate)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # print(f"[BookSim] Profiling Base NoC Latency: Pattern={traffic_pattern}, InjRate={injection_rate}...")

        config_content = f"""
topology = mesh;
k = {mesh_dim}; n = 2;
routing_function = dor; 
traffic = {traffic_pattern};
packet_size = {STANDARD_PACKET_SIZE}; 
injection_rate = {injection_rate};
sim_type = latency;
"""
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".cfg"
        ) as tmp_cfg:
            tmp_cfg.write(config_content)
            config_path = tmp_cfg.name

        try:
            result = subprocess.run(
                [self.booksim_binary, config_path],
                capture_output=True,
                text=True,
                check=True,
            )
            output = result.stdout

            if "Deadlock" in output or "Error" in output:
                print("[BookSim Warning] Deadlock detected in simulation.")

            lat_match = re.search(r"Packet latency average\s*=\s*([0-9.]+)", output)
            base_latency = float(lat_match.group(1)) if lat_match else 0.0

            hops_match = re.search(r"Hops average\s*=\s*([0-9.]+)", output)
            avg_hops = float(hops_match.group(1)) if hops_match else 0.0

            res = {"base_latency": base_latency, "avg_hops": avg_hops}
            self.cache[cache_key] = res
            return res

        except subprocess.CalledProcessError as e:
            print(f"[BookSim Error] {e.stderr}")
            return {"base_latency": 0.0, "avg_hops": 0.0}
        finally:
            os.remove(config_path)
