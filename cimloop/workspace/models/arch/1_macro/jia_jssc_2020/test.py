from _tests import scripts
from scripts.notebook_utils import *
from scripts import utils as utl
import os
THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_NAME = os.path.basename(THIS_SCRIPT_DIR)


results = utl.quick_run(macro=MACRO_NAME,dnn="resnet18",layer="09",system="ws_dummy_buffer_many_macro")
print("各元件能量",results.per_component_energy)
print("per_compute_energy",results.per_compute('per_component_energy'))
print("能量",results.energy)
print("周期数",results.cycles)
print("吞吐率",results.tops_1b)
print("TOPS/W",results.tops_per_w)
