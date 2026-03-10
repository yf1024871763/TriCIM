from _tests import scripts
from scripts.notebook_utils import *
from scripts import utils as utl
import os
THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_NAME = os.path.basename(THIS_SCRIPT_DIR)


results = utl.quick_run(macro=MACRO_NAME,tile="raella",dnn="mobilenet_v3",layer="02")
print("各元件能量",results.per_component_energy)
print("能量",results.energy)
print("周期数",results.cycles)
print("吞吐率",results.tops)
print("TOPS/W",results.tops_per_w)
