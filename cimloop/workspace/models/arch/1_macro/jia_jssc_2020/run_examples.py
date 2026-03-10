import os
import sys
import logging
import timeloopfe.v4 as tl
from timeloopfe.v4.processors import EnableDummyTableProcessor
from _tests import scripts
from scripts.notebook_utils import *
from scripts import utils as utl
THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_NAME = os.path.basename(THIS_SCRIPT_DIR)
sys.path.append(os.path.abspath(os.path.join(THIS_SCRIPT_DIR, '..', '..', '..', '..')))

if __name__ == "__main__":
    results = utl.single_test(utl.quick_run(macro=MACRO_NAME))
    results.combine_per_component_area(["adc"], "ADC")
    results.add_compare_ref_area("ADC", [0.497e-6])

    results.combine_per_component_area(
        [
            "row_drivers",
            "weight_drivers",
            "cim_unit",
            "column_drivers",
            "bitcell_capacitor",
        ],
        "CiM",
    )
    results.add_compare_ref_area("CiM", [2.91e-6])

    results.combine_per_component_area(["out_datapath", "shift_add"], "NMC Data Path")
    results.add_compare_ref_area("NMC Data Path", [0.497e-6])

    results.combine_per_component_area(["input_zero_gating"], "Sparsity Controller")
    results.add_compare_ref_area("Sparsity Controller", [0.392e-6])
    bar_side_by_side(
    results[0].get_compare_ref_area()*1e12,
    xlabel="Component",
    ylabel="Area (um^2)",
    title="Area Breakdown",
    )
