from _tests import scripts
from scripts.notebook_utils import *
from scripts import utils as utl
import sys
import os
N_COLUMNS = 256
N_ADC_USES = N_COLUMNS * 1
N_OUTPUTS = N_COLUMNS / 1
def test_energy_breakdown():
    """
    ### Energy Breakdown

    This test replicates the results of Table I in the paper.

    We show the area and energy of the macro at 0.85V and 1.2V power supplies
    using 1b inputs and weights. We will report the energy of the ADC, CiM,
    and NMC data path.

    We see that increasing the voltage from 0.85V to 1.2V increases the energy
    consumption of each component of the macro.
    """
    '''
    results = utl.parallel_test(
        utl.delayed(utl.quick_run)(
            macro="jia_jssc_2020",
            variables=dict(
                VOLTAGE=x,
                INPUT_BITS=1,
                OUTPUT_BITS=1,
                WEIGHT_BITS=1,
            ),
        )
        for x in [0.85, 1.2]
    )'''
    '''
    results = single_test(quick_run(macro="jia_jssc_2020",dnn="resnet18",layer="02"))
    results.add_compare_ref_energy(
        "adc", [2.8707158725328e-06]
    )
    results.add_compare_ref_energy("row_drivers",[2.056853e-05])
    results.add_compare_ref_energy("column_drivers",[4.483643867e-07])
    results.add_compare_ref_energy("cim_unit",[1.3835666718e-07])
    '''

    results = single_test(quick_run(macro="jia_jssc_2020",dnn="resnet18",layer="09"))
    results.add_compare_ref_energy(
        "adc", [1.435357937664e-06]
    )
    results.add_compare_ref_energy("row_drivers",[5.96339431e-06])
    results.add_compare_ref_energy("column_drivers",[2.241821933567e-07])
    results.add_compare_ref_energy("cim_unit",[1.383566671e-07])
    return results

result =  test_energy_breakdown()
fig, ax = plt.subplots(figsize=(15, 5))
bar_side_by_side(
        result[0].get_compare_ref_energy()*1e12,
        xlabel="Component",
        ylabel="Energy (pJ/Full Array 1b MACs)",
        title=f"mapping optimization"
    )
plt.show()
