from _tests import scripts
from scripts.notebook_utils import *

result = run_test("jia_jssc_2020", "test_tops")

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
bar_side_by_side(
    {r.variables["VOLTAGE"]: r.tops_1b for r in result},
    xlabel="Voltage (V)",
    ylabel="Throughput (1b TOPS)",
    title="Voltage vs. Throughput",
    ax=ax[0],
)
bar_side_by_side(
    {r.variables["VOLTAGE"]: r.tops_per_w_1b for r in result},
    xlabel="Voltage (V)",
    ylabel="Energy Efficiency (1b TOPS/W)",
    title="Voltage vs. Energy Efficiency",
    ax=ax[1],
)
bar_side_by_side(
    {r.variables["VOLTAGE"]: r.tops_per_mm2_1b for r in result},
    xlabel="Voltage (V)",
    ylabel="Compute DensThity (1b TOPS/mm^2)",
    title="Voltage vs. Compute Density",
    ax=ax[2],
)
plt.show()
