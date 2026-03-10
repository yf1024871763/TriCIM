from _tests import scripts
from scripts.notebook_utils import *
import sys
import os
IGNORE = ["system", "macro_in_system", "1bit_x_1bit_mac"]
display_markdown(
    f"""
Model of "A Programmable Heterogeneous Microprocessor Based on Bit-Scalable
In-Memory Computing", JSSC 2020
--------------------------------------------------------------------------------

Paper by Hongyang Jia, Hossein Valavi, Yinqi Tang, Jintao Zhang, and Naveen
Verma

## Description of The Macro

This macro partitions a large SRAM array into a 4x4 grid of subarrays that may
be indepently power gated. Additionally, it connects every group of three
columns to reuse outputs rather than inputs.

Inputs and outputs are both sliced into 1b slices, and the macro may sum
the results of multiple slices to produce a final output using
variable-precision inputs and weights.

{get_important_variables_markdown('jia_jssc_2020')}

### Macro Level

- **Input Path**: Inputs are passed through zero gating before being
  XNOR-encoded and sent to row drivers. Row drivers support 1b input
  slices, so N-bit inputs are processed in N+1 cycles (XNOR adds an extra bit).
- **Weight Path**: Weight drivers are used to rewrite weights in the array.
- **Output Path**: Column drivers activate array columns to read analog outputs,
  which are then converted to digital using an 8b ADC. After the ADC,
  outputs are accumulated in a shift-add that sums output results across
  different input and weight slices. Finally, an output datapath performs
  quantization and activation functions on the outputs.

Next, there are four column groups in the macro, and each column group contains
64 column subgroups. Column groups may be independently power gated. Inputs are
reused between column groups and column subgroups.

### Column Subgroup Level 
 
There are 3 folded columns in a column subgroup. Folded columns, unlike standard
columns, reuse outputs rather than inputs. Beyond this, there are no additional
components in a column subgroup.

### Folded Column Level

- *Input Path*: Each input is passed directly to a row group in the folded
  column.
- *Weight Path*: A column bandwidth limiter sets the read and write bandwidth of
  each array column. Each weight is then passed to a row in the column.
- *Output Path*: A column bandwidth limiter sets the read and write bandwidth of
  each array column. One output is reused between rows in the column.

Inside each folded column, four row groups of 64 rows each reuse outputs. Row
groups may be independently power gated.

### Row Level

Each row contains a CiM unit that is composed of a SRAM cell and a capacitor to
perform analog MAC operations. The CiM unit uses a 1x1x8 (1b input slice x
1b weight slice x 8b output) virtualized MAC unit to compute the MAC
operation.
"""
)
result = run_test("jia_jssc_2020", "test_area_breakdown")
bar_side_by_side(
    result[0].get_compare_ref_area()*1e12,
    xlabel="Component",
    ylabel="Area (um^2)",
    title="Area Breakdown",
)
#display_diagram(get_diagram("jia_jssc_2020", ignore=IGNORE))
