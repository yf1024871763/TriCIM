import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import re
import matplotlib.patches as mpatches
from matplotlib import gridspec
import seaborn as sns


def plot_bubble(save_path, layers_cal, layers_bubble, labels=None, actual_time=0):
    """
    Visualize computation time and bubble time
    :param layers_cal: List of tuples [(start1, end1), (start2, end2), ...] computation time intervals for each layer
    :param layers_bubble: List of dicts, bubble time intervals for each layer
    :param actual_time: Total system execution time
    """
    # Ensure necessary libraries are imported
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    no_labels = False

    # Data validation
    assert len(layers_cal) == len(
        layers_bubble
    ), "Number of layers in computation time and bubble time must match"
    if labels is None:
        labels = [str(i + 1) for i in range(len(layers_cal))]
        no_labels = True

    # Preprocess data
    processed_data = []
    for cal_times, bubble_dict in zip(layers_cal, layers_bubble):
        # Convert computation time
        cal_intervals = [(cal_times[0], cal_times[1] - cal_times[0])]

        # Convert bubble time
        bubble_intervals = sorted(bubble_dict.items())

        # Merge into the same layer data
        processed_data.append({"cal": cal_intervals, "bubble": bubble_intervals})

    percent = []
    for data in processed_data:
        if data["bubble"] == []:
            percent.append(0)
            continue
        percent.append((sum(y for _, y in data["bubble"]) / data["cal"][0][1]) * 100)

    # Create plot
    fig, ax = plt.subplots(figsize=(20, 10))

    # Define color scheme - using hex codes is more reliable
    CAL_COLOR = "#6495ED"  # Clear blue
    BUBBLE_COLOR = "lightgray"  # Light gray
    BORDER_COLOR = "black"  # Border color

    # Draw layer by layer
    for layer_idx, layer_data in enumerate(reversed(processed_data)):
        y_center = layer_idx + 1  # Position of the layer on the y-axis

        # Plot computation time (with black border) - use color parameter instead of facecolors
        if layer_data["cal"]:
            ax.broken_barh(
                layer_data["cal"],
                (y_center - 0.45, 0.9),
                color=CAL_COLOR,  # Fixed to color parameter
                edgecolor=BORDER_COLOR,
                linewidth=0.5,
                label="Compute time" if layer_idx == 0 else None,
            )

        # Plot bubble time (with black border)
        if layer_data["bubble"]:
            ax.broken_barh(
                layer_data["bubble"],
                (y_center - 0.45, 0.9),
                color=BUBBLE_COLOR,  # Fixed to color parameter
                edgecolor=BORDER_COLOR,
                linewidth=0.2,
                hatch="//",
                label="Bubble time" if layer_idx == 0 else None,
            )

    # Automatically calculate time range
    all_points = []
    for layer in processed_data:
        all_points += [t for t, _ in layer["cal"] + layer["bubble"]]
        all_points += [t + d for t, d in layer["cal"] + layer["bubble"]]
    min_t = min(all_points) if all_points else 0
    max_t = max(all_points) if all_points else actual_time

    # Axis settings
    ax.set_yticks(np.arange(1, len(processed_data) + 1))

    ax.set_yticklabels(reversed(labels), fontsize=20)
    ax.tick_params(axis="x", labelsize=20)
    ax.set_xlim(min_t * 0.95, max(max_t, actual_time) * 1.05)
    ax.set_xlabel("Timestamp", fontsize=20)
    ax.set_title(f"Computation and bubble time (Time Cost: {actual_time})", fontsize=25)

    if no_labels:
        ax.set_yticks([])  # Clear y-axis ticks, do not show any numbers

    # Legend settings - ensure they match the actual colors
    legend_elements = [
        Patch(facecolor=CAL_COLOR, edgecolor=BORDER_COLOR, label="Compute time"),
        Patch(
            facecolor=BUBBLE_COLOR,
            edgecolor=BORDER_COLOR,
            hatch="//",
            label="Bubble time",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=20)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)  # Simplify directory creation
    plt.savefig(f"{save_path}/Bubble.png", dpi=400, bbox_inches="tight")
    plt.savefig(f"{save_path}/Bubble.svg", dpi=600, format="svg")

    # Plot percentage graph
    plt.figure(figsize=(20, 9))
    bars = plt.bar(labels, percent, 0.45, color=CAL_COLOR)  # Use the same blue color
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=15)
    plt.ylabel("Percent (%)", fontsize=20)
    plt.title(f"Bubble percentage with pipeline", fontsize=25)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=14,
        )  # Add percentage symbol

    plt.tight_layout()
    plt.savefig(f"{save_path}/Bubble_percentage.png", dpi=400, bbox_inches="tight")

    plt.show()


def plot_combined_timelines_block_batch(
    save_path,
    layers_cal,
    layers_bubble,
    layers_weight,
    layer_names=None,
    block_labels=None,
    actual_time=0,
):
    """
    Visualize computation time, bubble time, and weight update time for multiple blocks
    The y-axis represents blocks, and within each block, layers are ordered from top to bottom (in correct sequence)

    :param save_path: Image save path
    :param layers_cal: 3D list [block][batch][layer]
                       Computation time intervals for each layer, can be a tuple(start,end) or list[(start,end),...]
    :param layers_bubble: 3D list [block][batch][layer]
                          Bubble time intervals for each layer, can be a tuple(start,end) or list[(start,end),...]
    :param layers_weight: 3D list [block][batch][layer]
                          Weight update time intervals for each layer, can be a tuple(start,duration) or list[(start,duration),...]
    :param layer_names: List of layer names (optional)
    :param block_labels: Labels for each block (optional)
    :param actual_time: Total system execution time (optional)
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    num_blocks = len(layers_cal)
    num_batches = len(layers_cal[0]) if num_blocks > 0 else 0

    if block_labels is None:
        block_labels = [f"Block {i}" for i in range(num_blocks)]

    CAL_COLOR = "#6495ED"  # Blue - Computation time
    BUBBLE_COLOR = "lightgrey"  # Gray - Bubble time
    WEIGHT_COLOR = "#D63344"  # Red - Weight update time
    BORDER_COLOR = "black"

    if layer_names is None:
        default_layer_names = ["Q", "K", "V", "A", "Z0", "Z1", "FFN1", "FFN2"]
    else:
        default_layer_names = layer_names

    all_points = []
    fig, ax = plt.subplots(figsize=(20, 12))

    y_offset = 0
    block_positions = []

    for block_idx in range(num_blocks):
        num_layers_block = max(len(batch) for batch in layers_cal[block_idx])

        if layer_names is None:
            cur_layer_names = default_layer_names[:num_layers_block]
        else:
            cur_layer_names = layer_names[:num_layers_block]

        layer_height = 1

        for layer_idx in range(num_layers_block):
            layer_y = -(y_offset + layer_idx)

            for batch_idx in range(num_batches):
                # =========================
                # Compute time
                # =========================
                if layer_idx < len(layers_cal[block_idx][batch_idx]):
                    cal_data = layers_cal[block_idx][batch_idx][layer_idx]
                    if cal_data:
                        if isinstance(cal_data, tuple):
                            cal_data = [cal_data]
                        for start, end in cal_data:
                            duration = end - start
                            ax.broken_barh(
                                [(start, duration)],
                                (layer_y - layer_height / 2, layer_height),
                                facecolor=CAL_COLOR,
                                edgecolor=BORDER_COLOR,
                                linewidth=0.1,
                            )
                            all_points.extend([start, end])

                # =========================
                # Bubble time
                # =========================
                if layer_idx < len(layers_bubble[block_idx][batch_idx]):
                    bubble_data = layers_bubble[block_idx][batch_idx][layer_idx]
                    if bubble_data:
                        if isinstance(bubble_data, tuple):
                            bubble_data = [bubble_data]
                        elif isinstance(bubble_data, dict):
                            # dict: {start: duration, ...}
                            bubble_data = [(s, s + d) for s, d in bubble_data.items()]
                        for start, end in bubble_data:
                            duration = end - start
                            ax.broken_barh(
                                [(start, duration)],
                                (layer_y - layer_height / 2, layer_height),
                                facecolor=BUBBLE_COLOR,
                                edgecolor=BORDER_COLOR,
                                linewidth=0.1,
                                hatch="//",
                                alpha=0.4,
                            )
                            all_points.extend([start, end])

                # =========================
                # Weight update
                # =========================
                if layer_idx < len(layers_weight[block_idx][batch_idx]):
                    weight_data = layers_weight[block_idx][batch_idx][layer_idx]
                    if weight_data:
                        if isinstance(weight_data, tuple):
                            weight_data = [weight_data]
                        for start, duration in weight_data:
                            ax.broken_barh(
                                [(start, duration)],
                                (layer_y - layer_height / 2, layer_height),
                                facecolor=WEIGHT_COLOR,
                                edgecolor=BORDER_COLOR,
                                linewidth=0.1,
                            )
                            all_points.extend([start, start + duration])

        block_positions.append(-(y_offset + num_layers_block / 2 - 0.5))
        y_offset += num_layers_block

    ax.set_yticks(block_positions)
    ax.set_yticklabels(block_labels, fontsize=14)
    ax.set_ylabel("Blocks", fontsize=16)

    if all_points:
        min_t = min(all_points)
        max_t = max(all_points)
        ax.set_xlim(min_t * 0.95, max(max_t, actual_time) * 1.05)
        print(f"Time range: {min_t} to {max_t}")
    else:
        print("Warning: No time data found")
        ax.set_xlim(0, 1)

    ax.set_xlabel("Timestamp", fontsize=16)
    ax.set_title(
        f"Computation, Bubble and Weight Update Times (Time Cost: {actual_time})",
        fontsize=18,
    )

    legend_elements = [
        Patch(facecolor=CAL_COLOR, edgecolor=BORDER_COLOR, label="Compute time"),
        Patch(
            facecolor=BUBBLE_COLOR,
            edgecolor=BORDER_COLOR,
            hatch="//",
            label="Bubble time",
        ),
        Patch(facecolor=WEIGHT_COLOR, edgecolor=BORDER_COLOR, label="Weight update"),
    ]
    # ax.legend(handles=legend_elements, loc='upper right', fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(
        f"{save_path}/Combined_Timelines_Block_Batch.png", dpi=400, bbox_inches="tight"
    )
    plt.savefig(f"{save_path}/Combined_Timelines_Block_Batch.svg", dpi=600)
    plt.close(fig)
    print(f"Image saved to: {save_path}/Combined_Timelines_Block_Batch.png")


def plot_combined_timelines_batch_layers(
    save_path,
    layers_cal,
    layers_bubble,
    layers_weight,
    labels=None,
    batch_labels=None,
    actual_time=0,
):
    num_layers = len(layers_cal)
    num_batches = len(layers_cal[0]) if num_layers > 0 else 0

    if labels is None:
        labels = [f"Layer {i+1}" for i in range(num_layers)]
    if batch_labels is None:
        batch_labels = [f"Batch {i+1}" for i in range(num_batches)]

    fig, ax = plt.subplots(figsize=(22, 12))

    CAL_COLOR = "#6495ED"
    BUBBLE_COLOR = "#FFA500"
    WEIGHT_COLOR = "#D63344"
    BORDER_COLOR = "black"

    all_points = []
    layer_height = 0.8

    for layer_idx in range(num_layers):
        y_center = num_layers - layer_idx

        for batch_idx in range(num_batches):
            cal_data = layers_cal[layer_idx][batch_idx]
            if cal_data and isinstance(cal_data, tuple) and len(cal_data) == 2:
                start, end = cal_data
                ax.broken_barh(
                    [(start, end - start)],
                    (y_center - layer_height / 2, layer_height),
                    facecolor=CAL_COLOR,
                    edgecolor=BORDER_COLOR,
                    linewidth=0.2,
                    label="Compute time" if (layer_idx == 0 and batch_idx == 0) else None,
                )
                all_points.extend([start, end])

            bubble_data = layers_bubble[layer_idx][batch_idx]
            if bubble_data:
                if isinstance(bubble_data, dict):
                    for start, duration in bubble_data.items():
                        ax.broken_barh(
                            [(start, duration)],
                            (y_center - layer_height / 2, layer_height),
                            facecolor=BUBBLE_COLOR,
                            edgecolor=BORDER_COLOR,
                            hatch="//",
                            linewidth=0.1,
                            alpha=0.8,
                            label="Bubble time"
                            if (layer_idx == 0 and batch_idx == 0)
                            else None,
                        )
                        all_points.extend([start, start + duration])
                elif isinstance(bubble_data, tuple) and len(bubble_data) == 2:
                    start, end = bubble_data
                    ax.broken_barh(
                        [(start, end - start)],
                        (y_center - layer_height / 2, layer_height),
                        facecolor=BUBBLE_COLOR,
                        edgecolor=BORDER_COLOR,
                        hatch="//",
                        linewidth=0.1,
                        alpha=0.8,
                        label="Bubble time"
                        if (layer_idx == 0 and batch_idx == 0)
                        else None,
                    )
                    all_points.extend([start, end])

            weight_data = layers_weight[layer_idx][batch_idx]
            if weight_data:
                if isinstance(weight_data, tuple) and len(weight_data) == 2:
                    weight_data = [weight_data]
                if isinstance(weight_data, list):
                    for start, duration in weight_data:
                        ax.broken_barh(
                            [(start, duration)],
                            (y_center - layer_height / 2, layer_height),
                            facecolor=WEIGHT_COLOR,
                            edgecolor=BORDER_COLOR,
                            linewidth=0.2,
                            label="Weight update"
                            if (layer_idx == 0 and batch_idx == 0)
                            else None,
                        )
                        all_points.extend([start, start + duration])

    ax.set_yticks(range(1, num_layers + 1))
    ax.set_yticklabels(reversed(labels), fontsize=14)
    ax.set_ylabel("Layers", fontsize=16)

    if all_points:
        min_t, max_t = min(all_points), max(all_points)
        ax.set_xlim(min_t * 0.95, max(max_t, actual_time) * 1.05)
    else:
        ax.set_xlim(0, actual_time if actual_time > 0 else 1)

    ax.set_xlabel("Timestamp", fontsize=16)
    ax.set_title(
        f"Computation, Bubble and Weight Update Times (Time Cost: {actual_time})",
        fontsize=18,
    )

    legend_elements = [
        Patch(facecolor=CAL_COLOR, edgecolor=BORDER_COLOR, label="Compute time"),
        Patch(
            facecolor=BUBBLE_COLOR,
            edgecolor=BORDER_COLOR,
            hatch="//",
            label="Bubble time",
        ),
        Patch(facecolor=WEIGHT_COLOR, edgecolor=BORDER_COLOR, label="Weight update"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(
        f"{save_path}/Combined_Timelines_BatchLayers.png",
        dpi=400,
        bbox_inches="tight",
    )
    plt.savefig(f"{save_path}/Combined_Timelines_BatchLayers.svg", dpi=600)
    plt.close(fig)
    print(f"Image saved to: {save_path}/Combined_Timelines_BatchLayers.png")


def export_to_excel(data, filename="output.xlsx", sheet_name="Sheet1"):
    """
    Export list data to an Excel file

    Args:
        data (list): The list of data to export, can be 1D or 2D
        filename (str): Output Excel filename, defaults to 'output.xlsx'
        sheet_name (str): Excel sheet name, defaults to 'Sheet1'
    """
    try:
        # Check if data is a 1D list
        if all(not isinstance(item, list) for item in data):
            # If it's a 1D list, convert to a DataFrame with a single column
            df = pd.DataFrame(data, columns=["Data"])
        else:
            # If it's a 2D list, directly convert to DataFrame
            df = pd.DataFrame(data)

        # Export to Excel
        df.to_excel(filename, sheet_name=sheet_name, index=False)
        print(f"Data successfully exported to {filename}")

    except Exception as e:
        print(f"Error during export: {e}")


def get_metric(file_path):
    try:
        results = {"energy": None, "utilization": None, "cycles": None}

        with open(file_path, "r") as file:
            for line in file:
                # Match Energy line
                match_energy = re.search(
                    r"Energy:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*uJ", line
                )
                if match_energy and results["energy"] is None:
                    results["energy"] = float(match_energy.group(1))

                # Match Utilization line
                match_util = re.search(r"Utilization:\s*([+-]?\d+(?:\.\d+)?)\s*%", line)
                if match_util and results["utilization"] is None:
                    results["utilization"] = float(match_util.group(1))

                # Match Cycles line
                match_cycles = re.search(r"Cycles:\s*(\d+)", line)
                if match_cycles and results["cycles"] is None:
                    results["cycles"] = int(match_cycles.group(1))

                # Early exit if all three metrics are obtained
                if all(results.values()):
                    break

        return results

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def get_dummy_top_scalar_access(file_path, stats_identifier):
    """
    Extract the 'Total scalar accesses' value for dummy_top starting with a specific stats section from a text file

    Args:
    file_path (str): Path of the text file to analyze
    stats_identifier (str): Header identifying the target stats section, e.g., "Operational Intensity Stats"

    Returns:
    int or None: The scalar access integer value if a match is found; otherwise None
    """
    try:
        with open(file_path, "r") as file:
            content = file.read()

            # Use regex to find the specified stats section and the following dummy_top section
            pattern = re.compile(
                rf"{re.escape(stats_identifier)}(.*?)"  # Match stats section
                rf"=== dummy_top ===(.*?)"  # Match target dummy_top marker
                rf"Op per Byte\s*:\s*([\d.e+]+)",  # Match 'Op per Byte' line to ensure the correct section is obtained
                re.DOTALL,
            )

            match = pattern.search(content)
            if match:
                # Extract the content of the dummy_top section
                dummy_top_content = match.group(2)

                # Use regex to extract the Total scalar accesses value
                scalar_pattern = r"Total scalar accesses\s*:\s*(\d+)"
                scalar_match = re.search(scalar_pattern, dummy_top_content)

                if scalar_match:
                    return int(scalar_match.group(1))

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
    except Exception as e:
        print(f"Error: Exception occurred while reading file: {e}")

    return None


def extract_dummy_top_scalar_reads(file_path):
    # Initialize result dictionary
    result = {"Weights": None, "Inputs": None, "Outputs": None}

    current_section = None
    in_dummy_top = False

    with open(file_path, "r") as file:
        for line in file:
            # Check if entering the dummy_top area
            if "Level 16" in line:
                in_dummy_top = True
                continue

            if not in_dummy_top:
                continue

            # Detect start of subsections
            if "Weights:" in line:
                current_section = "Weights"
            elif "Inputs:" in line:
                current_section = "Inputs"
            elif "Outputs:" in line:
                current_section = "Outputs"

            # Extract scalar reads value
            if current_section and "Scalar reads (per-instance)" in line:
                # Extract the numeric part after the colon
                value_str = line.split(":")[-1].strip()
                # Remove commas and convert to integer
                value = int(value_str.replace(",", ""))
                result[current_section] = value

            # Stop processing when the next Level is encountered
            if "Level 17" in line or "Summary Stats" in line:
                break

    return result


def extract_component_scalar_access(file_path, component_name):
    """
    Extract scalar access data for a specified component from the Timeloop stats file

    Args:
    file_path -- Timeloop stats file path
    component_name -- Target component name

    Returns:
    Dictionary containing the following keys:
      'Weights' - (reads + fills) value
      'Inputs' - (reads + fills) value
      'Outputs' - (reads + fills) value
    """
    # Initialize result dictionary
    result = {
        "Weights": {"reads": 0, "fills": 0},
        "Inputs": {"reads": 0, "fills": 0},
        "Outputs": {"reads": 0, "fills": 0},
    }

    current_section = None
    found_component = False
    component_found = False

    with open(file_path, "r") as file:
        for line in file:
            # Check if entering the target component area
            if f"=== {component_name} ===" in line:
                if component_found:
                    continue
                found_component = True
                component_found = True
                continue

            if not found_component:
                continue

            # Detect start of subsections
            if "Weights:" in line:
                current_section = "Weights"
            elif "Inputs:" in line:
                current_section = "Inputs"
            elif "Outputs:" in line:
                current_section = "Outputs"
            elif "=== " in line and " ===" in line:
                # Encountered a new component, stop processing
                break

            # Extract scalar reads value
            if current_section and "Scalar reads (per-instance)" in line:
                try:
                    value_str = line.split(":")[-1].strip()
                    value = int(value_str.replace(",", ""))
                    result[current_section]["reads"] = value
                except (ValueError, IndexError):
                    pass

            # Extract scalar fills value
            if current_section and "Scalar fills (per-instance)" in line:
                try:
                    value_str = line.split(":")[-1].strip()
                    value = int(value_str.replace(",", ""))
                    result[current_section]["fills"] = value
                except (ValueError, IndexError):
                    pass

            # Stop processing when the next component or EOF is encountered
            if "Summary Stats" in line or not line:
                break

    # Calculate total access for each section (reads + fills)
    final_result = {}
    for section, values in result.items():
        total = values["reads"] + values["fills"]
        if total > 0:  # Only include sections with accesses
            final_result[section] = total

    return final_result
