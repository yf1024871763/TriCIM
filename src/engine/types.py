from dataclasses import dataclass


@dataclass
class LayerMetricsBundle:
    workloads: list
    compute: list
    weight_access: list
    min_tiles: list
