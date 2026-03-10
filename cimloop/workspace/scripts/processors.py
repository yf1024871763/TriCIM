import timeloopfe.v4 as tl
import math

class ArrayContainer(tl.arch.Container):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        tl.arch.ArchNodes.add_attr("!ArrayContainer", ArrayContainer)


class MaxUtilizationDescriptorTop(tl.DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("spatial", MaxUtilizationDescriptor, None)
        super().add_attr("temporal", MaxUtilizationDescriptor, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial: MaxUtilizationDescriptor = self["spatial"]
        self.temporal: MaxUtilizationDescriptor = self["temporal"]


class MaxUtilizationDescriptor(tl.DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("factors", tl.constraints.Factors)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factors: tl.constraints.Factors = self["factors"]


class ArrayProcessor(tl.processors.Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def declare_attrs(self, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr(tl.problem.Problem, "name", str, None)
        super().add_attr(tl.problem.Problem, "dnn_name", str, None)
        super().add_attr(tl.problem.Problem, "notes", str, None)
        super().add_attr(tl.problem.Problem, "histograms", dict, {})
        super().add_attr(
            tl.arch.Leaf, "max_utilization", MaxUtilizationDescriptorTop, None
        )
        MaxUtilizationDescriptorTop.declare_attrs()
        MaxUtilizationDescriptor.declare_attrs()
        ArrayContainer.declare_attrs()

    def fetch_integer(self, spec: tl.Specification, node: tl.Node, key: str):
        v = node[key]
        if v in spec.variables:
            v = spec.variables[v]

        errstr = f"Non-integer value {v} for {key} in {node}"
        try:
            assert float(v).is_integer(), errstr
            return int(v)
        except ValueError as e:
            raise ValueError(errstr) from e

    def expand_utilization(self, spec: tl.Specification):
        expanded = {"C": 1, "M": 1}
        instance = spec.problem.instance
        for l in spec.get_nodes_of_type(tl.arch.Leaf):
            if l.constraints.spatial.get("factors_only", None) is not None:
                continue
            if l.max_utilization is not None:
                continue
            f = (
                l.spatial.get_fanout()
                // l.constraints.spatial.factors.get_minimum_product(instance)
            )
            if f <= 1:
                continue

            remaining_multipliers = []
            for m in expanded:
                if any(
                    m in spec.problem.shape.dataspace2dims(d)
                    for d in (l.constraints.spatial.no_iteration_over_dataspaces or [])
                ):
                    continue
                remaining_multipliers.append(m)

            mult_warning = {}
            prev_instance = dict(instance)

            if remaining_multipliers:
                for f in num2list_of_prime_factors(f):
                    target = min(remaining_multipliers, key=lambda x: instance[x])
                    prev = instance[target]
                    mult_warning[target] = f * mult_warning.get(target, 1)
                    instance[target] *= f

            if mult_warning:
                k = f"{','.join(mult_warning)}"
                t = f"({','.join(str(m) for m in mult_warning.values())})"
                s = f"({','.join(str(prev_instance[m]) for m in mult_warning)})"
                e = f"({','.join(str(instance[m]) for m in mult_warning)})"
                self.logger.warning(
                    f"To fill up {l.name}, multiplied {k} by {t}: {s} -> {e}"
                )
        return expanded

    def pre_parse_process(self, spec: tl.Specification):
        super().pre_parse_process(spec)
        # Pop the relevant items from the problem
        for x in ["name", "dnn_name", "notes"]:
            spec.problem.pop(x)

        histograms = spec.problem.pop("histograms")
        for k, v in histograms.items():
            spec.variables[f"{k.upper()}_HIST"] = v
        for k in list(spec.variables.keys()):
            if not k.endswith("_HIST"):
                spec.variables[k] = spec.variables.pop(k)

        prob = spec.problem
        n_rows = 1
        n_cols = 1
        n_rows_array =1
        n_cols_array =1
        parallel_dataspaces = {ds.name: 1 for ds in prob.shape.data_spaces}
        fanout_x = []
        fanout_y= []
        fanout_sum = 1
        for cim_container in spec.get_nodes_of_type(ArrayContainer):
            constraints = cim_container.constraints
            cim_container.attributes["_is_CiM"] = True

            spatial = cim_container.spatial
            meshX = self.fetch_integer(spec, spatial, "meshX")
            meshY = self.fetch_integer(spec, spatial, "meshY")
            fanout_sum *= int(meshX)
            fanout_x.append(meshX)
            fanout_y.append(meshY)
            assert meshX == 1 or meshY == 1, (
                f"Either meshX or meshY must be 1 in {spatial}. Got "
                f"{meshX=} and {meshY=}."
            )
            n_rows *= meshY
            n_cols *= meshX
            constraints.spatial.split = 99999 if meshY == 1 else 0
            for ds in parallel_dataspaces:
                if ds in (constraints.spatial.no_reuse or []):
                    parallel_dataspaces[ds] *= meshX * meshY
       
        v = spec.variables
        v["ARRAY_WORDLINES"] = f'{n_rows} * ({v["CIM_UNIT_DEPTH_CELLS"]})'
        v["ARRAY_BITLINES"] = f'{n_cols} * ({v["CIM_UNIT_WIDTH_CELLS"]})'

        for ds, n in parallel_dataspaces.items():
            v[f"ARRAY_PARALLEL_{ds.upper()}"] = n    
        '''
        for leaf in spec.get_nodes_of_type(tl.arch.Leaf):
            spatial = leaf.spatial
            if 'tile_in_chip' in leaf.name:
                tile_num = meshX
            if 'macro_in_tile' in leaf.name:
                meshX = self.fetch_integer(spec, spatial, "meshX")
                macro_num = meshX
            if 'array' in leaf.name:
                meshX = self.fetch_integer(spec, spatial, "meshX")
                array_num = meshX
        print("INPUT_BITS = ",v["INPUT_BITS"])
        data = v.copy()
        if isinstance(data["ENCODED_WEIGHT_BITS"], str):
            expression = data["ENCODED_WEIGHT_BITS"].replace("WEIGHT_BITS", str(data["WEIGHT_BITS"]))
            data["ENCODED_WEIGHT_BITS"] = eval(expression)  # 计算表达式的值
        if isinstance(data["ENCODED_OUTPUT_BITS"], str):
            expression = data["ENCODED_OUTPUT_BITS"].replace("OUTPUT_BITS", str(data["OUTPUT_BITS"]))
            data["ENCODED_OUTPUT_BITS"] = eval(expression)  # 计算表达式的值
        if isinstance(data["ENCODED_INPUT_BITS"], str):
            if "INPUT_BITS" in data["ENCODED_INPUT_BITS"]:
                expression = data["ENCODED_INPUT_BITS"].replace("INPUT_BITS", str(data["INPUT_BITS"]))
            data["ENCODED_INPUT_BITS"] = eval(expression)  # 计算表达式的值
        print(data["ENCODED_WEIGHT_BITS"])
        instance = spec.problem.instance
        tmp_instance = instance.copy()
        tmp_instance['Y'] =data["ENCODED_WEIGHT_BITS"]
        tmp_instance['Z'] =data["ENCODED_OUTPUT_BITS"] 
        weight_bits_per_slice = min(data["CIM_UNIT_WIDTH_CELLS"]*data["BITS_PER_CELL"],data["ENCODED_INPUT_BITS"])
        n_virtual_macs = data["ENCODED_OUTPUT_BITS"] * weight_bits_per_slice
        n_weith_slices = data['ENCODED_WEIGHT_BITS']//weight_bits_per_slice
        tmp_instance['Y'] //= n_virtual_macs//tmp_instance['Z'] 
        tmp_instance['M'] *= spec.variables["CIM_UNIT_DEPTH_CELLS"]
        tmp_instance['M'] //=n_weith_slices
        duplicate_factor_tmp=min(n_cols//(tmp_instance['M']*tmp_instance['Y']),n_rows//(tmp_instance['C']*tmp_instance['R']*tmp_instance['S']))
        remaining_fanout = fanout_sum //int(instance['M'])
        if remaining_fanout <duplicate_factor_tmp:
            duplicate_factor = remaining_fanout
        else :
            duplicate_factor = duplicate_factor_tmp
        print(duplicate_factor) 

        def find_max_divisible_number(limit, divisor):
    # 从 limit - 1 开始向下遍历
            for num in range(limit , 0, -1):
                if divisor % num == 0:  # 检查是否能被整除
                    return num
            return 1  # 如果没有找到，返回 None

        if(duplicate_factor <instance['P']):
            duplicate_factor = find_max_divisible_number(duplicate_factor,instance['P'])
        print('duplicate_factor = ',duplicate_factor)  

        

        for leaf in spec.get_nodes_of_type(tl.arch.Leaf):
            if (mu := leaf.max_utilization) is None:
                continue
            x = int(leaf.spatial.meshX)
            y = int(leaf.spatial.meshY)
            optimize_sucess = False   
            if 'row' in leaf.name or 'column' in leaf.name:
                for target in ["spatial", "temporal"]:
                    if (t := getattr(mu, target)) is None:
                        continue
                    
                    for i in [x]:
                        if x >1:
                            for dim in ['P','Q']:
                                print('tmp_instance = ',tmp_instance[dim])
                                if tmp_instance[dim]==1:break
                                if(tmp_instance[dim]>duplicate_factor):
                                    if(i<tmp_instance[dim] ) :
                                        if(i>=duplicate_factor):
                                            tmp = i % duplicate_factor
                                            if tmp !=0 :break
                                            leaf.constraints[target].factors.add_eq_factor(dim, duplicate_factor, overwrite=True)
                                            duplicate_factor = 1
                                            tmp_instance[dim] //= duplicate_factor
                                            optimize_sucess = True
                                        else:
                                            leaf.constraints[target].factors.add_eq_factor(dim, i, overwrite=True)
                                            tmp_instance[dim] = math.ceil(tmp_instance[dim]/i)
                                            duplicate_factor //=i
                                            optimize_sucess = True
                                            break
                                    elif (i>tmp_instance[dim]):
                                        tmp = i % duplicate_factor
                                        leaf.constraints[target].factors.add_eq_factor(dim, duplicate_factor, overwrite=True)
                                        duplicate_factor = 1
                                        tmp_instance[dim] = 1
                                        optimize_sucess = True
                                        break  
                            if optimize_sucess:break
        '''
        # Move ARRAY_WORDLINES and ARRAY_BITLINES to the top of the list
        for k in list(v.keys()):
            if k not in ["ARRAY_WORDLINES", "ARRAY_BITLINES"]:
                # If it can be casted directly to a number, it doesn't need to
                # be moved to the bottom.
                try:
                    float(v[k])
                except (ValueError, TypeError):
                    v[k] = v.pop(k)

    def process(self, spec: tl.Specification):
        if not spec.variables.pop("MAX_UTILIZATION", False):
            for l in spec.get_nodes_of_type(tl.arch.Leaf):
                l.pop("max_utilization", None)
            return
        instance = spec.problem.instance
        #instance["M"] *= spec.variables["CIM_UNIT_DEPTH_CELLS"]
        weight_slice_spill = spec.variables["N_WEIGHT_SLICES"]
        assert instance["M"] >= weight_slice_spill, (
            f"To map this problem, {weight_slice_spill} weight slices must "
            f"be mapped. We could map these to parallel output channels M, but "
            f"there are only {instance['M']} available."
        )
        #instance["M"] //= weight_slice_spill

        for l in spec.get_nodes_of_type(tl.arch.Leaf):
            l.pop("max_utilization", None)


def num2list_of_prime_factors(x: int):
    factors = []
    while x > 1:
        for i in range(2, x + 1):
            if x % i == 0:
                factors.append(i)
                x //= i
                break
    return factors
