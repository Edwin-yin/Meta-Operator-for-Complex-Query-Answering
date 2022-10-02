import copy
from collections import defaultdict

import pandas as pd

from fol import parse_formula
from utils.class_util import fixed_depth_nested_dict


def count_single_distance(formula_instance, distance_config):
    """
    distance_config should contain two keys: use_distance and count_distance_dict
    This function does two things, give final_accumulative, assign net index for fosq instance.
    symbol_accumulative is like {'p': {0: 1}} which means 0 net symbol of p operator have one instance!

    """
    formula_instance.clear_net_index()  # clear existing net_index (probably in saved pickle)
    symbol_accumulative = fixed_depth_nested_dict(int, 2)
    for operator in distance_config['count_distance_dict']:
        if distance_config['use_distance'] == 'leaf':
            _, accumulative = formula_instance.get_leaf_distance(distance_config['count_distance_dict'][operator])
        elif distance_config['use_distance'] == 'root':
            accumulative = formula_instance.get_root_distance(distance_config['count_distance_dict'][operator], 0)
        elif distance_config['use_distance'] in ['output_binary', 'input_binary', 'input', 'output']:
            accumulative = formula_instance.assign_net_symbol(distance_config['use_distance'],
                                                              list(distance_config['count_distance_dict'].keys()), 'a')
        else:
            assert False, 'Not valid distance setting!'
        if distance_config['use_distance'] in ['leaf', 'root']:  # If use root/leaf, index will be computed by distance.
            now_index, distance2index = 0, {}
            for distance in sorted(list(accumulative[operator])):
                distance2index[distance] = now_index
                symbol_accumulative[operator][now_index] += 1
                now_index += 1
            formula_instance.assign_net_index({operator: distance2index}, distance_config)
        else:
            symbol_accumulative = accumulative
    return symbol_accumulative


def count_all_distance(formula_id_file, distance_config, formula_specific):
    """
    contain_operator_dict is like {'type0002_(p,(p,(e)))': {'p': [0, 1]}}
    which means this formula(say '(p,(p,(e)))') contains p of 0,1 index.
    contain_formula_dict is transformed as {'p': {0: 'type0002_(p,(p,(e)))', 1: ''type0002_(p,(p,(e)))''}}

    operator_split_dict tells for how operator split into different meta-learning tasks.
    example: {'p': [1, 2, 3], 'i': [1, 2]},  {'p': 'a', 'p', 'i', 'n'}
    """
    formula_id_data = pd.read_csv(formula_id_file)
    all_formula = set()
    operator_split_dict = defaultdict(set)
    contain_operator_dict, contain_formula_dict = defaultdict(dict), fixed_depth_nested_dict(list, 2)
    if not distance_config:
        return None,  operator_split_dict
    for line_index in formula_id_data.index:
        type_str = formula_id_data['formula_id'][line_index]
        for normal_form in all_normal_form:
            formula = formula_id_data[normal_form][line_index]
            full_formula = f'{type_str}_{formula}'
            if full_formula in all_formula:
                pass
            else:
                all_formula.add(full_formula)
                formula_instance = parse_formula(formula)
                final_accumulative = count_single_distance(formula_instance, distance_config)
                for operator in distance_config['count_distance_dict']:
                    contain_operator_dict[full_formula][operator] = \
                        sorted(list(final_accumulative[operator].keys()))
                for operator in distance_config['count_distance_dict']:
                    operator_split_dict[operator].update(final_accumulative[operator].keys())
    # operator_split_dict['I'] = max(operator_split_dict['I'], operator_split_dict['i'])
    for operator in operator_split_dict:
        operator_split_dict[operator] = sorted(list(operator_split_dict[operator]))
    for full_formula in contain_operator_dict:
        for operator in contain_operator_dict[full_formula]:
            for operator_type in contain_operator_dict[full_formula][operator]:
                contain_formula_dict[operator][operator_type].append(full_formula)
    if formula_specific:
        formula_specific_dict = {formula: copy.deepcopy(distance_config) for formula in all_formula}
        return formula_specific_dict, operator_split_dict, contain_formula_dict
    return distance_config, operator_split_dict, contain_formula_dict


def count_freeze_dumps(formula_id_file, interested_normal_forms, formula_distance_config, operator_split_dict,
                       freeze_other: bool):
    """
    compute the freeze dumps for operator_MAML
    """
    formula_id_data = pd.read_csv(formula_id_file)
    all_formula = set()
    all_instance = {}
    freeze_dumps_dict = fixed_depth_nested_dict(str, 3)
    for line_index in formula_id_data.index:
        type_str = formula_id_data['formula_id'][line_index]
        for normal_form in interested_normal_forms:
            formula = formula_id_data[normal_form][line_index]
            full_formula = f'{type_str}_{formula}'
            if full_formula in all_formula:
                pass
            else:
                all_formula.add(full_formula)
                formula_instance = parse_formula(formula)
                count_single_distance(formula_instance, formula_distance_config[full_formula])
                all_instance[full_formula] = formula_instance
    for operator_name in formula_distance_config[list(formula_distance_config.keys())[0]]['count_distance_dict']:
        for operator_type in operator_split_dict[operator_name]:
            now_index_list = copy.deepcopy(operator_split_dict[operator_name])
            now_index_list.remove(operator_type)
            for formula in all_formula:
                now_instance = all_instance[formula]
                all_operators = {'p', 'i', 'u', 'd', 'D', 'e', 'r', 'n'}
                if freeze_other:
                    distance_freeze_dict = {operator: [-1] if operator not in operator_split_dict else
                                            list(range(operator_split_dict[operator_name])) for operator in all_operators}
                else:
                    distance_freeze_dict = {operator_name: now_index_list}
                freeze_dumps = now_instance.freeze_distance_dumps(distance_freeze_dict)
                freeze_dumps_dict[operator_name][operator_type][formula] = freeze_dumps
    return freeze_dumps_dict


all_normal_form = ['original', 'DeMorgan', 'DeMorgan+MultiI', 'DNF', 'diff', 'DNF+diff', 'DNF+MultiIU', 'DNF+MultiIUd',
                   'DNF+MultiIUD']
