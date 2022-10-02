from os.path import join

import numpy as np
from numpy.linalg import matrix_rank, norm
import pandas as pd
from collections import defaultdict

from fol import parse_formula
from utils.class_util import fixed_depth_nested_dict


def create_formula_representation(formula_list, count_list, store_fold, no_depth_operator_list):
    formula_dict = fixed_depth_nested_dict(int, 2)
    all_key = set()
    for formula in formula_list:
        formula_instance = parse_formula(formula)
        formula_accumulative = formula_instance.get_root_distance(count_list, 0)

        formula_count_dict = defaultdict(int)
        for operator_key in formula_accumulative:
            for length in formula_accumulative[operator_key]:
                if operator_key in no_depth_operator_list:
                    formula_count_dict[f'{operator_key}'] += formula_accumulative[operator_key][length]
                else:
                    formula_count_dict[f'{operator_key}_{length}'] = formula_accumulative[operator_key][length]

        formula_dict[formula] = formula_count_dict
        all_key.update(formula_count_dict.keys())
    all_key = list(all_key)
    all_key.sort()
    n, m = len(all_key), len(formula_list)
    representation_matrix = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if formula_dict[formula_list[i]][all_key[j]] != 0:
                representation_matrix[i][j] = 1
    representation_data = pd.DataFrame(data=representation_matrix, index=formula_list, columns=all_key)
    print(representation_data, f'rank is {matrix_rank(representation_matrix)}')
    representation_data.to_csv(join(store_fold, 'formula_representation.csv'))
    selected_row_index = extract_independent_rows(representation_matrix)
    print(representation_data.iloc[selected_row_index])
    return selected_row_index


def extract_independent_rows(matrix):
    index_list = np.argsort(norm(matrix, axis=1))
    # reverse_index_list = np.argsort(index_list)
    sorted_matrix = matrix[index_list]
    now_rank = 0
    selected_row_index = []
    for i in range(matrix.shape[0]):
        partial_matrix = sorted_matrix[:i + 1, :]
        next_rank = matrix_rank(partial_matrix)
        if next_rank > now_rank:
            selected_row_index.append(i)
            now_rank = next_rank
    origin_selected_index_list = index_list[np.array(selected_row_index)]
    origin_selected_index_list.sort()
    return origin_selected_index_list


if __name__ == "__main__":
    data_fold = 'data/FB15k-237-betae'
    train_formula_file = pd.read_csv(join(data_fold, 'train_formulas.csv'))
    train_formula_data = train_formula_file['DNF+MultiIU']
    count = ['p', 'i', 'I']
    np_depth_operator = ['e', 'n']
    formula_selected_row_index = create_formula_representation(train_formula_data, count, data_fold, np_depth_operator)
    truncated_formula_data = train_formula_file.iloc[formula_selected_row_index]
    truncated_formula_data.to_csv(join(data_fold, 'selected_formulas.csv'))





