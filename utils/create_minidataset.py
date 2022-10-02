import collections
import os
import pickle
import os
import numpy as np
import pandas as pd
import shutil


def transform_query(query, old_new_dict, old_new_relation_dict, name):
    if name == '1p':
        # print(query,type(query),type(query[0]),type(query[1]),query[1][0],type(query[1][0]))
        e, r = old_new_dict[query[0]], old_new_relation_dict[query[1][0]]
        new_query = tuple([e, (r,)])
    elif name == '2p':
        e1, r1, r2 = old_new_dict[query[0]], query[1][0], query[1][1]
        new_query = (e1, (r1, r2))
    elif name == '3p':
        e1, r1, r2, r3 = old_new_dict[query[0]
                                      ], query[1][0], query[1][1], query[1][2]
        new_query = (e1, (r1, r2, r3))
    elif name == '2i':
        e1, e2, r1, r2 = old_new_dict[query[0][0]], old_new_dict[query[1][0]
                                                                 ], old_new_relation_dict[query[0][1][0]], old_new_relation_dict[query[1][1][0]]
        new_query = (tuple([e1, (r1,)]), tuple([e2, (r2,)]))
    elif name == '3i':
        e1, e2, e3, r1, r2, r3 = old_new_dict[query[0][0]], old_new_dict[query[1][0]], old_new_dict[query[2][0]], \
            query[0][1], \
            query[1][1], query[2][1]
        new_query = ((e1, r1), (e2, r2), (e3, r3))
    elif name == 'ip':
        e1, e2, r1, r2, r3 = old_new_dict[query[0][0][0]], old_new_dict[query[0][1][0]], query[0][0][1], query[0][1][1], \
            query[1]
        new_query = (((e1, r1), (e2, r2)), r3)
    elif name == 'pi':
        e1, e2, r1, r2, r3 = old_new_dict[query[0][0]
                                          ], old_new_dict[1][0], query[0][1][0], query[0][1][1], query[1][1]
        new_query = ((e1, (r1, r2)), (e2, r2))
    elif name == '2in':
        e1, e2, r1, r2 = old_new_dict[query[0][0]
                                      ], old_new_dict[query[1][0]], query[0][1], query[1][1][0]
        new_query = ((e1, r1), (e2, (r2, 'n')))
    elif name == '3in':
        e1, e2, e3, r1, r2, r3 = old_new_dict[query[0][0]], old_new_dict[query[1][0]], old_new_dict[query[2][0]], \
            query[0][1], query[1][1], query[2][1][0]
        new_query = ((e1, r1), (e2, r2), (e3, (r3, 'n')))
    elif name == 'inp':
        e1, e2, r1, r2, r3 = old_new_dict[query[0][0][0]], old_new_dict[query[0][1][0]], query[0][0][1], query[0][1][0][
            0], query[2]
        new_query = (((e1, r1), (e2, (r2, 'n'))), r3)
    elif name == 'pin':
        e1, e2, r1, r2, r3 = old_new_dict[query[0][0]], old_new_dict[query[1][0]], query[0][1][0], query[0][1][1], \
            query[1][1][0]
        new_query = ((e1, (r1, r2)), (e2, (r3, 'n')))
    elif name == 'pni':
        e1, e2, r1, r2, r3 = old_new_dict[query[0][0]], old_new_dict[query[1][0]], query[0][1][0], query[0][1][1], \
            query[1][1]
        new_query = ((e1, (r1, r2, 'n')), (e2, r3))
    elif name == '2u-DNF':
        e1, e2, r1, r2 = old_new_dict[query[0][0]
                                      ], old_new_dict[query[1][0]], query[0][1], query[1][1]
        new_query = ((e1, r1), (e2, r2), ('u',))
    elif name == 'up-DNF':
        e1, e2, r1, r2, r3 = old_new_dict[query[0][0][0]], old_new_dict[query[0][1][0]], query[0][0][1], query[0][1][1], \
            query[1]
        new_query = (((e1, r1), (e2, r2), ('u',)), r3)
    elif name == '2u-DM':
        e1, e2, r1, r2 = old_new_dict[query[0][0][0]], old_new_dict[query[0][1][0]], query[0][0][1][0], query[0][1][1][
            0]
        new_query = (((e1, (r1, 'n')), (e2, (r2, 'n'))), ('n',))
    elif name == 'up-DM':
        e1, e2, r1, r2, r3 = old_new_dict[query[0][0][0]], old_new_dict[query[0][1][0]], query[0][0][1][0], \
            query[0][1][1][0], query[1][1]
        new_query = (((e1, (r1, 'n')), (e2, (r2, 'n'))), ('n', r3))
    else:
        new_query = None
        print('not valid name!')
    return new_query


def create_mini_query_set(path_folder, output_folder, all_formula_file, exempt_formula_file, mode, ratio, query_num):
    """
    If there's ratio, neglect query_num
    """
    all_formula_data = pd.read_csv(all_formula_file)
    exempt_formula_data = pd.read_csv(exempt_formula_file) if exempt_formula_file else {'formula_id': []}
    os.makedirs(output_folder, exist_ok=True)
    for type_str in all_formula_data['formula_id']:
        formula_path = os.path.join(path_folder, f'{mode}-{type_str}.csv')
        specific_formula_data = pd.read_csv(formula_path)
        if type_str in list(exempt_formula_data['formula_id']):
            shrank_formula_data = specific_formula_data
        else:
            if ratio:
                shrank_formula_data = specific_formula_data.loc[:int(len(specific_formula_data.index) * ratio) - 1]
            else:
                shrank_formula_data = specific_formula_data.loc[:query_num-1]
        shrank_formula_data.to_csv(os.path.join(output_folder, f'{mode}-{type_str}.csv'))


def shrink_train_query_set(path_folder, output_folder, exempt_formula_file, ratio, query_num):
    create_mini_query_set(path_folder, output_folder, os.path.join(path_folder, "train_formulas.csv"),
                          exempt_formula_file, 'train', ratio, query_num)
    create_mini_query_set(path_folder, output_folder, os.path.join(path_folder, "valid_formulas.csv"),
                          os.path.join(path_folder, "valid_formulas.csv"), 'valid', ratio, query_num)
    create_mini_query_set(path_folder, output_folder, os.path.join(path_folder, "test_formulas.csv"),
                          os.path.join(path_folder, "test_formulas.csv"), 'test', ratio, query_num)
    required_other_files = ['train.txt', 'valid.txt', 'test.txt', 'ent2id.pkl', 'id2ent.pkl', 'id2rel.pkl',
                            'rel2id.pkl']
    for file in required_other_files:
        shutil.copy(os.path.join(path_folder, file), os.path.join(output_folder, file))


if __name__ == "__main__":
    mode = 'train'
    FB15k_237_folder = "data/FB15k-237-betae"
    selected_file = 'data/FB15k-237-betae/selected_formulas.csv'
    one_hop_file = 'data/FB15k-237-betae/1p.csv'
    p_file = 'data/FB15k-237-betae/p_formulas.csv'
    minimal_selected_file = 'data/FB15k-237-betae/minimal_selected_formulas.csv'
    output_folder = "data/FB15k-237-betae-1p-0001"
    formula_file_path = os.path.join(FB15k_237_folder, f"{mode}_formulas.csv")

    shrink_train_query_set(FB15k_237_folder, output_folder, one_hop_file, 0.001, 0)

