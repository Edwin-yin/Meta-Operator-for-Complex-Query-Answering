import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import collections
import numpy as np
import sys
import shutil

all_formula_data = pd.read_csv('data/test_generated_formula_anchor_node=3.csv')


def fill_dict_to_array(whole_dict, array, depth2key, now_depth, full_depth):
    for index, key in enumerate(depth2key[now_depth]):
        if now_depth == full_depth:
            array[index] = whole_dict[key]
        else:
            fill_dict_to_array(whole_dict[key], array[index], depth2key, now_depth + 1, full_depth)


def nested_dict_to_array(nested_dictionary, meta_key_list):
    """
    key: p,e , MRR
    meta_key: formula/metric
    """
    nested_depth = len(meta_key_list)
    full_depth2key = {i: [] for i in range(nested_depth)}
    now_dict = nested_dictionary
    for depth in range(nested_depth):
        now_keys = now_dict.keys()
        full_depth2key[depth] = list(now_keys)
        now_dict = now_dict[full_depth2key[depth][0]]
    final_array = np.zeros([len(full_depth2key[i]) for i in range(nested_depth)])
    fill_dict_to_array(nested_dictionary, final_array, full_depth2key, 0, nested_depth - 1)
    return final_array, full_depth2key


def remove_checkpoint(folder_path, ckpt_step_list, exempt_step_list, exempt_largest: bool = True):
    ckpt_step_list.sort()
    exempt_step_list.sort()
    n = len(ckpt_step_list)
    if exempt_largest and ckpt_step_list[-1] not in exempt_step_list:
        exempt_step_list.append(ckpt_step_list[-1])
    for i, step in enumerate(ckpt_step_list):
        if step not in exempt_step_list:
            file_path = os.path.join(folder_path, f'{step}.ckpt')
            os.remove(file_path)
            print(f'Delete {step}.ckpt')


def merge_continue_folder(original_folder, continue_folder, ckpt_step_list, saving_step_list, saving_mode_list,
                          do_assertion: bool = True):
    ckpt_step_list.sort()
    saving_step_list.sort()
    shutil.copy(os.path.join(continue_folder, 'meta.json'), os.path.join(original_folder, 'continue_meta.json'))
    train_data = pd.read_csv(os.path.join(continue_folder, 'train.csv'))
    original_train_data = pd.read_csv(os.path.join(original_folder, 'train.csv'))
    original_train_steps = list(original_train_data['step'])
    with open(os.path.join(original_folder, 'train.csv'), 'at') as f:
        for index in train_data.index:
            step = train_data['step'].loc[index]
            if step in original_train_steps:
                if do_assertion:
                    assert original_train_data.loc[original_train_data['step'] == step].equals(
                        train_data.loc[train_data['step'] == step]), print(step)
            else:
                f.write(','.join([str(train_data[c].loc[index]) for c in train_data.columns]) + '\n')
    for step in ckpt_step_list:
        shutil.copy(os.path.join(continue_folder, f'{step}.ckpt'), os.path.join(original_folder, f'{step}.ckpt'))
    for step in saving_step_list:
        for mode in saving_mode_list:
            shutil.copy(os.path.join(continue_folder, f'all_logging_{mode}_{step}.pickle'),
                        os.path.join(original_folder, f'all_logging_{mode}_{step}.pickle'))


def new_merge_pickle(folder_path, step_list, meta_key_list, mode):
    all_logging = {}
    for step_id in range(len(step_list)):
        step = step_list[step_id]
        filename = f'all_logging_{mode}_{step}.pickle'
        with open(os.path.join(folder_path, filename), 'rb') as f:
            single_log = pickle.load(f)
            all_logging[step] = single_log
    final_array, depth2key = nested_dict_to_array(all_logging, meta_key_list)
    with open(os.path.join(folder_path, f'new_merge_logging_{mode}.pickle'), 'wb') as f:
        pickle.dump([final_array, depth2key, meta_key_list], f)


def new_read_merge_pickle(folder_path, fixed_dict, mode, transpose=False, percentage=False):
    with open(os.path.join(folder_path, f'new_merge_logging_{mode}.pickle'), 'rb') as f:
        single_log = pickle.load(f)
        final_array, depth2key, meta_key_list = single_log
        if percentage:
            final_array *= 100
        selected_index_list = []
        left_meta_key_index_list = []
        for i, meta_key in enumerate(meta_key_list):
            if meta_key in fixed_dict:
                index2key = depth2key[i]
                key2index = {index2key[j]: j for j in range(len(index2key))}
                fixed_index = key2index[fixed_dict[meta_key]]
            else:
                fixed_index = slice(len(depth2key[i]))
                left_meta_key_index_list.append(i)
            selected_index_list.append(fixed_index)
        assert len(left_meta_key_index_list) <= 2, "Get more than two meta keys unfixed!"
        selected_log = final_array[tuple(selected_index_list)]
        if len(left_meta_key_index_list) == 1:
            left_meta_key_index = left_meta_key_index_list[0]
            left_meta_key_name = meta_key_list[left_meta_key_index]
            reindexed_data = pd.DataFrame(data=selected_log, index=depth2key[left_meta_key_index_list[0]],
                                          columns=[str(fixed_dict)])
            reindexed_data.to_csv(os.path.join(folder_path, f'selected_log_{mode}_{left_meta_key_name}.csv'))
        elif len(left_meta_key_index_list) == 2:
            reindexed_data = pd.DataFrame(
                data=selected_log, index=depth2key[left_meta_key_index_list[0]],
                columns=depth2key[left_meta_key_index_list[1]])
            left_meta_key_name_list = [meta_key_list[i] for i in left_meta_key_index_list]
            if transpose:
                reindexed_data = reindexed_data.transpose()
            reindexed_data.to_csv(
                os.path.join(folder_path, f'selected_log_{mode}_'
                                          f'{left_meta_key_name_list[0]}_{left_meta_key_name_list[1]}.csv'))
        return reindexed_data


def pickle_select_form(pickle_path, test_step, meta_key_list, fixed_dict, normal_form, formula_file=None):
    if formula_file:
        formula_data = pd.read_csv(formula_file)
    else:
        formula_data = all_formula_data
    new_merge_pickle(pickle_path, [test_step], meta_key_list, 'test')
    loading_data = new_read_merge_pickle(pickle_path, fixed_dict, 'test', False, False)
    if normal_form == 'best':
        best_formula_list, best_score_list = [], []
        for type_str in formula_data.index:
            now_best_score, now_best_formula = 0, None
            for possible_formula in formula_data.loc[type_str]:
                if possible_formula in loading_data.index:
                    if loading_data.loc[possible_formula].values[0] > now_best_score:
                        now_best_score, now_best_formula = loading_data.loc[possible_formula].values[0], \
                                                           possible_formula
            best_formula_list.append(now_best_formula)
            best_score_list.append(now_best_score)
        output_data = pd.DataFrame(data={'formula': best_formula_list, 'score': best_score_list},
                                   index=formula_data.index)
    else:
        all_formulas = formula_data[normal_form]
        normal_form_index_list = [i for i in loading_data.index if loading_data in all_formulas]
        output_data = loading_data.loc[normal_form_index_list]
    output_data.to_csv(os.path.join(pickle_path, f'chose_form_{normal_form}.csv'))


def process_output_whole_folder(whole_folder, use_MAML, auto_delete):
    output_dict = {}
    exist_sub_dir = False
    delete_folder = False
    for sub_file in os.listdir(whole_folder):
        full_sub_path = os.path.join(whole_folder, sub_file)
        if os.path.isdir(full_sub_path) and sub_file != '.ipynb_checkpoints':
            exist_sub_dir = True
            sub_output_dict = process_output_whole_folder(full_sub_path, use_MAML, auto_delete)
            output_dict.update(sub_output_dict)
    if not exist_sub_dir:  # The final dir that contains output
        now_model_name = whole_folder.split('/')[-1].split('_')[0]
        file_list = os.listdir(whole_folder)
        ckpt_step_list = [int(ckpt_file.split('.')[0]) for ckpt_file in file_list if ckpt_file.endswith('.ckpt')]
        logging_test_step = [int(logging_file.split('.')[0].split('_')[-1])
                             for logging_file in file_list if logging_file.endswith('.pickle')
                             and logging_file.split('.')[0].split('_')[2] == 'test']
        largest_ckpt_step = max(ckpt_step_list) if ckpt_step_list else 0
        print(f'processing folder {whole_folder}， {len(logging_test_step)}')
        if len(logging_test_step):
            if use_MAML:
                new_merge_pickle(whole_folder, sorted(logging_test_step),
                                 ["step", "adaptation_step", "formula", "metric"], 'test')
                new_read_merge_pickle(whole_folder, {'metric': 'MRR', 'adaptation_step': 5}, mode='test',
                                      percentage=True, transpose=False)
                new_read_merge_pickle(whole_folder, {'metric': 'MRR', 'step': max(logging_test_step)}, mode='test',
                                      percentage=True, transpose=False)
            else:
                new_merge_pickle(whole_folder, sorted(logging_test_step),
                                 ["step", "formula", "metric"], 'test')
                new_read_merge_pickle(whole_folder, {'metric': 'MRR'}, mode='test', percentage=True, transpose=False)
        output_dict[whole_folder] = largest_ckpt_step
        if largest_ckpt_step == 0 and len(logging_test_step) == 0:
            delete_folder = True
    # if delete_folder and not exist_sub_dir and auto_delete:
        # shutil.rmtree(whole_folder)
    return output_dict


def aggregate_test(folder_path, prefix):
    output_folder = os.path.join(folder_path, f'{prefix}_aggregated')
    os.makedirs(output_folder, exist_ok=True)
    for sub_file in os.listdir(folder_path):
        if sub_file.startswith(prefix) and os.path.isdir(os.path.join(folder_path, sub_file)) and sub_file != f'{prefix}_aggregated':
            for sub_sub_file in os.listdir(os.path.join(folder_path, sub_file)):
                #print(sub_sub_file)
                if sub_sub_file.endswith('.pickle'):
                    full_path = os.path.join(folder_path, sub_file, sub_sub_file)
                    shutil.copy(full_path, os.path.join(output_folder, sub_sub_file))



EFO1_train = {  # 1234
    'LogicE': 'EFO-1_log/EFO-1_train/LogicE220812.20:14:348238bbd0'
}

NGC_naive_distance = {
    'non_distance': "EFO-1_log/EFO-1_LogicE_p_distance0220605.15:59:43d89bcf0a",
    'root_0003': "EFO-1_log/EFO-1_LogicE_p_distance1220605.16:00:3497943bea",
    'leaf_0003': "EFO-1_log/EFO-1_LogicE_p_distance2220605.16:05:5626abed3a",
    'root_0006': "EFO-1_log/EFO-1_LogicE_p_distance3220605.16:05:568a0fc320",
    'leaf_0006': "EFO-1_log/EFO-1_LogicE_p_distance4220605.16:01:45872a0fe8",
    'root_001': "EFO-1_log/EFO-1_LogicE_p_distance5220605.16:04:05e32825d4",
    'leaf_001': "EFO-1_log/EFO-1_LogicE_p_distance6220605.16:02:52eef92295"
}

LogicE_operator_MAML = {  # first three are 1234
    'original_bad_finetune_not_shrink': "EFO-1_log/operator_Meta/LogicE220708.23:33:42811410b7",
    'original': "EFO-1_log/operator_Meta/LogicE220727.16:50:132e7153da",
    'original_new': "EFO-1_log/operator_Meta/LogicE220804.16:37:2749ec6896",
    'NGC_only_p': "EFO-1_log/operator_Meta/LogicE_only_p220805.09:21:153dc2f177",
    'NGC_big_lr': "EFO-1_log/operator_Meta/LogicE_big_lr220804.13:59:404b537793",
    'NGC_original': "EFO-1_log/operator_Meta/LogicE_default220805.09:21:02007008e7",
    'NGC_leaf': 'EFO-1_log/operator_Meta/LogicE_leaf220805.09:13:574efd75e9',
    'NGC_multiple_i': 'EFO-1_log/operator_Meta/LogicE_multiple_i220806.12:19:005e514919',
    'NGC_multiple_i_2': "EFO-1_log/operator_Meta/LogicE_multiple_i_2220806.12:19:388a2d598f",
    'NGC_not_shrink': "EFO-1_log/operator_Meta/LogicE_not_shrink220805.09:15:192b266eca",
    'NGC_selected': "EFO-1_log/operator_Meta/LogicE_selected220805.09:20:13b226afc1",
    'NGC_adapt_p': "EFO-1_log/operator_Meta/LogicE_adapt_p220805.09:15:48da37a3c3",
    'NGC_adapt_p_with_p': "EFO-1_log/operator_Meta/LogicE_adapt_p_with_p220805.09:21:155f787e94",
    'NGC_big_lr_real': 'EFO-1_log/operator_Meta/LogicE_big_lr_real220811.13:55:4031ca748f',
    'NGC_big_lr_not_shrink': 'EFO-1_log/operator_Meta/LogicE_big_lr_no_shrink220811.13:55:5973ef7ad4',
    'NGC_01_lr': 'EFO-1_log/operator_Meta/LogicE_01_lr220811.13:55:40e305980a',
    'NGC_01_lr_not_shrink': 'EFO-1_log/operator_Meta/LogicE_01_lr_not_shrink220811.13:56:51ae67b41d',
    'NGC_08_lr': 'EFO-1_log/operator_Meta/LogicE_008_lr220811.13:56:5106d2f95b',
    'NGC_08_lr_not_shrink': 'EFO-1_log/operator_Meta/LogicE_008_lr_not_shrink220811.14:06:1739f18e9e',
    'NGC_adapt_non_distance': 'EFO-1_log/operator_Meta/LogicE_adapt_non_instance220812.03:47:13056faf62',
    'NGC_adapt_non_distance_2': 'EFO-1_log/operator_Meta/LogicE_adapt_non_instance_2220812.04:00:4770759ba7',
    '1p-14968-default': 'EFO-1_log/operator_Meta-1p-14968/LogicE_default220805.10:37:38ce83c1f2',
    '1p-14968-selected': 'EFO-1_log/operator_Meta-1p-14968/LogicE_selected220805.10:37:51d09022fe',
    '1p-14968_adapt_i': 'EFO-1_log/operator_Meta-1p-14968/LogicE_adapt_i220817.08:33:154f9af053',
    '1p-14968_selected_adapt_i': 'EFO-1_log/operator_Meta-1p-14968/LogicE_selected_adapt_i220817.08:41:59b54a9957',
    '1p-01-default': 'EFO-1_log/operator_Meta-1p-01/LogicE_default220805.10:39:21374c450f',
    '1p-01-selected': 'EFO-1_log/operator_Meta-1p-01/LogicE_selected220805.10:39:2166ce7e30',
    '1p-01-adapt_i': 'EFO-1_log/operator_Meta-1p-01/LogicE_adapt_i220817.08:42:582a667bd5',
    '1p-01-selected_adapt_i': 'EFO-1_log/operator_Meta-1p-01/LogicE_selected_adapt_i220817.08:42:375671fa1f',
    '1p-001-default': 'EFO-1_log/operator_Meta-1p-001/LogicE_default220817.08:38:5932ffbc65',
    '1p-001-selected': 'EFO-1_log/operator_Meta-1p-001/LogicE_selected220817.08:39:19cebdea8b',
    '1p-001-adapt_i': 'EFO-1_log/operator_Meta-1p-001/LogicE_adapt_i220817.08:43:563c7cba95',
    '1p-001-selected_adapt_i': 'EFO-1_log/operator_Meta-1p-001/LogicE_selected_adapt_i220817.08:44:12dcf230a7',
    '1p-1496-default': 'EFO-1_log/operator_Meta-1p-1496/LogicE_default220817.08:44:46311ad12d',
    '1p-1496-adapt_i': 'EFO-1_log/operator_Meta-1p-1496/LogicE_adapt_i220817.08:52:39a7149e80',
    '1p=1496-selected': 'EFO-1_log/operator_Meta-1p-1496/LogicE_selected220817.08:44:30c7904b79',
    '1p-1496-selected_adapt_i': 'EFO-1_log/operator_Meta-1p-1496/LogicE_selected_adapt_i220817.08:46:039bc25169'
}

ConE_operator_MAML = {  # 1234
    'original': 'EFO-1_log/operator_Meta/ConE220821.20:54:169bcad129'
}

LogicE_operator_MAML_NGC_continue = {
    'NGC_big_lr': 'EFO-1_log/operator_Meta/LogicE_big_lr220811.13:50:505a7e48b1',
    'NGC_adapt_p': 'EFO-1_log/operator_Meta/LogicE_adapt_p220811.14:03:452609470d',
    'NGC_adapt_p_with_p': 'EFO-1_log/operator_Meta/LogicE_adapt_p_with_p220811.14:05:48c023b0c1',
    'NGC_leaf': 'EFO-1_log/operator_Meta/LogicE_leaf220811.13:55:40c8f1dcd3',
    'NGC_multiple_i': 'EFO-1_log/operator_Meta/LogicE_multiple_i220812.04:01:4169c64043',
    'NGC_multiple_i_2': 'EFO-1_log/operator_Meta/LogicE_multiple_i_2220811.14:06:2287d4d295',
    'NGC_not_shrink': 'EFO-1_log/operator_Meta/LogicE_not_shrink220811.13:59:15c98a519d',
    'NGC_selected': 'EFO-1_log/operator_Meta/LogicE_selected220811.14:05:2356928109'
}
NGC_original_for_compare = {
    '1p-14968-default': 'EFO-1_log/original-1p-14968/LogicE_default220811.13:55:0938859355',
    '1p-14968-selected': 'EFO-1_log/original-1p-14968/LogicE_selected220811.13:59:15122aafd6',
    '1p-01-default': 'EFO-1_log/original-1p-01/LogicE_default220811.14:02:072d7ff680',
    '1p-01-selected': 'EFO-1_log/original-1p-01/LogicE_selected220811.14:02:0710c595f4',
    'original_multiI': 'EFO-1_log/EFO-1_train/LogicE_multiple_i220812.03:58:277a90c2e3',
    '1p-001-default': 'EFO-1_log/original-1p-001/LogicE_default220817.08:13:54de9c73b9',
    '1p-001-selected': 'EFO-1_log/original-1p-001/LogicE_selected220817.08:14:2719843422',
    '1p-1496-default': 'EFO-1_log/original-1p-1496/LogicE_default220817.08:30:434ea827f1',
    '1p-1496-selected': 'EFO-1_log/original-1p-1496/LogicE_selected220817.08:33:049a42ff6e'
}
evaluate_LogicE_operator_MAML = {  # 1234
    'benchmark': "EFO-1_log/benchmark_operator_Meta/benchmark_LogicE220723.22:44:181fcc36f5"
}
# 1234
test_full_formula = {
    'EFO-1': "EFO-1_log/test_full_formula/test_EFO-1_LogicE.yaml220729.16:50:04256de4ed/",
    'MAML': "EFO-1_log/test_full_formula/test_MAML_LogicE.yaml220730.14:51:10b1f37cbe"
}

MAML_ConE_1234 = {
    '4w': 'EFO-1_log/MAML_log/MAML_ConE220308.22:54:17183ac563',
    '8w': 'EFO-1_log/MAML_log/MAML_ConE220306.16:34:25a6d93d41'
}

LogicE_operator_MAML_debugged = {  # NGC   but not have contain formula dict,
    'original': "EFO-1_log/operator_Meta/LogicE_default220827.13:29:24a5274ee8",
    'NGC_only_p': 'EFO-1_log/operator_Meta/LogicE_only_p220827.13:38:50a4b7cd31',
    'NGC_big_lr': 'EFO-1_log/operator_Meta/LogicE_big_lr220827.13:29:36560861aa',
    'NGC_original': "EFO-1_log/operator_Meta/LogicE_default220827.13:29:24a5274ee8",
    'NGC_leaf': 'EFO-1_log/operator_Meta/LogicE_leaf220827.13:30:06bd5f7db2',
    'NGC_multiple_i': None,
    'NGC_multiple_i_2': None,
    'NGC_not_shrink': "EFO-1_log/operator_Meta/LogicE_not_shrink220827.13:38:246571dde3",
    'NGC_selected': "EFO-1_log/operator_Meta/LogicE_selected220827.13:38:27be00376b",
    'NGC_adapt_p': "EFO-1_log/operator_Meta/LogicE_adapt_p220827.13:34:1331f18141",
    'NGC_adapt_p_with_p': "EFO-1_log/operator_Meta/LogicE_adapt_p_with_p220827.13:38:418f427941",
    'NGC_big_lr_real': 'EFO-1_log/operator_Meta/LogicE_big_lr_real220827.13:38:5104bfe037',
    'NGC_big_lr_not_shrink': 'EFO-1_log/operator_Meta/LogicE_big_lr_no_shrink220827.13:34:16c7e7f47b',
    'NGC_01_lr': 'EFO-1_log/operator_Meta/LogicE_01_lr220827.13:34:11eab71668',
    'NGC_01_lr_not_shrink': 'EFO-1_log/operator_Meta/LogicE_01_lr_not_shrink220827.13:41:42242f9a23',
    'NGC_008_lr': 'EFO-1_log/operator_Meta/LogicE_008_lr220827.13:41:45f77e2086',
    'NGC_008_lr_not_shrink': 'EFO-1_log/operator_Meta/LogicE_008_lr_not_shrink220827.13:41:44f29cdcba',
    'NGC_adapt_non_distance': 'EFO-1_log/operator_Meta/LogicE_adapt_non_instance220827.13:38:26fba3709d',
    'NGC_adapt_non_distance_2': 'EFO-1_log/operator_Meta/LogicE_adapt_non_instance_2220827.13:38:274f35e4a3',
    'NGC_entBN': 'EFO-1_log/operator_Meta/LogicE_entBN220830.13:03:34e42d6b29',
    'NGC_entrelBN': 'EFO-1_log/operator_Meta/LogicE_entrelBN220830.13:03:527fe3e0b1',
    'NGC_proBN': 'EFO-1_log/operator_Meta/LogicE_proBN220830.13:02:5176ead43a',
    '1p-14968-default': 'EFO-1_log/operator_Meta-1p-14968/LogicE_default220827.13:25:565455be8a',
    '1p-14968-selected': 'EFO-1_log/operator_Meta-1p-14968/LogicE_selected220827.13:30:11da7b71a2',
    '1p-14968_adapt_i': 'EFO-1_log/operator_Meta-1p-14968/LogicE_adapt_i220827.13:38:1130780592',
    '1p-14968_selected_adapt_i': 'EFO-1_log/operator_Meta-1p-14968/LogicE_selected_adapt_i220827.13:34:084e17a37c',
    '1p-01-default': 'EFO-1_log/operator_Meta-1p-01/LogicE_default220827.13:38:2993d478d3',
    '1p-01-selected': 'EFO-1_log/operator_Meta-1p-01/LogicE_selected220827.13:34:09d4ae89ff',
    '1p-01-adapt_i': 'EFO-1_log/operator_Meta-1p-01/LogicE_adapt_i220827.13:34:074617f4b6',
    '1p-01-selected_adapt_i': 'EFO-1_log/operator_Meta-1p-01/LogicE_selected_adapt_i220827.13:34:084b4b6135',
    '1p-01-only_p': 'EFO-1_log/operator_Meta-1p-01/only_p220831.01:08:409061cb30',
    '1p-001-default': 'EFO-1_log/operator_Meta-1p-001/LogicE_default220827.13:38:414ecae8bb',
    '1p-001-default_another': 'EFO-1_log/operator_Meta-1p-001/LogicE_default220827.13:39:01d11c2827',
    '1p-001-selected': 'EFO-1_log/operator_Meta-1p-001/LogicE_selected220827.13:38:421af73fad',
    '1p-001-adapt_i': 'EFO-1_log/operator_Meta-1p-001/LogicE_adapt_i220827.13:39:00927b0ee4',
    '1p-001-selected_adapt_i': 'EFO-1_log/operator_Meta-1p-001/LogicE_selected_adapt_i220827.13:39:00f8143573',
    '1p-001-only-p': 'EFO-1_log/operator_Meta-1p-001/only_p220830.23:42:0000f9f94a',
    '1p-1496-default': 'EFO-1_log/operator_Meta-1p-1496/LogicE_default220827.13:38:388d57457e',
    '1p-1496-adapt_i': 'EFO-1_log/operator_Meta-1p-1496/LogicE_adapt_i220827.13:34:29f6a4a5a4',
    '1p=1496-selected': 'EFO-1_log/operator_Meta-1p-1496/LogicE_selected220827.13:34:22f4eb1bde',
    '1p-1496-selected_adapt_i': 'EFO-1_log/operator_Meta-1p-1496/LogicE_selected_adapt_i220827.13:34:31996778e4',
    '1p-0001-only-p': 'EFO-1_log/operator_Meta-1p-0001/only_p220830.12:28:56eeec39c2'
}
NGC_unfinished = {
    '4_001': 'EFO-1_log/operator_Meta/LogicE_4_step220823.12:25:48b3cbe776'
}

LogicE_operator_contain_operator_dict_1p_01 = {  # after 9.10, NGC
    'root_004_1': 'EFO-1_log/operator_Meta-1p-01/LogicE_root_lr_0.004_step_1_m_0_0220912.18:04:41df2990ac',
    'input_004_1': 'EFO-1_log/operator_Meta-1p-01/LogicE_input_lr_0.004_step_1_m_0_0220912.13:37:386f958f8e',
    'input_016_1': 'EFO-1_log/operator_Meta-1p-01/LogicE_input_lr_0.016_step_1_m_0_0220912.13:41:464395a58e',
    'output_004_1': 'EFO-1_log/operator_Meta-1p-01/LogicE_output_lr_0.004_step_1_m_0_0220912.18:02:51f8234838'
}
continue_dict = {
    'EFO-1_log/operator_Meta-1p-01/BetaE_input_lr_0.002_step_4_m_0_0220913.09:19:17237664ad': 150000,
    'EFO-1_log/operator_Meta-1p-01/BetaE_input_lr_0.004_step_1_m_0_0220913.09:19:337f6ab6e0': 150000,
    'EFO-1_log/operator_Meta-1p-01/BetaE_input_lr_0.016_step_1_m_0_0220913.09:19:51e962972b': 150000,

    'EFO-1_log/operator_Meta-1p-01/BetaE_leaf_lr_0.002_step_4_m_0_0220913.09:20:4134bc7168': 150000,


    'EFO-1_log/operator_Meta-1p-01/LogicE_input_lr_0.002_step_4_m_0_0220912.13:41:41880775de': 300000,

    'EFO-1_log/operator_Meta-1p-01/LogicE_leaf_lr_0.002_step_4_m_0_0220912.17:52:55d08b3d8c': 300000,

    'EFO-1_log/operator_Meta-1p-01/LogicE_output_lr_0.002_step_4_m_0_0220912.17:55:397bed58c6': 300000,

    'EFO-1_log/operator_Meta-1p-01/LogicE_root_lr_0.002_step_4_m_0_0220912.18:04:32772a4a2b': 300000

}
to_continue_dict = {
    'BetaE_input_002': '',
    'BetaE_input_004': 'EFO-1_log/operator_Meta-1p-01/BetaE_input_lr_0.004_step_1_m_0_0220913.09:19:337f6ab6e0',
    'BetaE_input_016': 'EFO-1_log/operator_Meta-1p-01/BetaE_input_lr_0.016_step_1_m_0_0220913.09:19:51e962972b',
    'BetaE_leaf_002': 'EFO-1_log/operator_Meta-1p-01/BetaE_leaf_lr_0.002_step_4_m_0_0220913.09:20:4134bc7168',
}

finish_continue_dict = {
    'BetaE_input_002': 'EFO-1_log/operator_Meta-1p-01/LogicE_output_binary_lr_0.004_step_1_m_0_0220926.13:44:4708fe4b21',
    'BetaE_input_004': 'EFO-1_log/operator_Meta-1p-01/BetaE_input_lr_0.004_step_1_m_0_0220918.13:57:5729ef46b0',
    'BetaE_input_016': 'EFO-1_log/operator_Meta-1p-01/BetaE_input_lr_0.016_step_1_m_0_0220918.13:58:18df9aa162',
    'BetaE_leaf_002': 'EFO-1_log/operator_Meta-1p-01/BetaE_leaf_lr_0.002_step_4_m_0_0220918.13:58:29952f203d'
}

"""
merge_continue_folder(LogicE_operator_MAML['NGC_multiple_i'], LogicE_operator_MAML_NGC_continue['NGC_multiple_i'],
                      list(range(270000, 540000, 90000)), list(range(225000, 465000, 15000)), ['valid', 'test'], False)
merge_continue_folder(LogicE_operator_MAML['NGC_multiple_i_2'], LogicE_operator_MAML_NGC_continue['NGC_multiple_i_2'],
                      list(range(270000, 540000, 90000)), list(range(225000, 465000, 15000)), ['valid', 'test'], False)

merge_continue_folder(LogicE_operator_MAML['NGC_adapt_p_with_p'],
                      LogicE_operator_MAML_NGC_continue['NGC_adapt_p_with_p'], [450000],
                      list(range(420000, 465000, 15000)), ['valid', 'test'], False)
merge_continue_folder(LogicE_operator_MAML['NGC_leaf'],
                      LogicE_operator_MAML_NGC_continue['NGC_leaf'], list(range(360000, 540000, 90000)),
                      list(range(360000, 465000, 15000)), ['valid', 'test'], False)
merge_continue_folder(LogicE_operator_MAML['NGC_not_shrink'],
                      LogicE_operator_MAML_NGC_continue['NGC_not_shrink'], list(range(360000, 540000, 90000)),
                      list(range(360000, 465000, 15000)), ['valid', 'test'], False)
merge_continue_folder(LogicE_operator_MAML['NGC_selected'],
                      LogicE_operator_MAML_NGC_continue['NGC_selected'], list(range(450000, 540000, 90000)),
                      list(range(435000, 465000, 15000)), ['valid', 'test'], False)

new_merge_pickle(LogicE_operator_MAML['NGC_adapt_non_distance'], list(range(15000, 465000, 15000)),
                 ["step", "adaptation_step", "formula", "metric"], 'test')
new_read_merge_pickle(LogicE_operator_MAML['NGC_adapt_non_distance'], {'metric': 'MRR', 'adaptation_step': 5},
                      mode='test', percentage=True, transpose=False)
new_merge_pickle(LogicE_operator_MAML['NGC_adapt_non_distance_2'], list(range(15000, 465000, 15000)),
                 ["step", "adaptation_step", "formula", "metric"], 'test')
new_read_merge_pickle(LogicE_operator_MAML['NGC_adapt_non_distance_2'], {'metric': 'MRR', 'adaptation_step': 5},
                      mode='test', percentage=True, transpose=False)
new_merge_pickle(LogicE_operator_MAML['1p-001-default'], list(range(15000, 465000, 15000)),
                 ["step", "adaptation_step", "formula", "metric"], 'test')
new_read_merge_pickle(LogicE_operator_MAML['1p-001-default'], {'metric': 'MRR', 'step': 15000},
                      mode='test', percentage=True, transpose=False)                      

"""

"""
new_merge_pickle(LogicE_operator_MAML['1p-01-adapt_i'], list(range(15000, 465000, 15000)),
                 ["step", "adaptation_step", "formula", "metric"], 'test')
new_read_merge_pickle(LogicE_operator_MAML['1p-01-adapt_i'], {'metric': 'MRR', 'adaptation_step': 5},
                      mode='test', percentage=True, transpose=False)
new_merge_pickle(NGC_original_for_compare['1p-1496-selected'], list(range(15000, 465000, 15000)),
                 ["step", "formula", "metric"], 'test')
new_read_merge_pickle(NGC_original_for_compare['1p-1496-selected'], {'metric': 'MRR'},
                      mode='test', percentage=True, transpose=False)
for key in to_continue_dict:
    merge_continue_folder(to_continue_dict[key], finish_continue_dict[key], [300000, 450000],
                          list(range(165000, 465000, 15000)), ['valid', 'test'], False)
  
"""
#output_dict = process_output_whole_folder('task_MAML_log', False, False)
#
#aggregate_test('EFO-1_log/NGC_test_urgent', 'LogicE_0001_eval_leaf_lr_0.008')
output_dict = process_output_whole_folder('EFO-1_log/NGC_test_urgent/LogicE_0001_eval_leaf_lr_0.008_aggregated', False, False)