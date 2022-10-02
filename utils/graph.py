import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import collections
import numpy as np
import sys
from fol import beta_query_v2, parse_formula, beta_query
from utils.independent_util import all_normal_form
from utils.new_graph import nested_dict_to_array

beta_step = [15000 * i for i in range(1, 21)] + [360000, 420000, 450000]
beta_valid_step = [15000 * i for i in range(1, 21)] + [360000, 420000]

step_dict = {i: beta_step[i] for i in range(len(beta_step))}
inverse_step_dict = {beta_step[i]: i for i in range(len(beta_step))}

all_metrics = ['MRR', 'HITS1', 'HITS3', 'HITS10', 'retrieval_accuracy']
model_supportform_dict = {
    'Beta': ['DeMorgan', 'DeMorgan+MultiI', 'DNF+MultiIU'],
    'Logic': ['DeMorgan', 'DeMorgan+MultiI', 'DNF+MultiIU'],
    'NewLook': ['DNF+MultiIUd', 'DNF+MultiIUD']
}
model_compareform_dict = {
    'Beta': ['original', 'DeMorgan', 'DeMorgan+MultiI', 'DNF', 'DNF+MultiIU'],
    'Logic': ['original', 'DeMorgan', 'DeMorgan+MultiI', 'DNF', 'DNF+MultiIU'],
    'NewLook': ['diff', 'DNF+diff', 'DNF+MultiIUd', 'DNF+MultiIUD']
}

beta_query_DNF = {
    '1p': '(p,(e))',
    '2p': '(p,(p,(e)))',
    '3p': '(p,(p,(p,(e))))',
    '2i': '(i,(p,(e)),(p,(e)))',
    '3i': '(I,(p,(e)),(p,(e)),(p,(e)))',
    'ip': '(p,(i,(p,(e)),(p,(e))))',
    'pi': '(i,(p,(e)),(p,(p,(e))))',
    '2in': '(i,(n,(p,(e))),(p,(e)))',
    '3in': '(I,(n,(p,(e))),(p,(e)),(p,(e)))',
    'inp': '(p,(i,(n,(p,(e))),(p,(e))))',
    'pin': '(i,(n,(p,(e))),(p,(p,(e))))',
    'pni': '(i,(n,(p,(p,(e)))),(p,(e)))',
    '2u-DNF': '(u,(p,(e)),(p,(e)))',
    'up-DNF': '(u,(p,(p,(e))),(p,(p,(e))))',
    '2u-DM': '(n,(i,(n,(p,(e))),(n,(p,(e)))))',
    'up-DM': '(p,(n,(i,(n,(p,(e))),(n,(p,(e))))))'
}
all_formula_data = pd.read_csv('data/test_generated_formula_anchor_node=3.csv')


def print_loss(path):
    """
    Open train.csv and print loss change w.r.t step
    """
    data_file = os.path.join(path, 'train.csv')
    df = pd.read_csv(data_file)
    loss = np.asarray(df['loss'])
    step = np.asarray(df['step'])
    loss = np.log(loss)
    plt.plot(step, loss)
    plt.ylabel('loss')
    plt.xlabel('step')
    plt.show()


def compare_loss(data_file, data_2, choose_len=None, use_name1='loss', use_name2='loss', make_comparison=False):
    df, df2 = pd.read_csv(data_file), pd.read_csv(data_2)
    loss, loss2 = np.asarray(df[use_name1]), np.asarray(df2[use_name2])
    step = np.asarray(df['step'])
    minlen = min(len(loss), len(loss2))
    if choose_len:
        loss = loss[:choose_len]
        loss2 = loss2[:choose_len]
        step = step[:choose_len]
    if len(loss) > minlen:
        loss = loss[:minlen]
    else:
        loss2 = loss2[:minlen]

    compare = np.log(loss) - np.log(loss2)
    if make_comparison:
        plt.plot(step, compare)
        plt.plot(step, np.zeros_like(compare), color='r')
    else:
        plt.plot(step, loss, label='MAML')
        plt.plot(step, loss2, color='r', label='original')
    plt.legend()
    plt.ylabel('MRR')
    plt.xlabel('step')
    plt.show()


def log_all_metrics(path, step, mode, log_meta_formula=beta_query_v2.values()):
    """
    old logging used for 'eval_test_(p,(e)).csv' like
    """
    log = collections.defaultdict(lambda: collections.defaultdict(float))

    for meta_formula in log_meta_formula:
        # if meta_formula != 'p(e)|p(e)' and meta_formula != 'p(p(e)|p(e))':
        foq_instance = parse_formula(meta_formula)
        foq_formula = foq_instance.formula
        data_file = os.path.join(path, f'eval_{mode}_{foq_formula}.csv')
        df = pd.read_csv(data_file)
        step_range = np.asarray(df['step'])
        step_index = np.where(step_range == step)[0]
        for metric in df.columns:
            if metric != 'step':
                log[metric][foq_formula] = df[metric][step_index].values[0]
    averaged_metric = {}
    averaged_my_formula = [parse_formula(formula).formula for formula in log_meta_formula]
    for metric in log:
        averaged_metric[metric] = \
            sum([log[metric][foq_formula] for foq_formula in averaged_my_formula]) / len(averaged_my_formula)
    all_data = pd.DataFrame.from_dict(log)
    all_data.to_csv(os.path.join(path, f'eval_{mode}_{step}_average.csv'))
    print(all_data)
    print(averaged_metric)


'''
def log_old_metrics(path, step, mode, log_meta_formula=beta_query.values()):
    log = collections.defaultdict(lambda: collections.defaultdict(float))

    for meta_formula in log_meta_formula:
        # if meta_formula != 'p(e)|p(e)' and meta_formula != 'p(p(e)|p(e))':
        foq_instance = parse_foq_formula(meta_formula)
        foq_formula = foq_instance.meta_formula
        data_file = os.path.join(path, f'eval_{mode}_{foq_formula}.csv')
        df = pd.read_csv(data_file)
        step_range = np.asarray(df['step'])
        step_index = np.where(step_range == step)[0]
        for metric in df.columns:
            if metric != 'step':
                log[metric][foq_formula] = df[metric][step_index].values[0]
    averaged_metric = {}
    averaged_my_formula = [parse_foq_formula(formula).meta_formula for formula in log_meta_formula]
    for metric in log:
        averaged_metric[metric] = \
            sum([log[metric][foq_formula] for foq_formula in averaged_my_formula]) / len(averaged_my_formula)
    all_data = pd.DataFrame.from_dict(log)
    all_data.to_csv(os.path.join(path, f'eval_{mode}_{step}_average.csv'))
    print(all_data)
    print(averaged_metric)
'''


def read_beta_log(path, mode='test', chosen_step=None, averaged_meta_formula=beta_query_v2.values()):
    """
    Brute force reading logging for beta, returns the dict from formula, metrics to a list of scores (steps)
    """
    train_log = collections.defaultdict(lambda: collections.defaultdict(float))
    valid_log = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    test_log = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    beta_valid = collections.defaultdict(lambda: collections.defaultdict(list))
    beta_test = collections.defaultdict(lambda: collections.defaultdict(list))
    beta_log_path = os.path.join(path, 'train.log')
    with open(beta_log_path, 'r') as f:
        for line in f.readlines():
            if line[29:50] == 'Training average loss':
                info = line[58:]
                step, score = info.split(':')
                step, score = eval(step), eval(score)
                train_log['loss'][step] = score
            elif line[29:54] == 'Training average positive':
                info = line[75:]
                step, score = info.split(':')
                step, score = eval(step), eval(score)
                train_log['positive_loss'][step] = score
            elif line[29:54] == 'Training average negative':
                info = line[75:]
                step, score = info.split(':')
                step, score = eval(step), eval(score)
                train_log['negative_loss'][step] = score
            elif line[29:35] == 'Valid ':
                info = line[35:].split(' ')
                beta_name, metric, step, score = info[0], info[1], eval(info[4][:-1]), eval(info[5])
                if beta_name in beta_query_v2:
                    foq_instance = parse_formula(beta_query_v2[beta_name])
                    foq_formula = foq_instance.formula
                    valid_log[step][metric][foq_formula] = score
                    beta_valid[foq_formula][metric].append(score)
            elif line[29:34] == 'Test ' and line[34:38] != 'info':
                info = line[34:].split(' ')
                beta_name, metric, step, score = info[0], info[1], eval(info[4][:-1]), eval(info[5])
                if beta_name in beta_query_v2:
                    foq_instance = parse_formula(beta_query_v2[beta_name])
                    foq_formula = foq_instance.formula
                    test_log[step][metric][foq_formula] = score
                    beta_test[foq_formula][metric].append(score)
    train_data = pd.DataFrame.from_dict(train_log)
    train_data.to_csv(os.path.join(path, 'beta_train.csv'))
    # print(pd.DataFrame.from_dict(valid_log[chosen_step]))
    for step in eval(f'{mode}_log'):
        valid_data = pd.DataFrame.from_dict(valid_log[step])
        valid_data.to_csv(os.path.join(path, f'beta_valid_{step}.csv'))
        test_data = pd.DataFrame.from_dict(test_log[step])
        test_data.to_csv(os.path.join(path, f'beta_test_{step}.csv'))
    if chosen_step is not None:
        print(pd.DataFrame.from_dict(test_log[chosen_step]))
    else:
        print(test_data)
    averaged_metric = {}
    averaged_my_formula = [parse_formula(formula).formula for formula in averaged_meta_formula]
    for metric in test_log[15000]:
        if chosen_step is not None:
            averaged_metric[metric] = sum([test_log[chosen_step][metric][foq_formula]
                                           for foq_formula in averaged_my_formula]) / len(averaged_meta_formula)
    print(averaged_metric)
    return train_log, beta_valid, beta_test


def read_logic_log(path, mode='test', chosen_step=None, averaged_meta_formula=beta_query_v2.values()):
    """
    Brute force reading logging for logic, returns the dict from formula, metrics to a list of scores (steps)
    """
    train_log = collections.defaultdict(lambda: collections.defaultdict(float))
    valid_log = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    test_log = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    logic_valid = collections.defaultdict(lambda: collections.defaultdict(list))
    logic_test = collections.defaultdict(lambda: collections.defaultdict(list))
    beta_log_path = os.path.join(path, 'train.log')
    with open(beta_log_path, 'r') as f:
        for line in f.readlines():
            if line[29:50] == 'Training average loss':
                info = line[58:]
                step, score = info.split(':')
                step, score = eval(step), eval(score)
                train_log['loss'][step] = score
            elif line[29:61] == 'Training average positive_sample':
                info = line[75:]
                step, score = info.split(':')
                step, score = eval(step), eval(score)
                train_log['positive_loss'][step] = score
            elif line[29:61] == 'Training average negative_sample':
                info = line[75:]
                step, score = info.split(':')
                step, score = eval(step), eval(score)
                train_log['negative_loss'][step] = score
            elif line[29:35] == 'Valid ':
                info = line[35:].split(' ')
                beta_name, metric, step, score = info[0], info[1], eval(info[4][:-1]), eval(info[5])
                if beta_name in beta_query_v2:
                    foq_instance = parse_formula(beta_query_v2[beta_name])
                    foq_formula = foq_instance.formula
                    valid_log[step][metric][foq_formula] = score
                    logic_valid[foq_formula][metric].append(score)
            elif line[29:34] == 'Test ' and line[34:38] != 'info':
                info = line[34:].split(' ')
                beta_name, metric, step, score = info[0], info[1], eval(info[4][:-1]), eval(info[5])
                if beta_name in beta_query_v2:
                    foq_instance = parse_formula(beta_query_v2[beta_name])
                    foq_formula = foq_instance.formula
                    test_log[step][metric][foq_formula] = score
                    logic_test[foq_formula][metric].append(score)
    train_data = pd.DataFrame.from_dict(train_log)
    train_data.to_csv(os.path.join(path, 'beta_train.csv'))
    # print(pd.DataFrame.from_dict(valid_log[chosen_step]))
    for step in eval(f'{mode}_log'):
        valid_data = pd.DataFrame.from_dict(valid_log[step])
        valid_data.to_csv(os.path.join(path, f'logic_valid_{step}.csv'))
        test_data = pd.DataFrame.from_dict(test_log[step])
        test_data.to_csv(os.path.join(path, f'logic_test_{step}.csv'))
    if chosen_step is not None:
        print(pd.DataFrame.from_dict(test_log[chosen_step]))
    else:
        print(test_data)
    averaged_metric = {}
    averaged_my_formula = [parse_formula(formula).formula for formula in averaged_meta_formula]
    for metric in test_log[15000]:
        if chosen_step is not None:
            averaged_metric[metric] = sum([test_log[chosen_step][metric][foq_formula]
                                           for foq_formula in averaged_my_formula]) / len(averaged_meta_formula)
    print(averaged_metric)
    return train_log, logic_valid, logic_test


def plot_comparison(beta_log, my_log, all_formula):
    # metric in ['MRR', 'HITS1', 'HITS3', 'HITS10']:
    """
    compare metric difference for two logging dict.
    """
    for metric in ['MRR']:
        for meta_formula in all_formula:
            foq_instance = parse_formula(beta_query_v2[meta_formula])
            foq_formula = foq_instance.formula
            beta_score = np.asarray(beta_log[foq_formula][metric])
            my_score = np.asarray(my_log[foq_formula][metric])
            n = len(my_score)
            beta_plot_step = np.asarray(beta_step)[:n]
            plt.plot(beta_plot_step, beta_score[:n], color='red', label=f'{meta_formula}_beta')
            plt.plot(beta_plot_step, my_score, linestyle=':', color='blue', label=f'{meta_formula}_ours')
        plt.title(all_formula)
        plt.legend()
        plt.show()


def comparison(path, all_meta_formula):
    our_train = pd.read_csv(os.path.join(path, 'train.csv'))
    my_valid = collections.defaultdict(lambda: collections.defaultdict(list))
    my_test = collections.defaultdict(lambda: collections.defaultdict(list))
    beta_train, beta_valid, beta_test = read_beta_log(path)
    for mode in ['valid', 'test']:
        for meta_formula in all_meta_formula:
            foq_instance = parse_formula(beta_query_v2[meta_formula])
            foq_formula = foq_instance.formula
            df = pd.read_csv(os.path.join(path, f'eval_{mode}_{foq_formula}.csv'))
            for metric in df.columns:
                if metric != 'step' and metric != 'num_queries':
                    for i in range(len(df[metric])):
                        eval(f'my_{mode}')[foq_formula][metric].append(df[metric][i])
        # plot_comparison(eval(f'beta_{mode}'), eval(f'my_{mode}'), ['1p', '2p', '3p'], mode)
        # plot_comparison(eval(f'beta_{mode}'), eval(f'my_{mode}'), ['2i', '3i'], mode)
        plot_comparison(eval(f'beta_{mode}'), eval(f'my_{mode}'), ['1p', '2p', '2i'])
        plot_comparison(eval(f'beta_{mode}'), eval(f'my_{mode}'), ['3p', '3i'])


def typestr2benchmarkname(type_str: str, mode=None, step=None):
    if mode:
        return f'eval_{mode}_{step}_{type_str}.csv'
    else:
        return f'eval_{type_str}.csv'


def log_benchmark(folder_path, id_list, typestr2filename, percentage=False, mode=None, step=None):
    """
    The first function to run for reading logging.
    Calculate scores for all formulas in a file for a given metric.
    It allows for given mode and step or take it as None as the benchmark setting.
    All_log : metric, normal_form, id -> score
    """
    all_log = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    for task_id in id_list:
        type_str = f'type{task_id:04d}'
        filename = typestr2filename(type_str, mode, step)
        # real_index = all_formula.loc[all_formula['formula_id'] == f'type{id_str}'].index[0]
        if os.path.exists(os.path.join(folder_path, filename)):
            single_log = pd.read_csv(os.path.join(folder_path, filename))
            index2metrics = single_log['Unnamed: 0']
            for normal_form in single_log.columns:
                if normal_form != 'Unnamed: 0':
                    for index in range(len(single_log[normal_form])):
                        if percentage and index2metrics[index] != 'num_queries':
                            all_log[index2metrics[index]][normal_form][task_id] = single_log[normal_form][index] * 100
                        else:
                            all_log[index2metrics[index]][normal_form][task_id] = single_log[normal_form][index]
        else:
            print(f'Warnings: {os.path.join(folder_path, filename)} not exists.')
    for metric in all_log:
        data_metric = pd.DataFrame.from_dict(all_log[metric])
        if mode:
            data_metric.to_csv(os.path.join(folder_path, f'all_formula_{mode}_{step}_{metric}.csv'))
            print(f'save to all_formula_{mode}_{step}_{metric}.csv')
        else:
            data_metric.to_csv(os.path.join(folder_path, f'all_formula_{metric}.csv'))
            print(f'save to all_formula_{metric}.csv')
    return all_log


def normal_form_comparison(folder_path, form1, form2, metrics, save_csv=False, percentage=False):
    """
    Do comparison between two given forms for all metric in metrics.
    This should be used after log_benchmark.
    """
    unequal_task = set()
    form1_log, form2_log = collections.defaultdict(lambda: collections.defaultdict(float)), \
                           collections.defaultdict(lambda: collections.defaultdict(float))
    comparison_log = collections.defaultdict(list)
    for metric in metrics:
        metric_logging = pd.read_csv(os.path.join(folder_path, f'all_formula_{metric}.csv'))
        index2taskid = metric_logging['Unnamed: 0']
        for index in range(len(index2taskid)):
            taskid = index2taskid[index]
            id_str = '0' * (4 - len(str(taskid))) + str(taskid)
            formula_index = all_formula_data.loc[all_formula_data['formula_id'] == f'type{id_str}'].index[0]
            formula1, formula2 = all_formula_data[form1][formula_index], all_formula_data[form2][formula_index]
            score1, score2 = metric_logging[form1][index], metric_logging[form2][index]
            if formula1 != formula2 and str(score1) != 'nan' and str(score2) != 'nan':
                # what if two scores are same
                if taskid not in unequal_task:
                    assert metric == metrics[0]
                    unequal_task.add(taskid)
                form1_log[metric][taskid], form2_log[metric][taskid] = score1, score2
    if len(unequal_task) > 0:
        for metric in metrics:
            averaged1, averaged2 = sum(form1_log[metric][taskid] for taskid in form1_log[metric]) / \
                                   len(form1_log[metric]), \
                                   sum(form2_log[metric][taskid] for taskid in form2_log[metric]) / \
                                   len(form2_log[metric])
            comparison_log[metric] = [averaged1, averaged2]
    else:
        for metric in metrics:
            comparison_log[metric] = [0, 0]
    form1_win_rate = sum(form1_log['MRR'][taskid] > form2_log['MRR'][taskid] for taskid in unequal_task)
    form2_win_rate = sum(form1_log['MRR'][taskid] < form2_log['MRR'][taskid] for taskid in unequal_task)
    comparison_log['win_rate'] = [form1_win_rate, form2_win_rate]
    comparison_log['different_queries'] = [len(unequal_task), len(unequal_task)]
    if save_csv:
        compare_taskid = {}
        for metric in metrics:
            compare_taskid[f'{form1}_{metric}'] = form1_log[metric]
            compare_taskid[f'{form2}_{metric}'] = form2_log[metric]
        compare_taskid[f'{form1}_formula'] = {}
        compare_taskid[f'{form2}_formula'] = {}
        compare_taskid['winner'] = {}
        for taskid in unequal_task:
            id_str = '0' * (4 - len(str(taskid))) + str(taskid)
            formula_index = all_formula_data.loc[all_formula_data['formula_id'] == f'type{id_str}'].index[0]
            formula1, formula2 = all_formula_data[form1][formula_index], all_formula_data[form2][formula_index]
            compare_taskid[f'{form1}_formula'][taskid] = formula1
            compare_taskid[f'{form2}_formula'][taskid] = formula2
            if compare_taskid[f'{form1}_MRR'][taskid] > compare_taskid[f'{form2}_MRR'][taskid]:
                compare_taskid['winner'][taskid] = form1
            elif compare_taskid[f'{form1}_MRR'][taskid] < compare_taskid[f'{form2}_MRR'][taskid]:
                compare_taskid['winner'][taskid] = form2
            else:
                compare_taskid['winner'][taskid] = 'draw'
        data = pd.DataFrame.from_dict(compare_taskid)
        data.to_csv(os.path.join(folder_path, f'compare_detail_{form1}_{form2}.csv'))

    '''
    df = pd.DataFrame.from_dict(comparison_log)
    df.to_csv(os.path.join(folder_path, f'compare_{form1}_{form2}.csv'))
    '''
    return comparison_log


def compare_all_form(folder_path, form_list, metrics, save_csv=False):
    """
    Do pairwise comparison in all normal form in form_list.
    This should be used after log_benchmark.
    """
    difference_mrr = collections.defaultdict(lambda: collections.defaultdict(list))
    difference_number = collections.defaultdict(lambda: collections.defaultdict(int))
    difference_win_rate = collections.defaultdict(lambda: collections.defaultdict(float))
    n = len(form_list)
    for i in range(n):
        for j in range(n):
            difference_number[form_list[j]][form_list[i]] = 0
            difference_win_rate[form_list[j]][form_list[i]] = 0
    for i in range(n):
        for j in range(i + 1, n):
            comparison_log = normal_form_comparison(folder_path, form_list[i], form_list[j], metrics, save_csv)
            difference_mrr[form_list[j]][form_list[i]] = comparison_log['MRR']
            difference_number[form_list[j]][form_list[i]] = comparison_log['different_queries'][0]
            difference_number[form_list[i]][form_list[j]] = comparison_log['different_queries'][0]
            formj_win, formi_win = comparison_log['win_rate']
            if formj_win + formi_win > 0:
                j_against_i = formj_win / (formj_win + formi_win) * 100
                difference_win_rate[form_list[j]][form_list[i]] = j_against_i
                difference_win_rate[form_list[i]][form_list[j]] = 100 - j_against_i
            else:
                difference_win_rate[form_list[j]][form_list[i]] = 0
                difference_win_rate[form_list[i]][form_list[j]] = 0

    dm, dn, dw = pd.DataFrame.from_dict(difference_mrr), pd.DataFrame.from_dict(difference_number), \
                 pd.DataFrame.from_dict(difference_win_rate)
    dm.to_csv(os.path.join(folder_path, f'allmrr_compare.csv'))
    dn.to_csv(os.path.join(folder_path, f'alllength_compare.csv'))
    dw.to_csv(os.path.join(folder_path, f'allwin_rate_compare.csv'))


def log_benchmark_depth_anchornode(folder_path, support_normal_forms, metrics):
    """
    Statistic for groupings of anchor node and depth.
    This should be used after log_benchmark.
    """
    query_type_num = len(all_formula_data['original'])
    all_logging = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    averaged_split = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    averaged_all = collections.defaultdict(lambda: collections.defaultdict(float))
    for normal_form in support_normal_forms:
        for i in range(1, 4):
            for j in range(1, 4):
                for metric in metrics:
                    averaged_split[normal_form][(i, j)][metric] = 0
    for metric in metrics:
        metric_logging = pd.read_csv(os.path.join(folder_path, f'all_formula_{metric}.csv'))
        index2taskid = metric_logging['Unnamed: 0']
        for index in range(len(index2taskid)):
            taskid = index2taskid[index]
            id_str = '0' * (4 - len(str(taskid))) + str(taskid)
            formula_index = all_formula_data.loc[all_formula_data['formula_id'] == f'type{id_str}'].index[0]
            depth = all_formula_data['original_depth'][formula_index]
            anchornode_num = all_formula_data['num_anchor_nodes'][formula_index]
            for normal_form in support_normal_forms:
                query_scores = metric_logging.loc[index][normal_form]
                all_logging[normal_form][(anchornode_num, depth)][metric].append(query_scores)
    all_number = sum(len(all_logging[support_normal_forms[0]][key][metrics[0]])
                     for key in all_logging[support_normal_forms[0]])
    assert all_number == query_type_num  # all query type are included
    for normal_form in support_normal_forms:
        for key in all_logging[normal_form]:
            for metric in metrics:
                averaged_split[normal_form][key][metric] = sum(all_logging[normal_form][key][metric]) \
                                                           / len(all_logging[normal_form][key][metric])
    for normal_form in support_normal_forms:
        for metric in metrics:
            averaged_all[normal_form][metric] = sum(sum(all_logging[normal_form][key][metric])
                                                    for key in all_logging[normal_form])
            averaged_all[normal_form][metric] /= query_type_num
            averaged_split[normal_form]['average'][metric] = averaged_all[normal_form][metric]
        df = pd.DataFrame.from_dict(averaged_split[normal_form])
        if normal_form != 'DNF+MultiIUd':
            df.to_csv(os.path.join(folder_path, f'anchornode_depth_of_{normal_form}.csv'))
        else:
            df.to_csv(os.path.join(folder_path, f'anchornode_depth_of_new_form.csv'))
    return averaged_split


def answer_statistic(data_folder, formula_id_file):
    """
    Statistic just for the number of answer set.
    """
    formula_id_data = pd.read_csv(formula_id_file)
    query_id_str_list = formula_id_data['formula_id']
    statistis_grouping = collections.defaultdict(list)
    statistis_grouping_averaged = collections.defaultdict(lambda: collections.defaultdict(float))
    for i in range(1, 4):
        for j in range(1, 4):
            statistis_grouping[(i, j)] = []
    for type_str in query_id_str_list:
        filename = os.path.join(data_folder, f'data-{type_str}.csv')
        dense = filename.replace('data', 'tmp').replace('csv', 'pickle')
        if os.path.exists(dense):
            print("load from existed files", type_str)
            with open(dense, 'rb') as f:
                data = pickle.load(f)
                easy_answer_set = data['easy_answer_set']
                hard_answer_set = data['hard_answer_set']
                easy_ans_num, hard_ans_num = sum(len(easy) for easy in easy_answer_set) / len(easy_answer_set), \
                                             sum(len(hard) for hard in hard_answer_set) / len(hard_answer_set)
                formula_index = formula_id_data.loc[formula_id_data['formula_id'] == f'{type_str}'].index[0]
                depth = formula_id_data['original_depth'][formula_index]
                anchor_node_num = formula_id_data['num_anchor_nodes'][formula_index]
                statistis_grouping[(anchor_node_num, depth)].append(hard_ans_num)
        else:
            query_data = pd.read_csv(filename)
            all_easy_ans, all_hard_ans = query_data.easy_answers.map(lambda x: list(eval(x))).tolist(), \
                                         query_data.hard_answers.map(lambda x: list(eval(x))).tolist()
            easy_ans_num, hard_ans_num = sum(len(easy) for easy in all_easy_ans) / len(all_easy_ans), \
                                         sum(len(hard) for hard in all_hard_ans) / len(all_hard_ans)
            formula_index = formula_id_data.loc[formula_id_data['formula_id'] == f'{type_str}'].index[0]
            depth = formula_id_data['original_depth'][formula_index]
            anchor_node_num = formula_id_data['num_anchor_nodes'][formula_index]
            statistis_grouping[(anchor_node_num, depth)].append(hard_ans_num)

    for key in statistis_grouping:
        statistis_grouping_averaged[key]['hard'] = sum(statistis_grouping[key]) / len(statistis_grouping[key])
        print(key, len(statistis_grouping[key]))
    data_averaged = pd.DataFrame.from_dict(statistis_grouping_averaged)
    data_averaged.to_csv(os.path.join(data_folder, 'size_statistics_grouping_formhard.csv'))


def compare_over_steps(step_list, folder_path, id_list, typestr2filename, normal_form, type_id,
                       percentage=False, mode=None):
    all_step_log = collections.defaultdict(lambda: collections.defaultdict(float))
    for step in step_list:
        step_log = log_benchmark(folder_path, id_list, typestr2filename, percentage, mode, step)
        for metric in step_log:
            all_step_log[metric][step] = step_log[metric][normal_form][type_id]
    step_data = pd.DataFrame.from_dict(all_step_log)
    step_data.to_csv(os.path.join(folder_path, f'step_change_for_{type_id}_{normal_form}.csv'))
    return all_step_log


def compare_models(formula_file, file1, file2, output_folder, chosen_form=None):
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)
    formula_data = pd.read_csv(formula_file)
    assert data1.columns == data2.columns
    for normal_form in all_normal_form:
        subtract = collections.defaultdict(list)
        for taskid in data1[normal_form]:
            type_str = f'type{taskid:04d}'
            corresponding_formula = formula_data[normal_form][type_str]
            subtract[taskid].append(corresponding_formula)
            subtract[taskid].append(data1[normal_form][taskid] - data2[normal_form][taskid])
        subtract_data = pd.from_dict(subtract)
        subtract_data.to_csv(os.path.join(output_folder, 'compare'))


def merge_pickle(folder_path, step_list, mode_list=('valid', 'test')):
    """
    all log will be a nested list
    according to the order of [mode, step, formula, metric] : formula is

    valid_log, test_log will be a list of dict indicating the index to the name
    """
    all_log = [[], []]
    valid_log_index = [{i: step_list[i] for i in range(len(step_list))}]
    test_log_index = [{i: step_list[i] for i in range(len(step_list))}]
    index_name_list = ['step', 'formula', 'metric']
    for mode_id in range(len(mode_list)):
        for step_id in range(len(step_list)):
            mode, step = mode_list[mode_id], step_list[step_id]
            filename = f'all_logging_{mode}_{step}.pickle'
            with open(os.path.join(folder_path, filename), 'rb') as f:
                single_log = pickle.load(f)
                new_log_list = []
                if step_id == 0:
                    new_formula_index = {}
                    new_metric_index = {}
                for formula_id, formula in enumerate(single_log):
                    new_log_list.append([])
                    if step_id == 0:
                        new_formula_index[formula_id] = formula
                    if step_id == 0 and formula_id == 0:
                        for metric_id, metric in enumerate(single_log[formula]):
                            new_metric_index[metric_id] = metric
                    for metric_id, metric in enumerate(single_log[formula]):
                        new_log_list[formula_id].append(single_log[formula][metric])
                if mode_list[mode_id] == 'valid':
                    all_log[0].append(new_log_list)
                else:
                    all_log[1].append(new_log_list)
        if mode_list[mode_id] == 'valid':
            valid_log_index.append(new_formula_index)
            valid_log_index.append(new_metric_index)
        else:
            test_log_index.append(new_formula_index)
            test_log_index.append(new_metric_index)
    with open(os.path.join(folder_path, 'merge_logging.pickle'), 'wb') as f:
        pickle.dump([all_log, valid_log_index, test_log_index, index_name_list], f)


def read_merge_pickle(folder_path, fixed_dict, percentage=False, mode='test'):
    with open(os.path.join(folder_path, 'merge_logging.pickle'), 'rb') as f:
        single_log = pickle.load(f)
        all_log, valid_log_index, test_log_index, index_name_list = single_log
        valid_log, test_log = all_log
        if mode == 'valid':
            considered_log = np.asarray(valid_log)
            used_log_index = valid_log_index
        else:
            considered_log = np.asarray(test_log)
            used_log_index = test_log_index

        selected_index_list = []
        for i, index_name in enumerate(index_name_list):
            if index_name in fixed_dict:
                name2index = {used_log_index[i][index]: index for index in used_log_index[i]}
                selected_index_list.append(name2index[fixed_dict[index_name]])
            else:
                selected_index_list.append(slice(len(used_log_index[i].keys())))
        selected_log = considered_log[tuple(selected_index_list)]

        # re-indexing by using the step/formula/metric name
        left_key_binary_index = np.asarray([(key not in fixed_dict) for key in index_name_list])
        left_key_index = np.where(left_key_binary_index == 1)[0]
        left_key_name_list = [index_name_list[int(left_key_single_index)] for left_key_single_index in left_key_index]
        left_key_dict_list = [used_log_index[int(left_key_single_index)] for left_key_single_index in left_key_index]
        for i, left_key_dict in enumerate(left_key_dict_list):
            assert len(left_key_dict) == selected_log.shape[i]

        if len(left_key_dict_list) == 1:
            left_key_dict, left_key_name = left_key_dict_list[0], left_key_name_list[0]
            reindexed_log = {left_key_dict[i]: selected_log[i] for i in range(len(left_key_dict))}
            reindexed_data = pd.DataFrame.from_dict(reindexed_log, orient='index', columns=[str(fixed_dict)])
            reindexed_data.to_csv(os.path.join(folder_path, f'selected_log_{left_key_name}.csv'))
        elif len(left_key_dict_list) == 2:
            reindexed_data = pd.DataFrame(
                data=selected_log, index=left_key_dict_list[0].values(), columns=left_key_dict_list[1].values())
            reindexed_data.to_csv(
                os.path.join(folder_path, f'selected_log_{mode}_{left_key_name_list[0]}_{left_key_name_list[1]}.csv'))
        else:
            print('Do not left more than 2 indices unfixed.')


def select_form(experiment_folder, experiment_file, chose_form, formula_file=None):
    """
    chosen form can be one of the normal form, also can be best which means the best normal form according to experiment
    result.
    chose_form can also be 'best' which means namely give the best one.
    """
    # todo:guess need update
    experiment_formula_data = pd.read_csv(os.path.join(experiment_folder, experiment_file))
    experiment_formula_dict = {experiment_formula_data[experiment_formula_data.columns[0]][i]:
                                   experiment_formula_data[experiment_formula_data.columns[1]][i]
                               for i in range(len(experiment_formula_data.index))}
    if formula_file:
        formula_data = pd.read_csv(formula_file)
    else:
        formula_data = all_formula_data
    output_dict = {}
    if chose_form != 'best':
        consider_formulas = formula_data[chose_form]
        for formula in consider_formulas:
            output_dict[formula] = experiment_formula_dict[formula]
    else:
        for query_type in formula_data.index:
            best_result = 0
            best_formula = None
            for possible_formula in formula_data.loc[query_type]:
                if possible_formula in experiment_formula_dict:
                    if experiment_formula_dict[possible_formula] > best_result:
                        best_result = experiment_formula_dict[possible_formula]
                        best_formula = possible_formula
            output_dict[f'{query_type}_{best_formula}'] = best_result
    output_data = pd.DataFrame.from_dict(output_dict, orient='index')
    output_data.to_csv(os.path.join(experiment_folder, f'chose_form_{chose_form}.csv'))



box_query_v2 = {
    '1p': '(p,(e))',
    '2p': '(p,(p,(e)))',
    '3p': '(p,(p,(p,(e))))',
    '2i': '(i,(p,(e)),(p,(e)))',
    '3i': '(i,(p,(e)),(p,(e)),(p,(e)))',
    'ip': '(p,(i,(p,(e)),(p,(e))))',
    'pi': '(i,(p,(p,(e))),(p,(e)))',
    '2u-DNF': '(u,(p,(e)),(p,(e)))',
    'up-DNF': '(u,(p,(p,(e))),(p,(p,(e))))',
}

check_query = {
    '1p': '(p,(e))',
    '2p': '(p,(p,(e)))',
    '3p': '(p,(p,(p,(e))))',
    '2i': '(i,(p,(e)),(p,(e)))',
    '3i': '(i,(p,(e)),(p,(e)),(p,(e)))',
    'ip': '(p,(i,(p,(e)),(p,(e))))',
    'pi': '(i,(p,(p,(e))),(p,(e)))',
    '2in': '(i,(p,(e)),(n,(p,(e))))',
    '3in': '(i,(p,(e)),(p,(e)),(n,(p,(e))))',
    'inp': '(p,(i,(p,(e)),(n,(p,(e)))))',
    'pin': '(i,(p,(p,(e))),(n,(p,(e))))',
    'pni': '(i,(n,(p,(p,(e)))),(p,(e)))',
    '2u-DNF': '(u,(p,(e)),(p,(e)))',
    'up-DNF': '(u,(p,(p,(e))),(p,(p,(e))))',
    '2u-DM': '(n,(i,(n,(p,(e))),(n,(p,(e)))))',
    'up-DM': '(p,(n,(i,(n,(p,(e))),(n,(p,(e))))))',
}
DNF_query = {
    '1p': '(p,(e))',
    '2p': '(p,(p,(e)))',
    '3p': '(p,(p,(p,(e))))',
    '2i': '(i,(p,(e)),(p,(e)))',
    '3i': '(i,(p,(e)),(p,(e)),(p,(e)))',
    'ip': '(p,(i,(p,(e)),(p,(e))))',
    'pi': '(i,(p,(p,(e))),(p,(e)))',
    '2in': '(i,(p,(e)),(n,(p,(e))))',
    '3in': '(i,(p,(e)),(p,(e)),(n,(p,(e))))',
    'inp': '(p,(i,(p,(e)),(n,(p,(e)))))',
    'pin': '(i,(p,(p,(e))),(n,(p,(e))))',
    'pni': '(i,(n,(p,(p,(e)))),(p,(e)))',
    '2u-DNF': '(u,(p,(e)),(p,(e)))',
    'up-DNF': '(u,(p,(p,(e))),(p,(p,(e))))',
}
# print_loss(graph_path)
'''
test_step = 450000
test_path = "/home/hyin/FirstOrderQueryEstimation/log/newdev/Logic-unbounded210813.22:26:062c614d51/"
old_path = "/home/hyin/FirstOrderQueryEstimation/log/newdev/Logic-unbounded210813.21:19:26aaf6eebf/"
# test_path = "/home/hyin/FirstOrderQueryEstimation/log/dev/default210705.14:43:26fba267b0/"
logic_path = "/data/zwanggc/Logic-unbounded210813.22:24:17607989e2/"
#compare_loss(test_path, test_path, choose_len=3000)
log_all_metrics(test_path, test_step, 'test', log_meta_formula=check_query.values())
log_all_metrics(old_path, test_step, 'test', log_meta_formula=check_query.values())
'''
p_list = [0, 1, 2, 1116, 1117]
i_list = [13, 137, 1113, 1114]
beta_list = list(range(0, 14))
all_3_3_list = list(range(0, 531))
benchmark_FB15k_237 = {
    'Beta': "/home/zwanggc/FirstOrderQueryEstimation/benchmark_log/benchmark_FB15k-237/Beta_full211021.21:53:5622b7307f/",
    'NLK': "/home/zwanggc/FirstOrderQueryEstimation/benchmark_log/benchmark_FB15k-237/NLK_full211022.10:23:213a1fea21/",
    'Logic': "/home/hyin/FirstOrderQueryEstimation/benchmark_log/benchmark_FB15k-237/Logic_full211022.14:06:57d7bd0d37/",
    # 2335
    'Logic_new': "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/benchmark_FB15k-237/Logic_unbounded210816.19:16"
                 ":29590496f9220327.21:28:44255e5888",  # 2335
    'Box': "/home/zwanggc/FirstOrderQueryEstimation/benchmark_log/benchmark_FB15k-237/Box_full210822.00:56:4448dc3a71"
}

benchmark_NELL = {
    'Beta': "/home/zwanggc/FirstOrderQueryEstimation/benchmark_log/benchmark_NELL/Beta_full211021.21:54:2510bcf310/",
    'Logic': "/home/hyin/FirstOrderQueryEstimation/benchmark_log/benchmark_NELL/Logic_full211022.14:06:0128fd7614/",
    # 2335
    'NLK': "/home/zwanggc/FirstOrderQueryEstimation/benchmark_log/benchmark_NELL/NLK_full211022.14:23:059a6b9d86/",
}

benchmark_FB15k = {
    'Beta': "/home/zwanggc/FirstOrderQueryEstimation/benchmark_log/benchmark_FB15k/Beta_full211021.21:52:5760fc2d24/",
    'Logic': "/home/zwanggc/FirstOrderQueryEstimation/benchmark_log/benchmark_FB15k/Logic_full211022.10:14:23940a46a4/",
    'NLK': "/home/zwanggc/FirstOrderQueryEstimation/benchmark_log/benchmark_FB15k/NLK_full211022.14:22:1240d958f1/"
}

benchmark_FB15k_237_Logic_generalize = {   #  all in 2335
    '1p': "/home/hyin/FirstOrderQueryEstimation/benchmark_log/benchmark_generalize/Logic_1p210825.15:55:2565735b8d",
    '2p': "/home/hyin/FirstOrderQueryEstimation/benchmark_log/benchmark_generalize/Logic_2p210825.16:02:51b8e4878b",
    '3p': "/home/hyin/FirstOrderQueryEstimation/benchmark_log/benchmark_generalize/Logic_3p210825.16:07:530d917424",
    '2i': "/home/hyin/FirstOrderQueryEstimation/benchmark_log/benchmark_generalize/Logic_2i210825.16:26:241f438fbf",
    '2i_new': "/home/hyin/FirstOrderQueryEstimation/benchmark_log/benchmark_generalize/Logic_2i210828.17:53:31146141ce",
    '3i': "/home/hyin/FirstOrderQueryEstimation/benchmark_log/benchmark_generalize/Logic_3i210825.16:44:0584f53968",
    '3i_new': "/home/hyin/FirstOrderQueryEstimation/benchmark_log/benchmark_generalize/Logic_3i210828.17:56:27662c4441"
}

FB15_237_data, FB_data, NELL_data = 'data/benchmark/FB15k-237', 'data/benchmark/FB15k', 'data/benchmark/NELL'

EFO1_LogicE_train_path = {
    'zwanggc': "/data/zwanggc/FirstOrderQueryEstimation/EFO-1_log/test_EFO-1_train/EFO-1_LogicE211210.17:52:3681317f8a",
    '2335': "/home/hyin/FirstOrderQueryEstimation/log/newdev/Logic_unbounded210816.19:16:29590496f9",
    '2335_new': "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/EFO-1_train/EFO-1_LogicE220327.18:54:25b1e65a2e"
}


MAML_LogicE = {  # all in 2335
    'original': "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/test_MAML/MAML_LogicE220105.11:06:165b6b16ac",
    'shrink_adapt': "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/new_test_MAML/MAML_LogicE_valid_1000220116.23:38:0810b52787",
    'shrink_adapt_finetune': "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/new_test_MAML/MAML_LogicE_valid_1000220117.00:06:14def8dee2",
    'small_batch': "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/MAML_small_batch/MAML_LogicE_train_500220130.10:58:004d1fcf5f"
}

EFO1_Logic_comparison_path = {
    'finetune_included': "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/finetune_included/EFO-1_LogicE_finetune_included220209.10:19:57088c1f07"
}

benchmark_MAML_LogicE_path = {  # all in 2335
    'original': "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/benchmark_MAML/MAML_LogicE_valid_1000220116.23:38:0810b52787220125.22:42:41b1bfc68b",
    'original_new':  "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/benchmark_FB15k-237/MAML_LogicE_valid_1000220117.00:06:14def8dee2220327.23:03:450c1f3865",
    # finetune_lr = 0.002 which means both lr shrinks
    'shrink_adapt_not_finetune': "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/benchmark_MAML/MAML_LogicE_valid_1000220116.23:38:0810b52787220125.22:43:0395080a4f",
    # finetune_lr = finetune_lr = 0.01   shrink adapt but not shrink finetune
    'no_finetune': '/home/hyin/FirstOrderQueryEstimation/EFO-1_log/benchmark_MAML/MAML_LogicE_valid_1000220116.23:38:0810b52787220126.23:34:2463d70505',
    'no_finetune_new': '/home/hyin/FirstOrderQueryEstimation/EFO-1_log/benchmark_FB15k-237/MAML_LogicE_valid_1000220117.00:06:14def8dee2220328.22:14:34bf4575ba',
    # simply do not finetune

}


MAML_small_not_shrink_adapt = "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/MAML_small_batch/MAML_LogicE_train_500220213.21:53:26a2adad28"
MAML_small_both_not_shrink = "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/MAML_small_batch/MAML_LogicE_train_500220213.21:54:25de0d58cf"
MAML_3w = "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/MAML_small_batch/MAML_LogicE_train_500220218.22:44:358ceb73c1"

ConE_1234 = "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/EFO-1_log/EFO-1_ConE220305.17:56:43c62ff4b7"

MAML_ConE_002_no_shrink = "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/MAML_log/MAML_ConE220309.23:12:5130c3edcd"
MAML_ConE_001_no_shrink = "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/MAML_log/MAML_ConE220310.22:22:24357674b4"
MAML_ConE_0005_no_shrink = "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/MAML_log/MAML_ConE220316.22:39:33f1a4a781"
#  The first two have the problem that the finetune lr is one fifth of the adaptation lr, not a quarter of.
MAML_ConE_0004_shrink = '/home/hyin/FirstOrderQueryEstimation/EFO-1_log/MAML_log/MAML_ConE220324.22:30:405901ef8d'

MAML_LogicE_02 = "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/MAML_log/MAML_LogicE220310.22:31:4643f803b2"
MAML_LogicE_01 = "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/MAML_log/MAML_LogicE220310.22:32:273a09a380"
MAML_LogicE_004 = "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/MAML_log/MAML_LogicE220310.22:34:5930f44f07"
# The above three is in 2335 and all use both_shrink setting

LogicE_selected_pure = "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/MAML_log/EFO-1_LogicE_selected220320.22:49:42bf4992cf"
MAML_LogicE_selected = "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/MAML_log/MAML_LogicE_selected220320.22:47:39f40ce77f"

LogicE_minimal_selected = "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/selected_train/EFO-1_LogicE_minimal_selected220329.23:09:060ba09e7b"

ConE_selected = '/home/zwanggc/FirstOrderQueryEstimation/EFO-1_log/selected_train_log/EFO-1_ConE_selected220323.18:52:16de598922'
MAML_ConE_0004_selected = '/home/hyin/FirstOrderQueryEstimation/EFO-1_log/selected_train/MAML_ConE_selected220324.23:31:4055bf1414'

BetaE_selected = '/home/hyin/FirstOrderQueryEstimation/EFO-1_log/selected_train/EFO-1_BetaE_selected220403.23:58:131e109866'

LogicE_naive_distance = "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/EFO-1_distance/LogicE_distance220605.22:58:05a569f023"
LogicE_leaf = "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/EFO-1_distance/LogicE_distance220605.23:03:0378f67e15"

LogicE_only_p = {
    'original': "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/EFO-1_train/LogicE_p220619.20:42:2267989526",
    'original_0004': "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/EFO-1_train/LogicE_p220608.23:46:338f95221c",
    'root': "EFO-1_log/EFO-1_distance_only_p/LogicE_distance_p220605.23:19:45e79ba4ca",
    'leaf': "EFO-1_log/EFO-1_distance_only_p/LogicE_distance_p220606.00:37:392f74d254",
    'root_0002': "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/EFO-1_distance_only_p/LogicE_distance_p220615.18:41:17cef3b188",
    'leaf_0002': "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/EFO-1_distance_only_p/LogicE_distance_p220615.18:41:5674c8fda7",
    'root_0003': "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/EFO-1_distance_only_p/LogicE_distance_p220613.14:07:430b7a8c32",
    'leaf_0003': "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/EFO-1_distance_only_p/LogicE_distance_p220613.14:10:132d58d711",
    'root_0005': "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/EFO-1_distance_only_p/LogicE_distance_p220612.11:58:26d9f88e1b",
    'leaf_0005': "/home/hyin/FirstOrderQueryEstimation/EFO-1_log/EFO-1_distance_only_p/LogicE_distance_p220612.11:58:074663c990"
}
LogicE_original_checkpoint_to_distance = {
    '35w': '/home/hyin/FirstOrderQueryEstimation/EFO-1_log/EFO-1_checkpoint2distance/30w220618.23:24:34cf3e039f',
    '20w': '/home/hyin/FirstOrderQueryEstimation/EFO-1_log/EFO-1_checkpoint2distance/30w220618.23:24:059d028227',
    '30w': '/home/hyin/FirstOrderQueryEstimation/EFO-1_log/EFO-1_checkpoint2distance/30w220618.23:24:272e4fd991',
    '25w': '/home/hyin/FirstOrderQueryEstimation/EFO-1_log/EFO-1_checkpoint2distance/30w220618.23:24:1849c931e0'
}


# log_benchmark(MAML_LogicE_path, beta_list, typestr2benchmarkname, False, 'test', '450000')


# compare_all_form(Box_path, all_normal_form, all_metrics)
# compare_all_form(Logic_path, model_compareform_dict['LogicE'], metrics=all_metrics, save_csv=True)
#  pandas_logging_depth_anchornode(NELL_result, model_supportform_dict, all_metrics)
# log_benchmark_depth_anchornode(Logic_path, model_supportform_dict['LogicE'], all_metrics)
# answer_statistic(NELL_data, formula_file)
# MAML_shrink_log = log_benchmark(MAML_LogicE_shrink_finetune_path, beta_list, typestr2benchmarkname, False, 'test', 450000)
# log_all_metrics(LogicE_train_path, 450000, 'test', beta_query_DNF.values())
# compare_over_steps(list(range(0, 465000, 15000)), finetune_included, beta_list, typestr2benchmarkname, 'DNF+MultiIU',
#                   0, False, 'test')

# compare_loss(os.path.join(finetune_included, 'step_change_for_0_DNF+MultiIU.csv'), os.path.join(MAML_small, 'step_change_for_0_DNF+MultiIU.csv'), 4100,
#              'MRR', 'MRR')

# new_merge_pickle(test_full_formula['EFO-1'], list(range(0, 20, 1)), ["step", "formula", "adaptation_step", "metric"], 'valid')

# merge_pickle(Logic_new_path, list(range(450000, 465000, 15000)), mode_list=['test'])
# merge_pickle(LogicE_only_p['root_0003'], list(range(15000, 465000, 15000)), mode_list=['test'])
# merge_pickle(LogicE_only_p['leaf_0003'], list(range(15000, 465000, 15000)), mode_list=['test'])
# merge_pickle(benchmark_MAML_no_finetune_new, list(range(450000, 465000, 15000)), mode_list=['test'])
# merge_pickle(Logic_new_path, list(range(450000, 465000, 15000)), ['test'])
# merge_pickle(benchmark_MAML_no_finetune, list(range(450000, 465000, 15000)))
# merge_pickle(MAML_small_not_shrink_adapt, list(range(15000, 465000, 15000)))
# merge_pickle(MAML_small_both_not_shrink, list(range(15000, 465000, 15000)))

# read_merge_pickle(Logic_new_path, {'metric': 'MRR', 'step': 450000}, percentage=False, mode='test')
# read_merge_pickle(LogicE_only_p['root_0003'], {'metric': 'MRR', 'step': 450000}, percentage=False, mode='test')
# read_merge_pickle(LogicE_only_p['leaf_0003'], {'metric': 'MRR', 'step': 450000}, percentage=False, mode='test')
# read_merge_pickle(benchmark_MAML_LogicE_new, {'metric': 'MRR', 'step': 450000}, percentage=False, mode='test')
# read_merge_pickle(benchmark_MAML_no_finetune_new, {'metric': 'MRR', 'step': 450000}, percentage=False, mode='test')
# read_merge_pickle(benchmark_MAML_no_finetune, {'metric': 'MRR'})
# log_old_metrics(old_path, test_step, 'test')
# train_all, valid_all, test_all = read_beta_log('../download_log/full/')
# train_part, valid_part, test_part = read_logic_log(logic_path, 'test', test_step, averaged_meta_formula=DNF_query.values())
# pickle_select_form(Logic_new_path, 450000, ["step", "formula", "metric"], {'metric': 'MRR', 'step': 450000}, 'DNF+MultiIU')
# select_form()
# comparison('../download_log/1p.2p.2i/', ['1p', '2p', '2i'])
