import argparse
import collections
import copy
import os
import inspect
from pprint import pprint
import pandas as pd
import torch
from tqdm.std import trange

from data_helper import TaskManager, BenchmarkFormManager, BenchmarkWholeManager
from fol import BoxEstimator, LogicEstimator, NLKEstimator, BetaEstimator4V, ConEstimator, Meta_LogicEstimator, \
    Meta_ConEstimator, Meta_BetaEstimator
from MetaLearning import maml_train_step, maml_finetuning, operator_level_maml_train_step, \
    operator_level_maml_finetuning
from utils.util import (Writer, load_data_with_indexing, load_task_manager, read_from_yaml,
                        set_global_seed)
from utils.independent_util import count_all_distance, all_normal_form, count_freeze_dumps
from utils.class_util import rename_ordered_dict
from train_test_iteration import train_step, eval_step, eval_step_whole
from benchmark_sampling import sample_finetune_data

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/task_MAML/LogicE_task_MAML_1p-0001.yaml', type=str)


# def training(model, opt, train_iterator, valid_iterator, test_iterator, writer, **train_cfg):
#     lr = train_cfg['learning_rate']
#     with tqdm.trange(train_cfg['steps']) as t:
#         for step in t:
#             log = train_step(model, opt, train_iterator, writer)
#             t.set_postfix({'loss': log['loss']})
#             if step % train_cfg['evaluate_every_steps'] and step > 0:
#                 eval_step(model, valid_iterator, 'valid', writer, **train_cfg)
#                 eval_step(model, test_iterator, 'test', writer, **train_cfg)

#             if step >= train_cfg['warm_up_steps']:
#                 lr /= 5
#                 # logging
#                 opt = torch.optim.Adam(
#                     filter(lambda p: p.requires_grad, model.parameters()),
#                     lr=lr
#                 )
#                 train_cfg['warm_up_steps'] *= 1.5
#             if step % train_cfg['save_every_steps']:
#                 pass
#             if step % train_cfg['log_every_steps']:
#                 pass


def save_eval(log, mode, step, writer):
    for t in log:
        logt = log[t]
        logt['step'] = step
        writer.append_trace(f'eval_{mode}_{t}', logt)


def save_benchmark(log, writer, step, taskmanger: BenchmarkFormManager):
    form_log = collections.defaultdict(lambda: collections.defaultdict(float))
    for normal_form in all_normal_form:
        formula = taskmanger.form2formula[normal_form]
        if formula in log:
            form_log[normal_form] = log[formula]
    writer.save_dataframe(form_log, f"eval_{taskmanger.mode}_{step}_{taskmanger.query_inform_dict['formula_id']}.csv")


def save_whole_benchmark(log, writer, step, whole_task_manager: BenchmarkWholeManager):
    for type_str in whole_task_manager.query_classes:
        save_benchmark(log, writer, step, whole_task_manager.query_classes[type_str])


def save_logging_pickle(log, writer, mode, step):
    writer.save_pickle(log, f"all_logging_{mode}_{step}.pickle")


def load_beta_model(checkpoint_path, model, optimizer):
    print('Loading checkpoint %s...' % checkpoint_path)
    checkpoint = torch.load(os.path.join(
        checkpoint_path, 'checkpoint'))
    init_step = checkpoint['step']
    model.load_state_dict(checkpoint['model_state_dict'])
    current_learning_rate = checkpoint['current_learning_rate']
    warm_up_steps = checkpoint['warm_up_steps']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return current_learning_rate, warm_up_steps, init_step


def load_multiple_net_model(step, checkpoint_path, model, opt, create_multiple_net=False):
    """
    For the possibility of using distance config, load a distance model to a non-distance one will enlarge the latter
    without any influence since the distance2index will not create.
    Load a non-distance model to distance one need to create it here.
    """
    print('Loading checkpoint %s...' % checkpoint_path)
    checkpoint = torch.load(os.path.join(
        checkpoint_path, f'{step}.ckpt'))
    model_parameter = checkpoint['model_parameter']
    opt_parameter = checkpoint['optimizer_parameter']
    model_parameter_key_list = copy.deepcopy(list(model_parameter.keys()))
    for key in model_parameter_key_list:
        if 'projection_net' in key and 'projection_net_' not in key:
            new_key = key.replace('projection_net', 'projection_net_0')
            model_parameter = rename_ordered_dict(model_parameter, key, new_key)
        elif 'center_net' in key and 'center_net_' not in key:
            new_key = key.replace('center_net', 'center_net_0')
            model_parameter = rename_ordered_dict(model_parameter, key, new_key)
    model_parameter_key_list = copy.deepcopy(list(model_parameter.keys()))
    now_model_parameter_key_list = copy.deepcopy(list(model.state_dict().keys()))
    checkpoint_projection_num, checkpoint_conjunction_num = 0, 0
    while True:
        if f'projection_net_{checkpoint_projection_num}.layer0.bias' not in model_parameter:
            break
        checkpoint_projection_num += 1
    while True:
        if f'center_net_{checkpoint_conjunction_num}.layer1.bias' not in model_parameter:
            break
        checkpoint_conjunction_num += 1

    if checkpoint_projection_num + checkpoint_conjunction_num > 2 and not create_multiple_net:
        need_delete_key_index_list = []
        for i in range(1, checkpoint_projection_num + 1):
            for j in range(len(model_parameter_key_list)):
                if f'projection_net_{i}' in model_parameter_key_list[j]:
                    need_delete_key_index_list.append(i)
        for i in range(1, checkpoint_conjunction_num + 1):
            for j in range(len(model_parameter_key_list)):
                if f'center_net_{i}' in model_parameter_key_list[j]:
                    need_delete_key_index_list.append(i)
        for i in need_delete_key_index_list[::-1]:
            del model_parameter[model_parameter_key_list[i]]
            del opt_parameter['state'][i]
        opt_parameter['param_groups'][0]['params'] = list(range(len(model_parameter.keys())))
    elif checkpoint_projection_num + checkpoint_conjunction_num == 2 and create_multiple_net:
        now_parameter_num = len(model_parameter.keys())
        need_rename_key_index_list = []
        opt_reindex_dict = {}
        for j in range(len(model_parameter_key_list)):
            key = model_parameter_key_list[j]
            if 'projection_net_0' in key:
                for i in range(0, model.projection_num):
                    new_key = key.replace('projection_net_0', f'projection_net_{i}')
                    new_index = now_model_parameter_key_list.index(new_key)
                    old_index = model_parameter_key_list.index(key)
                    model_parameter[new_key] = copy.deepcopy(model_parameter[key])
                    opt_reindex_dict[new_index] = copy.deepcopy(opt_parameter['state'][old_index])
            elif 'center_net_0' in key:
                for i in range(0, model.conjunction_num):
                    new_key = key.replace('center_net_0', f'center_net_{i}')
                    new_index = now_model_parameter_key_list.index(new_key)
                    old_index = model_parameter_key_list.index(key)
                    model_parameter[new_key] = copy.deepcopy(model_parameter[key])
                    opt_reindex_dict[new_index] = copy.deepcopy(opt_parameter['state'][old_index])
        for new_index in opt_reindex_dict:
            opt_parameter['state'][new_index] = copy.deepcopy(opt_reindex_dict[new_index])
        opt_parameter['param_groups'][0]['params'] = list(range(len(opt_parameter['state'])))
    model.load_state_dict(model_parameter)
    opt.load_state_dict(opt_parameter)
    learning_rate = checkpoint['learning_rate']
    warm_up_steps = checkpoint['warm_up_steps']
    return learning_rate, warm_up_steps


def load_model(step, checkpoint_path, model, opt):
    full_ckpt_pth = os.path.join(checkpoint_path, f'{step}.ckpt')
    print('Loading checkpoint %s...' % full_ckpt_pth)
    checkpoint = torch.load(full_ckpt_pth)
    model.load_state_dict(checkpoint['model_parameter'])
    opt.load_state_dict(checkpoint['optimizer_parameter'])
    current_learning_rate = checkpoint['learning_rate']
    warm_up_steps = checkpoint['warm_up_steps']
    return current_learning_rate, warm_up_steps


if __name__ == "__main__":

    args = parser.parse_args()
    # parse args and load config
    # configure = read_from_yaml('config/default.yaml')
    configure = read_from_yaml(args.config)
    print("[main] config loaded")
    pprint(configure)
    # initialize my log writer

    if configure['data']['type'] == 'beta':
        case_name = f'{configure.prefix}/{args.config.split("/")[-1].split(".")[0]}'
        # case_name = 'dev/default'
        writer = Writer(case_name=case_name, config=configure, log_path='results/log')
        # writer = SummaryWriter('./logs-debug/unused-tb')
    else:
        if configure['output']['output_path']:  # manually assign the output_path
            case_name = configure['output']['output_path']
        else:
            if 'train' in configure['action']:
                case_name = args.config.split("config")[-1][1:]
            else:  # loading existing checkpoint
                case_name = configure["load"]["checkpoint_path"].split("/")[-1]
        writer = Writer(case_name=case_name, config=configure, log_path=configure["output"]["prefix"])

    # initialize environments
    set_global_seed(configure.get('seed', 0))
    if configure.get('cuda', -1) >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(configure['cuda']))
        # logging.info('Device use cuda: %s' % configure['cuda'])
    else:
        device = torch.device('cpu')

    # prepare the procedure configs
    train_config = configure['train']
    train_config['device'] = device
    eval_config = configure['evaluate']
    eval_config['device'] = device

    # load the data
    print("[main] loading the data")
    data_folder = configure['data']['data_folder']
    entity_dict, relation_dict, projection_train, reverse_projection_train, projection_valid, \
    reverse_projection_valid, projection_test, reverse_projection_test = load_data_with_indexing(data_folder)
    n_entity, n_relation = len(entity_dict), len(relation_dict)

    # get model
    model_name = configure['estimator']['embedding']
    model_params = configure['estimator'][model_name]
    model_params['n_entity'], model_params['n_relation'] = n_entity, n_relation
    model_params['negative_sample_size'] = train_config['negative_sample_size']
    model_params['device'] = device

    """
    Do the distance here
    """
    if train_config['use_distance']:
        distance_config = {'use_distance': train_config['use_distance'],
                           'count_distance_dict': train_config['count_distance_dict']} \
            if train_config['use_distance'] else None
        formula_distance_config, operator_split_dict, contain_formula_dict = count_all_distance(
            configure['evaluate']['formula_id_file'], distance_config, True)
        # todo: the problem is which formula_file to choose from, now, I use the test one.
        all_formula_freeze_dumps = count_freeze_dumps(
            configure['evaluate']['formula_id_file'], configure['evaluate']['interested_normal_forms'],
            formula_distance_config, operator_split_dict, train_config['freeze_other'])
        model_params['projection_num'], model_params['conjunction_num'] = len(operator_split_dict['p']), \
            len(operator_split_dict['i'])
    else:
        distance_config, formula_distance_config, all_formula_freeze_dumps = None, None, None

    if model_name == 'beta':
        if train_config['train_method'] == 'MetaLearning' and configure['MetaLearning']['Algorithm'] == 'operator_MAML':
            model = Meta_BetaEstimator(**model_params)
        else:
            model = BetaEstimator4V(**model_params)
        allowed_norm = ['DeMorgan', 'DNF+MultiIU']
    elif model_name == 'box':
        model = BoxEstimator(**model_params)
        allowed_norm = ['DNF+MultiIU']
    elif model_name == 'logic':
        if train_config['train_method'] == 'MetaLearning' and configure['MetaLearning']['Algorithm'] == 'operator_MAML':
            model = Meta_LogicEstimator(**model_params)
        else:
            model = LogicEstimator(**model_params)
        allowed_norm = ['DeMorgan+MultiI', 'DNF+MultiIU']
    elif model_name == 'NewLook':
        model = NLKEstimator(**model_params)
        model.setup_relation_tensor(projection_train)
        allowed_norm = ['DNF+MultiIUD']
    elif model_name == 'ConE':
        if train_config['train_method'] == 'MetaLearning' and configure['MetaLearning']['Algorithm'] == 'operator_MAML':
            model = Meta_ConEstimator(**model_params)
        else:
            model = ConEstimator(**model_params)
        allowed_norm = ['DeMorgan+MultiI', 'DNF+MultiIU']
    else:
        assert False, 'Not valid model name!'
    model.to(device)

    lr = train_config['learning_rate']
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    init_step = 1
    # exit()
    # assert 2 * train_config['warm_up_steps'] == train_config['steps']

    valid_tm_list, test_tm_list = [], []
    train_path_iterator, train_path_tm, train_other_iterator, train_other_tm, finetune_whole_tm, valid_whole_tm, \
    test_whole_tm = None, None, None, None, None, None, None
    train_iterator, train_tm, valid_iterator, valid_tm, test_iterator, test_tm = None, None, None, None, None, None
    adapt_hyperparameter, finetune_hyperparameter = None, None
    freeze_formula_data, freeze_formula_dict = None, None
    meta_algorithm = None

    if train_config['train_method'] == 'MetaLearning':
        meta_algorithm = configure['MetaLearning']['Algorithm']
        meta_hyperparameter = configure['MetaLearning'][meta_algorithm]
        if meta_algorithm == 'MAML':
            adapt_sig_key = inspect.signature(maml_train_step).parameters.keys()
            finetune_sig_key = inspect.signature(maml_finetuning).parameters.keys()
        elif meta_algorithm == 'operator_MAML':
            adapt_sig_key = inspect.signature(operator_level_maml_train_step).parameters.keys()
            finetune_sig_key = inspect.signature(operator_level_maml_finetuning).parameters.keys()
        else:
            assert False
        adapt_hyperparameter = {key: meta_hyperparameter[key] for key in meta_hyperparameter if key in adapt_sig_key}
        finetune_hyperparameter = {key: meta_hyperparameter[key]
                                   for key in meta_hyperparameter if key in finetune_sig_key}
        if formula_distance_config:
            adapt_hyperparameter['formula_distance_config'] = formula_distance_config
            finetune_hyperparameter['formula_distance_config'] = formula_distance_config
            if meta_algorithm == 'operator_MAML':
                adapt_hyperparameter['all_formula_freeze_dumps'] = all_formula_freeze_dumps
                adapt_hyperparameter['selected_formulas_dict'] = contain_formula_dict
                finetune_hyperparameter['all_formula_freeze_dumps'] = all_formula_freeze_dumps
                finetune_hyperparameter['selected_formulas_dict'] = contain_formula_dict
        if 'valid' in configure['action'] or 'test' in configure['action']:
            if meta_algorithm == 'MAML':
                evaluate_file_path = configure['evaluate']['formula_id_file']
                evaluate_file_data = pd.read_csv(evaluate_file_path)
                finetune_folder = sample_finetune_data(data_folder, evaluate_file_path,
                                                       configure['MetaLearning']['MAML']['support_data_num'],
                                                       projection_train, projection_valid, reverse_projection_valid)
                finetune_whole_tm = BenchmarkWholeManager('finetune', evaluate_file_data, finetune_folder,
                                                          configure['evaluate']['interested_normal_forms'], device,
                                                          model, False, formula_distance_config)
            elif meta_algorithm == 'operator_MAML':
                # in operator_MAML, we can certainly avoid using data not in train data, finetune_data is train_data
                train_formula_id_data = pd.read_csv(configure['train']['formula_id_file'])
                finetune_whole_tm = BenchmarkWholeManager(
                    'train', train_formula_id_data, data_folder, configure['train']['interested_normal_forms'],
                    device, model, False, formula_distance_config)

            finetune_whole_tm.set_up_whole_iteration()

    if configure['data']['type'] == 'beta':
        if 'train' in configure['action']:
            print("[main] load training data")
            beta_path_tasks, beta_other_tasks = [], []
            for task in train_config['meta_queries']:
                if task in ['1p', '2p', '3p']:
                    beta_path_tasks.append(task)
                else:
                    beta_other_tasks.append(task)

            path_tasks = load_task_manager(
                configure['data']['data_folder'], 'train', task_names=beta_path_tasks)
            other_tasks = load_task_manager(
                configure['data']['data_folder'], 'train', task_names=beta_other_tasks)
            if len(beta_path_tasks) > 0:
                train_path_tm = TaskManager('train', path_tasks, device)
                train_path_iterator = train_path_tm.build_iterators(model, batch_size=train_config['batch_size'])
            if len(beta_other_tasks) > 0:
                train_other_tm = TaskManager('train', other_tasks, device)
                train_other_iterator = train_other_tm.build_iterators(model, batch_size=train_config['batch_size'])
            all_tasks = load_task_manager(
                configure['data']['data_folder'], 'train', task_names=train_config['meta_queries'])
            train_tm = TaskManager('train', all_tasks, device)
            train_iterator = train_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])

        if 'valid' in configure['action']:
            print("[main] load valid data")
            tasks = load_task_manager(configure['data']['data_folder'], 'valid',
                                      task_names=configure['evaluate']['meta_queries'])
            valid_tm = TaskManager('valid', tasks, device)
            valid_iterator = valid_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])

        if 'test' in configure['action']:
            print("[main] load test data")
            tasks = load_task_manager(configure['data']['data_folder'], 'test',
                                      task_names=configure['evaluate']['meta_queries'])
            test_tm = TaskManager('test', tasks, device)
            test_iterator = test_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
    elif configure['data']['type'] == 'EFO-1':
        if 'train' in configure['action']:
            train_formula_id_file = configure['train']['formula_id_file']
            train_formula_id_data = pd.read_csv(train_formula_id_file)
            path_formulas_index_list, other_formulas_index_list = [], []
            other_ops = ['i', 'I', 'u', 'U', 'n', 'd', 'D']
            for index in train_formula_id_data.index:
                original_formula = train_formula_id_data['original'][index]
                if True not in [ops in original_formula for ops in other_ops]:
                    path_formulas_index_list.append(index)
                else:
                    other_formulas_index_list.append(index)
            path_formula_id_data = train_formula_id_data.loc[path_formulas_index_list]
            other_formula_id_data = train_formula_id_data.loc[other_formulas_index_list]
            train_path_tm = BenchmarkWholeManager('train', path_formula_id_data, data_folder,
                                                  configure['train']['interested_normal_forms'], device, model, False,
                                                  formula_distance_config)
            if other_formulas_index_list:
                train_other_tm = BenchmarkWholeManager('train', other_formula_id_data, data_folder,
                                                       configure['train']['interested_normal_forms'], device, model,
                                                       False, formula_distance_config)
            freeze_formula_data = pd.read_csv(configure['train']['freeze_formula_file']) \
                if configure['train']['use_freeze'] else None
            freeze_formula_dict = {freeze_formula_data[freeze_formula_data.columns[0]][i]:
                                       freeze_formula_data[freeze_formula_data.columns[1]][i]
                                   for i in range(len(freeze_formula_data.index))} if freeze_formula_data else None
            # todo: need further update

        if 'valid' in configure['action']:
            valid_formula_id_file = configure['evaluate']['formula_id_file']
            valid_formula_id_data = pd.read_csv(valid_formula_id_file)
            valid_whole_tm = BenchmarkWholeManager('valid', valid_formula_id_data, data_folder,
                                                   configure['evaluate']['interested_normal_forms'], device, model,
                                                   False, formula_distance_config)
            """
            for i in valid_formula_id_data.index:
                type_str = valid_formula_id_data['formula_id'][i]
                filename = os.path.join(data_folder, f'valid-{type_str}.csv')
                valid_tm = BenchmarkFormManager('valid', valid_formula_id_data.loc[i], filename, device, model)
                valid_iterator = valid_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
                valid_tm_list.append(valid_tm)
            """

        if 'test' in configure['action']:
            test_formula_id_file = configure['evaluate']['formula_id_file']
            test_formula_id_data = pd.read_csv(test_formula_id_file)
            test_whole_tm = BenchmarkWholeManager('test', test_formula_id_data, data_folder,
                                                  configure['evaluate']['interested_normal_forms'], device, model,
                                                  old_name='benchmark' in data_folder,
                                                  all_distance2index=formula_distance_config)
            '''
            for i in test_formula_id_data.index:
                type_str = test_formula_id_data['formula_id'][i]
                old_filename = os.path.join(data_folder, f'data-{type_str}.csv')
                if os.path.exists(old_filename):
                    test_tm = BenchmarkFormManager('test', test_formula_id_data.loc[i], old_filename, device, model)
                else:
                    filename = os.path.join(data_folder, f'test-{type_str}.csv')
                    test_tm = BenchmarkFormManager('test', test_formula_id_data.loc[i], filename, device, model)
                test_iterator = test_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
                test_tm_list.append(test_tm)
                '''
    else:
        assert False, 'Not valid data type!'
    training_logs = []
    if configure['load']['load_model']:
        checkpoint_path, checkpoint_step = configure['load']['checkpoint_path'], configure['load']['step']
        if checkpoint_step != 0:
            if distance_config and meta_algorithm != 'operator_MAML':
                lr_dict, train_config['warm_up_steps'] = load_multiple_net_model(checkpoint_step, checkpoint_path,
                                                                                 model, opt, True)
            else:
                lr_dict, train_config['warm_up_steps'] = load_model(checkpoint_step, checkpoint_path, model, opt)
            if train_config['train_method'] == 'MetaLearning':
                lr, adapt_hyperparameter['adaptation_lr'], finetune_hyperparameter['finetune_lr'] = \
                    lr_dict['learning_rate'], lr_dict['adaptation_lr'], lr_dict['finetune_lr']
            else:
                lr = lr_dict
            init_step = checkpoint_step + 1  # I think there should be + 1 for train is before then save
        else:
            lr, train_config['warm_up_steps'], init_step = load_beta_model(checkpoint_path, model, opt)
    if configure['data']['type'] == 'EFO-1' and 'train' not in configure['action']:
        assert train_config['steps'] == init_step
    with trange(init_step, train_config['steps'] + 1) as t:
        for step in t:
            # basic training step
            if train_path_tm:
                if step >= train_config['warm_up_steps']:
                    lr /= 5
                    if train_config['train_method'] == 'MetaLearning':
                        if configure['MetaLearning']['shrink_adapt_lr']:
                            adapt_hyperparameter['adaptation_lr'] /= 5
                        if configure['MetaLearning']['shrink_finetune_lr']:
                            finetune_hyperparameter['finetune_lr'] /= 5
                    # logging
                    opt = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=lr
                    )
                    train_config['warm_up_steps'] *= 1.5
                if train_config['train_method'] == 'original':
                    _log = train_step(model, opt, train_path_tm, configure['train']['batch_size'], freeze_formula_dict,
                                      formula_distance_config)
                    if train_other_tm:
                        _log_other = train_step(model, opt, train_other_tm, configure['train']['batch_size'],
                                                freeze_formula_dict, formula_distance_config)
                        _log_second = train_step(model, opt, train_path_tm, configure['train']['batch_size'],
                                                 freeze_formula_dict, formula_distance_config)
                elif train_config['train_method'] == 'MetaLearning':
                    if meta_algorithm == 'MAML':
                        _log = maml_train_step(model, opt, train_path_tm, **adapt_hyperparameter)
                        if train_other_tm:
                            _log_other = maml_train_step(model, opt, train_other_tm, **adapt_hyperparameter)
                            _log_second = maml_train_step(model, opt, train_path_tm, **adapt_hyperparameter)
                    elif meta_algorithm == 'operator_MAML':
                        _log = operator_level_maml_train_step(model, opt, train_path_tm, **adapt_hyperparameter)
                        if train_other_tm:
                            _log_other = operator_level_maml_train_step(model, opt, train_other_tm,
                                                                        **adapt_hyperparameter)
                            _log_second = operator_level_maml_train_step(model, opt, train_path_tm,
                                                                         **adapt_hyperparameter)
                _alllog = {}
                for key in _log:
                    _alllog[f'all_{key}'] = (_log[key] + _log_other[key]) / 2 if train_other_tm else _log[key]
                    _alllog[key] = _log[key]
                _log = _alllog
                t.set_postfix({'loss': _log['loss']})
                training_logs.append(_log)
                if step % train_config['log_every_steps'] == 0:
                    for metric in training_logs[0].keys():
                        _log[metric] = sum(log[metric] for log in training_logs) / len(training_logs)
                    _log['step'] = step
                    training_logs = []
                    writer.append_trace('train', _log)
            if step % train_config['evaluate_every_steps'] == 0 or step == train_config['steps']:
                if configure['data']['type'] == 'beta':
                    '''
                    if train_iterator:
                        train_iterator = train_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
                        _log = eval_step(model, train_iterator, device, mode='train')
                        save_eval(_log, 'train', step, writer)
                    '''
                    if valid_iterator:
                        valid_iterator = valid_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
                        _log = eval_step(model, valid_iterator, mode='valid')
                        save_eval(_log, 'valid', step, writer)

                    if test_iterator:
                        test_iterator = test_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
                        _log = eval_step(model, test_iterator, mode='test')
                        save_eval(_log, 'test', step, writer)
                elif configure['data']['type'] == 'EFO-1':
                    # todo: test_in_train, namely test the train dataset
                    if train_config['train_method'] == 'MetaLearning':
                        if meta_algorithm == 'MAML':
                            if 'valid' in configure['action']:
                                eval_log = maml_finetuning(model, valid_whole_tm, finetune_whole_tm, 'valid',
                                                           **finetune_hyperparameter)
                                # save_whole_benchmark(eval_log, writer, step, valid_whole_tm)
                                # skip the csv storage since the pickle one is more favored.
                                save_logging_pickle(eval_log, writer, 'valid', step)
                            if 'test' in configure['action']:
                                eval_log = maml_finetuning(model, test_whole_tm, finetune_whole_tm, 'test',
                                                           **finetune_hyperparameter)
                                # save_whole_benchmark(eval_log, writer, step, test_whole_tm)
                                save_logging_pickle(eval_log, writer, 'test', step)
                        elif meta_algorithm == 'operator_MAML':
                            if 'valid' in configure['action']:
                                eval_log = operator_level_maml_finetuning(model, valid_whole_tm, finetune_whole_tm,
                                                                          'valid',
                                                                          **finetune_hyperparameter)
                                save_logging_pickle(eval_log, writer, 'valid', step)
                            if 'test' in configure['action']:
                                eval_log = operator_level_maml_finetuning(model, test_whole_tm, finetune_whole_tm,
                                                                          'test',
                                                                          **finetune_hyperparameter)
                                save_logging_pickle(eval_log, writer, 'test', step)
                    else:
                        if 'valid' in configure['action']:
                            eval_log = eval_step_whole(model, valid_whole_tm, 'valid',
                                                       configure['evaluate']['batch_size'], formula_distance_config)
                            # save_whole_benchmark(eval_log, writer, step, valid_whole_tm)
                            save_logging_pickle(eval_log, writer, 'valid', step)
                        if 'test' in configure['action']:
                            eval_log = eval_step_whole(model, test_whole_tm, 'test',
                                                       configure['evaluate']['batch_size'], formula_distance_config)
                            # save_whole_benchmark(eval_log, writer, step, test_whole_tm)
                            save_logging_pickle(eval_log, writer, 'test', step)
                        '''
                        for test_tm in test_tm_list:
                            test_iterator = test_tm.build_iterators(model,
                                                                    batch_size=configure['evaluate']['batch_size'])
                            _log = eval_step(model, test_iterator, mode='test')
                            save_benchmark(_log, writer, step, test_tm)
                                test_iterator =
                                test_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
                                _log_easy = eval_step(model, test_iterator, device, mode='test', allowed_easy_ans=True)
                                for formula in _log_easy:
                                    for metrics in _log_easy[formula]:
                                        _log[formula][f'easy_{metrics}'] = _log_easy[formula][metrics]
                            '''

            if (step % train_config['save_every_steps'] == 0 or step == train_config['steps']) and train_path_tm:
                if train_config['train_method'] == 'MetaLearning':
                    lr_dict = {'learning_rate': lr, 'adaptation_lr': adapt_hyperparameter['adaptation_lr'],
                               'finetune_lr': finetune_hyperparameter['finetune_lr']}
                    writer.save_model(model, opt, step, train_config['warm_up_steps'], lr_dict)
                else:
                    writer.save_model(model, opt, step, train_config['warm_up_steps'], lr)
