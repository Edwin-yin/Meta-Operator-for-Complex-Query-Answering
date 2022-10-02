import collections
import random
from copy import deepcopy
from typing import Any, List
from collections import OrderedDict, defaultdict
import math
import numpy as np

import torch
import torch.nn.functional as F

from utils.class_util import fixed_depth_nested_dict
from fol import AppFOQEstimator
from fol.appfoq import compute_final_loss
from data_helper import BenchmarkWholeManager
from train_test_iteration import eval_batch_query


def maml_train_step(model, optimizer, data_manager: BenchmarkWholeManager, adaptation_lr, adaptation_step, k_support,
                    k_query, batch_size: int, first_order: bool = False, test_outer_loss: bool = True,
                    formula_distance_config=None):
    # The support and query split in the MAML
    num_tasks = len(data_manager.formula2id)
    results = {
        'inner_losses': np.zeros((num_tasks, adaptation_step), dtype=np.float32),
        'outer_losses': np.zeros((num_tasks, adaptation_step + 1), dtype=np.float32),
        'mean_outer_loss': 0.
    }
    mean_outer_loss = torch.tensor(0., device=model.device)
    for i, full_formula in enumerate(data_manager.formula2id):
        all_data = data_manager.get_rawdata_single_task(batch_size, full_formula)
        real_data_len = len(all_data['batch_indices'])
        support_num = math.ceil(k_support / (k_support + k_query) * real_data_len)
        support_raw_data, query_raw_data = {}, {}
        for key in all_data:
            support_raw_data[key], query_raw_data[key] = all_data[key][:support_num], all_data[key][support_num:]
        # do not need permutation since have already done that
        adaptation_params, adaptation_results = adapt(model, support_raw_data, query_raw_data, full_formula,
                                                      data_manager, {}, adaptation_step, adaptation_lr, first_order,
                                                      test_outer_loss, formula_distance_config)
        query_emb = data_manager.index2emb(model, full_formula, query_raw_data['batch_indices'],
                                           None if not formula_distance_config
                                           else formula_distance_config[full_formula], None, adaptation_params)
        outer_loss = compute_single_task_loss(model, query_emb, query_raw_data['answer_set'], full_formula,
                                              adaptation_params)
        mean_outer_loss += outer_loss
        results['inner_losses'][i] = adaptation_results['inner_losses']
        results['outer_losses'][i][:adaptation_step] = adaptation_results['outer_losses']
        results['outer_losses'][i][adaptation_step] = outer_loss.item()
    mean_outer_loss /= num_tasks
    optimizer.zero_grad()
    mean_outer_loss.backward()

    optimizer.step()
    log = {'loss': mean_outer_loss.item(),
           'loss_inner_initial': np.mean(results['inner_losses'][:, 0]),
           'loss_inner_final': np.mean(results['inner_losses'][:, -1]),
           'loss_outer_initial': np.mean(results['outer_losses'][:, 0]),
           'loss_outer_final': np.mean(results['outer_losses'][:, -1])}
    return log


def maml_finetuning(model, test_data_manager: BenchmarkWholeManager, finetune_data_manager: BenchmarkWholeManager, mode,
                    finetune_step, finetune_lr, support_data_num, test_batch_size: int, first_order: bool = False,
                    test_within_adaptation: bool = False, formula_distance_config=None):
    test_data_manager.set_up_whole_iteration()  # set up iteration for every time of test
    test_within_adaptation = test_within_adaptation and mode == 'valid'

    logs = fixed_depth_nested_dict(float, 3) if test_within_adaptation else fixed_depth_nested_dict(float, 2)
    for full_formula in test_data_manager.formula2id:
        finetune_data = finetune_data_manager.get_rawdata_single_task(support_data_num, full_formula)
        # finetune first using the train data_manager
        if test_within_adaptation:
            adaptation_params = None
            for step in range(finetune_step):
                adaptation_params, adaptation_results = adapt(model, finetune_data, {}, full_formula,
                                                              finetune_data_manager, adaptation_params, 1, finetune_lr,
                                                              first_order, False, formula_distance_config)
                formula_log = test_single_formula(model, test_data_manager, mode, full_formula, adaptation_params,
                                                  test_batch_size, formula_distance_config)
                for metric in formula_log:
                    logs[full_formula][step][metric] = formula_log[metric]
        else:
            adaptation_params, adaptation_results = adapt(model, finetune_data, {}, full_formula, finetune_data_manager,
                                                          {}, finetune_step, finetune_lr, first_order, False,
                                                          formula_distance_config)
            formula_log = test_single_formula(model, test_data_manager, mode, full_formula, adaptation_params,
                                              test_batch_size, formula_distance_config)
            logs[full_formula] = formula_log
    return logs


def compute_single_task_loss(model, embeddings, answer_set_list, formula, meta_parameters) -> torch.float:
    positive_logit, negative_logit, subsampling_weight = model.criterion(
        embeddings, answer_set_list, False, 'u' in formula or 'U' in formula, meta_parameters)
    positive_loss, negative_loss = compute_final_loss(positive_logit, negative_logit, subsampling_weight)
    loss = (positive_loss + negative_loss) / 2
    return loss


def adapt(model: AppFOQEstimator, support_data: dict, query_data: dict, full_formula: str,
          data_manager: BenchmarkWholeManager, params=None, num_adaptation_steps=1, step_size=0.1,
          first_order: bool = False, test_outer_loss: bool = True, formula_distance_config=None):
    """
    Doing adaptation based on support_data. I also test the outer loss without gradient.
    params is the initial parameter.
    """
    results = {'inner_losses': np.zeros((num_adaptation_steps,), dtype=np.float32),
               'outer_losses': np.zeros((num_adaptation_steps,), dtype=np.float32)}
    for step in range(num_adaptation_steps):
        if test_outer_loss:
            with torch.no_grad():
                emb = data_manager.index2emb(model, full_formula, query_data['batch_indices'],
                                             None if not formula_distance_config
                                             else formula_distance_config[full_formula], params or {})
                loss_query: torch.Tensor = compute_single_task_loss(model, emb, query_data['answer_set'], full_formula,
                                                                    params or {})
                results['outer_losses'][step] = loss_query.item()
        support_emb = data_manager.index2emb(model, full_formula, support_data['batch_indices'],
                                             None if not formula_distance_config
                                             else formula_distance_config[full_formula], None, params or {})
        inner_loss = compute_single_task_loss(model, support_emb, support_data['answer_set'], full_formula,
                                              params or {})
        results['inner_losses'][step] = inner_loss.item()
        model.zero_grad()
        if not params:
            params = OrderedDict()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    params[name] = param
        grads = torch.autograd.grad(inner_loss, params.values(), create_graph=model.training and not first_order,
                                    allow_unused=True)
        # todo: I'm not sure why in the fine-tuning we don't need to calculate the high order gradient.
        updated_params = OrderedDict()
        if isinstance(step_size, (dict, OrderedDict)):
            for (name, param), grad in zip(params.items(), grads):
                updated_params[name] = param if grad is None else param - step_size[name] * grad
        else:
            for (name, param), grad in zip(params.items(), grads):
                updated_params[name] = param if grad is None else param - step_size * grad
        params = updated_params
    return params, results


def test_single_formula(model, test_data_manager: BenchmarkWholeManager, mode, full_formula, adaptation_params,
                        test_batch_size: int, formula_distance_config=None):
    single_formula_log = defaultdict(float)
    single_task_manager = test_data_manager.query_classes[
        test_data_manager.full_formula_to_type_str[full_formula]].tasks[full_formula]
    single_task_manager.restart()  # Need to make sure the index is 0
    num_of_batch = math.ceil(single_task_manager.length / test_batch_size)
    with torch.no_grad():
        for batch_id in range(num_of_batch):
            batch_test_data = test_data_manager.get_rawdata_single_task(test_batch_size, full_formula, False)
            if mode == 'train':
                easy_ans_list = [[] for _ in range(len(batch_test_data['batch_indices']))]
                hard_ans_list = batch_test_data['answer_set']
            else:
                easy_ans_list = batch_test_data['easy_answer_set']
                hard_ans_list = batch_test_data['hard_answer_set']
            batch_pred_emb = test_data_manager.index2emb(model, full_formula, batch_test_data['batch_indices'],
                                                         None if not formula_distance_config else
                                                         formula_distance_config[full_formula], None, adaptation_params)
            batch_log = eval_batch_query(model, batch_pred_emb, full_formula, easy_ans_list, hard_ans_list, False,
                                         adaptation_params)
            for batch_metric in batch_log:
                single_formula_log[batch_metric] += batch_log[batch_metric]
    for log_metric in single_formula_log.keys():
        if log_metric != 'num_queries':
            single_formula_log[log_metric] /= single_formula_log['num_queries']
    return single_formula_log
