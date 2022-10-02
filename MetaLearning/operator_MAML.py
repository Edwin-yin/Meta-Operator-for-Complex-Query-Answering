import collections
import copy
import random
from copy import deepcopy
from typing import Any, List
from collections import OrderedDict, defaultdict
import numpy as np

import torch

from fol.appfoq import compute_final_loss
from data_helper import BenchmarkWholeManager
from train_test_iteration import eval_batch_query, data_manager_output2loss, eval_step_whole
from .MAML import compute_single_task_loss


def operator_level_maml_train_step(model, optimizer, data_manager: BenchmarkWholeManager, adaptation_lr,
                                   adaptation_step, k_support, k_query, batch_size: int, all_formula_freeze_dumps,
                                   operator_params_dict, selected_formulas_dict, first_order, weight_decay, momentum,
                                   test_outer_loss: bool = True, formula_distance_config=None,
                                   adapt_non_distance=False):
    support_num = int(k_support / (k_support + k_query) * batch_size)
    query_num = batch_size - support_num
    model.eval()  # Do not train BatchNormalization in inner_loop
    all_final_params, all_inner_loss = operator_level_adapt(
        model, data_manager, adaptation_step, adaptation_lr, support_num, all_formula_freeze_dumps,
        formula_distance_config, operator_params_dict, selected_formulas_dict, first_order, weight_decay, momentum,
        False, adapt_non_distance)
    model.train()
    if test_outer_loss:
        with torch.no_grad():
            outer_data_initial = data_manager.get_whole_batch_data(model, query_num, False, None,
                                                                   None, None)
            _, _, outer_loss_initial = data_manager_output2loss(outer_data_initial, model, None)
    final_params = fill_up_distance_meta_parameters(all_final_params[adaptation_step], model, all_formula_freeze_dumps,
                                                    operator_params_dict)
    outer_data = data_manager.get_whole_batch_data(model, query_num, True, None, formula_distance_config, final_params)
    _, _, outer_loss = data_manager_output2loss(outer_data, model, None)
    optimizer.zero_grad()
    outer_loss.backward()
    optimizer.step()
    all_inner_loss = np.array(all_inner_loss)
    log = {
        'loss': outer_loss.item(),
        'inner_loss_initial': np.mean(all_inner_loss[:, 0]),
        'inner_loss_final': np.mean(all_inner_loss[:, -1]),
    }
    if test_outer_loss:
        log['outer_loss_initial'] = outer_loss_initial.item()
    return log


def operator_level_maml_finetuning(model, test_data_manager: BenchmarkWholeManager,
                                   finetune_data_manager: BenchmarkWholeManager, mode,
                                   finetune_step, finetune_lr, support_data_num, test_batch_size: int,
                                   all_formula_freeze_dumps, operator_params_dict, selected_formulas_dict, first_order,
                                   weight_decay, momentum, test_within_adaptation: bool = False,
                                   formula_distance_config=None, adapt_non_distance: bool = False):
    model.eval()
    torch.cuda.empty_cache()
    test_data_manager.set_up_whole_iteration()  # set up iteration for every time of test
    # test_within_adaptation = test_within_adaptation and mode == 'valid'
    # TODO: how to use the finetune data, all formula equally 50 seems more reasonable
    final_params, all_inner_loss = operator_level_adapt(
        model, finetune_data_manager, finetune_step, finetune_lr, support_data_num, all_formula_freeze_dumps,
        formula_distance_config, operator_params_dict, selected_formulas_dict, first_order, weight_decay, momentum,
        True, adapt_non_distance)
    # in the finetune time, it's common to use detach
    if test_within_adaptation:
        adapt_log = {}
        for i in range(finetune_step + 1):
            use_meta_parameters = fill_up_distance_meta_parameters(final_params[i], model, all_formula_freeze_dumps,
                                                                   operator_params_dict)
            adapt_log[i] = eval_step_whole(model, test_data_manager, mode, test_batch_size, formula_distance_config,
                                           all_formula_freeze_dumps, use_meta_parameters)
        return adapt_log
    else:
        use_meta_parameters = fill_up_distance_meta_parameters(final_params[finetune_step], model,
                                                               all_formula_freeze_dumps, operator_params_dict)
        logs = eval_step_whole(model, test_data_manager, mode, test_batch_size, formula_distance_config,
                               all_formula_freeze_dumps, use_meta_parameters)
        return logs


def fill_up_distance_meta_parameters(unfilled_parameters, model, all_formula_freeze_dumps, operator_params_dict):
    """
    For meta-learning, consider the case that meta_parameters didn't collect all params in original model.
    Fill up the meta_parameters dict here.
    """
    now_saved_parameters = set()
    all_operator_prefix = defaultdict(list)
    all_parameter_dict = OrderedDict(model.named_parameters())
    for operator_name in all_formula_freeze_dumps:
        for operator_index in all_formula_freeze_dumps[operator_name]:
            if f'{operator_name}_{operator_index}' in unfilled_parameters:
                now_saved_parameters.update(unfilled_parameters[f'{operator_name}_{operator_index}'].keys())
                all_operator_prefix[operator_name].append(operator_index)
    #  Do not use p_0 to fill p_1 /p_2, use p_prototype to fill p_1 /p_2
    for operator_name in all_operator_prefix:
        chosen_params = unfilled_parameters[f'{operator_name}_{all_operator_prefix[operator_name][0]}']
        for operator_index in all_formula_freeze_dumps[operator_name]:
            if operator_index not in all_operator_prefix[operator_name]:
                if f'{operator_name}_{operator_index}' not in unfilled_parameters:
                    unfilled_parameters[f'{operator_name}_{operator_index}'] = {}
                for name in chosen_params:
                    unfilled_parameters[f'{operator_name}_{operator_index}'][name] = all_parameter_dict[name]
    #  If there's no p/i/e/r at all
    for name, param in model.named_parameters():
        if param.requires_grad and name not in now_saved_parameters:
            for operator_name in operator_params_dict:
                if operator_params_dict[operator_name] in name:
                    for operator_index in all_formula_freeze_dumps[operator_name]:
                        if f'{operator_name}_{operator_index}' not in unfilled_parameters:
                            unfilled_parameters[f'{operator_name}_{operator_index}'] = {}
                        unfilled_parameters[f'{operator_name}_{operator_index}'][name] = param
                    break
            else:  # must be non_distance embedding, mostly e,r embedding
                unfilled_parameters[name] = param
    return unfilled_parameters


def fill_up_meta_parameters(unfilled_parameters, model):
    for name, param in model.named_parameters():
        if param.requires_grad and name not in unfilled_parameters:
            unfilled_parameters[name] = param
    return unfilled_parameters


def operator_level_adapt(model, adapt_data_manager: BenchmarkWholeManager, adapt_step, adapt_lr, adapt_batch_num,
                         all_formula_freeze_dumps, formula_distance_config, operator_params_dict,
                         selected_formulas_dict, first_order, weight_decay, momentum, detach,
                         adapt_non_distance: bool = False):
    final_params = {step: {} for step in range(adapt_step + 1)}
    all_saved_params_name = set()
    all_inner_loss = []
    all_parameter_dict = OrderedDict(model.named_parameters())
    # multiple forward means multiple negative_sampling for the reason that it's done in model.criterion

    for i, operator_name in enumerate(all_formula_freeze_dumps):
        for j, operator_type in enumerate(all_formula_freeze_dumps[operator_name]):
            if set(selected_formulas_dict[operator_name][operator_type]).intersection(adapt_data_manager.formula2id):
                for step in range(adapt_step + 1):
                    final_params[step][f'{operator_name}_{operator_type}'] = {}
                now_inner_loss = []
                now_adapt_params_name = set()
                params = OrderedDict()
                for name, param in all_parameter_dict.items():
                    if param.requires_grad:
                        if operator_params_dict[operator_name] in name:
                            params[name] = param
                            all_saved_params_name.add(name)
                            now_adapt_params_name.add(name)
                mom_buffer = None
                for step in range(adapt_step):
                    to_adapt_params = OrderedDict({name: params[name] for name in now_adapt_params_name})
                    final_params[step][f'{operator_name}_{operator_type}'].update(to_adapt_params)
                    filled_params = fill_up_meta_parameters(params, model)
                    params, inner_loss, mom_buffer = inner_adapt(model, adapt_data_manager, adapt_batch_num, adapt_lr,
                                                     to_adapt_params, filled_params, False,
                                                     all_formula_freeze_dumps[operator_name][operator_type],
                                                     None,   # use None in distance_config ensures net_index=-1
                                                     selected_formulas_dict[operator_name][operator_type],
                                                     first_order, weight_decay, momentum, mom_buffer, detach)
                    now_inner_loss.append(inner_loss)
                all_inner_loss.append(now_inner_loss)
                final_params[adapt_step][f'{operator_name}_{operator_type}'] = params
    if adapt_non_distance:
        # Take a forward to compute the params for non-distance operators.
        params = OrderedDict()
        now_adapt_params_name = []
        now_inner_loss = []
        for name, param in all_parameter_dict:
            if param.requires_grad and name not in all_saved_params_name:
                params[name] = param
                now_adapt_params_name.append(name)
        mom_buffer = None
        for step in range(adapt_step):
            filled_params = fill_up_meta_parameters(params, model)
            to_adapt_params = OrderedDict({name: filled_params[name] for name in now_adapt_params_name})
            final_params[step].update(to_adapt_params)
            params, inner_loss, mom_buffer = inner_adapt(model, adapt_data_manager, adapt_batch_num, adapt_lr, to_adapt_params,
                                             filled_params, True, None, None, None, first_order, weight_decay,
                                             momentum, mom_buffer, detach)
            now_inner_loss.append(inner_loss)
        all_inner_loss.append(now_inner_loss)
        final_params[adapt_step].update(params)
    else:  # update data index here if not adapt_non_distance
        adapt_data_manager.simply_update_index(adapt_batch_num, None)
    return final_params, all_inner_loss


def inner_adapt(model, adapt_data_manager: BenchmarkWholeManager, adapt_batch_num, adapt_lr, to_adapt_params,
                filled_params, update_data_index, freeze_dumps, distance_config, selected_formulas,
                first_order, weight_decay, momentum,
                mom_buffer, detach):
    """
    Inner single step adaptation given to_adapt_params (namely given those params to update).
    Empower it by adding weight_decay and momentum.
    """
    data = adapt_data_manager.get_whole_batch_data(model, adapt_batch_num, update_data_index, freeze_dumps,
                                                   distance_config, filled_params, selected_formulas)
    _, _, inner_loss = data_manager_output2loss(data, model, None)
    # TODO: seems freeze_dumps in inner_loss is unneeded
    grads = torch.autograd.grad(inner_loss, to_adapt_params.values(),
                                create_graph=not detach and not first_order, allow_unused=True)
    updated_params = OrderedDict()
    if mom_buffer is None:
        mom_buffer = OrderedDict()
        if momentum > 0:
            for name, param in to_adapt_params.items():
                mom_buffer[name] = torch.zeros_like(param)
    for (name, param), grad in zip(to_adapt_params.items(), grads):
        if grad is None:
            updated_param = param
        else:
            if weight_decay > 0:
                grad = grad + weight_decay * param
            if momentum > 0:
                grad = grad + momentum * mom_buffer[name]
                mom_buffer[name] = grad
            updated_param = param - adapt_lr * grad
        if detach:
            updated_param = updated_param.detach().requires_grad_(True)
        updated_params[name] = updated_param
    return updated_params, inner_loss.item(), mom_buffer
