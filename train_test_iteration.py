import torch
import collections
import math
from tqdm.std import tqdm

from fol.appfoq import compute_final_loss
from fol import order_bounds
from data_helper import BenchmarkWholeManager
from utils.class_util import nested_dict


def train_step(model, opt, data_manager: BenchmarkWholeManager, batch_size: int, freeze_data=None,
               distance_config=None):
    model.train()
    torch.autograd.set_detect_anomaly(True)
    opt.zero_grad()
    data = data_manager.get_whole_batch_data(model, batch_size, True, freeze_data, distance_config)
    positive_loss, negative_loss, loss = data_manager_output2loss(data, model, freeze_data)
    loss.backward()
    opt.step()
    log = {
        'po': positive_loss.item(),
        'ne': negative_loss.item(),
        'loss': loss.item()
    }
    if model.name == 'logic':
        entity_embedding = model.entity_embeddings.weight.data
        if model.bounded:
            model.entity_embeddings.weight.data = order_bounds(entity_embedding)
        else:
            model.entity_embeddings.weight.data = torch.clamp(entity_embedding, 0, 1)
    return log


def data_manager_output2loss(data, model, freeze_data=None):
    """
    Basically in train_step but extract these to use it in meta learning.
    freeze_data is used in confirming whether freeze 'e' embedding in criterion
    """
    emb_list, answer_list = [], []
    union_emb_list, union_answer_list = [], []
    if freeze_data:
        # In this case, I will compute logit for every formula correspondingly.
        positive_logit_list, negative_logit_list, subsampling_weight_list = [], [], []
        for full_formula in data:
            single_positive_logit, single_negative_logit, single_subsampling_weight = model.criterion(
                data[full_formula]['emb'], data[full_formula]['answer_set'],
                freeze_entity='"e", "a": ["True"]' in freeze_data[full_formula],
                union='u' in full_formula or 'U' in full_formula)
            positive_logit_list.append(single_positive_logit)
            negative_logit_list.append(single_negative_logit)
            subsampling_weight_list.append(single_subsampling_weight)
        all_positive_logit = torch.cat(positive_logit_list, dim=0)
        all_negative_logit = torch.cat(negative_logit_list, dim=0)
        all_subsampling_weight = torch.cat(subsampling_weight_list, dim=0)
    else:
        for full_formula in data:
            if 'u' in full_formula or 'U' in full_formula:  # TODO: consider 'evaluate_union' in the future
                union_emb_list.append(data[full_formula]['emb'])
                union_answer_list.append(data[full_formula]['answer_set'])
            else:
                emb_list.append(data[full_formula]['emb'])
                answer_list.extend(data[full_formula]['answer_set'])
        pred_embedding = torch.cat(emb_list, dim=0)
        all_positive_logit, all_negative_logit, all_subsampling_weight = model.criterion(pred_embedding, answer_list)
        for i in range(len(union_emb_list)):
            union_positive_logit, union_negative_logit, union_subsampling_weight = \
                model.criterion(union_emb_list[i], union_answer_list[i], union=True)
            all_positive_logit = torch.cat([all_positive_logit, union_positive_logit], dim=0)
            all_negative_logit = torch.cat([all_negative_logit, union_negative_logit], dim=0)
            all_subsampling_weight = torch.cat([all_subsampling_weight, union_subsampling_weight], dim=0)
    positive_loss, negative_loss = compute_final_loss(all_positive_logit, all_negative_logit, all_subsampling_weight)
    loss = (positive_loss + negative_loss) / 2
    return positive_loss, negative_loss, loss


def eval_step(model, eval_iterator, mode, allowed_easy_ans=False, **kwargs):
    """
    old evaluation function that uses a iterator from Benchmark form manager
    """
    logs = nested_dict()
    with torch.no_grad():
        for data in tqdm(eval_iterator):
            for full_formula in data:
                pred = data[full_formula]['emb']
                if mode in ['train', 'finetune']:
                    easy_ans_list = []
                    hard_ans_list = data[full_formula]['answer_set']
                else:
                    easy_ans_list = data[full_formula]['easy_answer_set']
                    hard_ans_list = data[full_formula]['hard_answer_set']
                batch_log = eval_batch_query(model, pred, full_formula, easy_ans_list, hard_ans_list, allowed_easy_ans,
                                             **kwargs)
                for eval_metric in batch_log:
                    logs[full_formula][eval_metric] += batch_log[eval_metric]
        for full_formula in logs.keys():
            for log_metric in logs[full_formula].keys():
                if log_metric != 'num_queries':
                    logs[full_formula][log_metric] /= logs[full_formula]['num_queries']
    return logs


def eval_batch_query(model, pred_emb, full_formula, easy_ans_list, hard_ans_list, allowed_easy_ans: bool = False,
                     meta_parameters=None):
    """
    eval a batch of query of the same formula, the pred_emb of the query has been given.
    pred_emb:  (disjunction_num)*batch*emb_dim
    easy_ans_list: list of easy_ans
    """
    device = model.device
    logs = collections.defaultdict(float)
    with torch.no_grad():
        # TODO: never change the type_str since 'u' check is in here
        all_logit = model.compute_all_entity_logit(pred_emb, union=('u' in full_formula or 'U' in full_formula),
                                                   meta_parameters=meta_parameters)
        # batch*nentity
        argsort = torch.argsort(all_logit, dim=1, descending=True)
        ranking = argsort.clone().to(torch.float)
        #  create a new torch Tensor for batch_entity_range
        ranking = ranking.scatter_(1, argsort, torch.arange(model.n_entity).to(torch.float).
                                   repeat(argsort.shape[0], 1).to(device))
        # achieve the ranking of all entities
        for i in range(all_logit.shape[0]):
            if allowed_easy_ans:
                easy_ans = []
                hard_ans = list(set(easy_ans_list[full_formula]['hard_answer_set'][i]).union
                                (set(hard_ans_list[full_formula]['easy_answer_set'][i])))
            else:
                easy_ans = easy_ans_list[i]
                hard_ans = hard_ans_list[i]
            num_hard = len(hard_ans)
            num_easy = len(easy_ans)
            assert len(set(hard_ans).intersection(set(easy_ans))) == 0
            # only take those answers' rank
            cur_ranking = ranking[i, list(easy_ans) + list(hard_ans)]
            cur_ranking, indices = torch.sort(cur_ranking)
            masks = indices >= num_easy
            answer_list = torch.arange(num_hard + num_easy).to(torch.float).to(device)
            cur_ranking = cur_ranking - answer_list + 1
            # filtered setting: +1 for start at 0, -answer_list for ignore other answers
            cur_ranking = cur_ranking[masks]
            # only take indices that belong to the hard answers
            mrr = torch.mean(1. / cur_ranking).item()
            h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
            h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
            h10 = torch.mean(
                (cur_ranking <= 10).to(torch.float)).item()
            add_hard_list = torch.arange(num_hard).to(torch.float).to(device)
            hard_ranking = cur_ranking + add_hard_list  # for all hard answer, consider other hard answer
            logs['retrieval_accuracy'] += torch.mean(
                (hard_ranking <= num_hard).to(torch.float)).item()
            logs['MRR'] += mrr
            logs['HITS1'] += h1
            logs['HITS3'] += h3
            logs['HITS10'] += h10
        num_query = all_logit.shape[0]
        logs['num_queries'] += num_query
    # torch.cuda.empty_cache()
    return logs


def eval_step_whole(model, test_data_manager: BenchmarkWholeManager, mode: str, batch_size: int, distance_config=None,
                    all_formula_freeze_dumps=None, meta_parameters=None):
    """
    New evaluation function that uses BenchmarkWholeManager.
    """
    model.eval()
    test_data_manager.set_up_whole_iteration()
    logs = nested_dict()
    for full_formula in test_data_manager.formula2id:
        single_task_manager = test_data_manager.query_classes[
            test_data_manager.full_formula_to_type_str[full_formula]].tasks[full_formula]
        num_of_batch = math.ceil(single_task_manager.length / batch_size)
        with torch.no_grad():
            for batch_id in range(num_of_batch):
                batch_test_data = test_data_manager.get_rawdata_single_task(batch_size, full_formula, False)
                batch_pred_emb = test_data_manager.index2emb(model, full_formula, batch_test_data['batch_indices'],
                                                             distance_config[full_formula] if distance_config else None,
                                                             all_formula_freeze_dumps[full_formula] if
                                                             all_formula_freeze_dumps else None, meta_parameters)
                if mode in ['train', 'finetune']:
                    easy_ans_list = [[] for _ in range(len(batch_test_data['batch_indices']))]
                    hard_ans_list = batch_test_data['answer_set']
                else:
                    easy_ans_list = batch_test_data['easy_answer_set']
                    hard_ans_list = batch_test_data['hard_answer_set']
                batch_log = eval_batch_query(model, batch_pred_emb, full_formula, easy_ans_list, hard_ans_list, False,
                                             meta_parameters)
                for batch_metric in batch_log:
                    if type(logs[full_formula][batch_metric]) is not float:
                        logs[full_formula][batch_metric] = 0
                    logs[full_formula][batch_metric] += batch_log[batch_metric]
    for full_formula in logs.keys():
        for log_metric in logs[full_formula].keys():
            if log_metric != 'num_queries':
                logs[full_formula][log_metric] /= logs[full_formula]['num_queries']
    return logs
