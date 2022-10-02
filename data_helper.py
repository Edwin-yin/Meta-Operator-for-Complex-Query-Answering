#!/usr/bin/python3
import copy
import os
import pickle
from collections import defaultdict
from typing import List
from math import ceil

import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset
from tqdm import tqdm

from fol import parse_formula, beta_query_v2
from utils.class_util import fixed_depth_nested_dict
from utils.independent_util import count_single_distance, all_normal_form


class Task:
    def __init__(self, filename, task_betaname):
        self.filename = filename
        self.device = None
        self.query_instance = None
        self.beta_name = task_betaname
        self.answer_set = None
        self.easy_answer_set = None
        self.hard_answer_set = None
        self.i = 0
        self.length = 0
        self._load()
        self.idxlist = np.random.permutation(len(self))
        # self.idxlist = np.arange(len(self))

    def to(self, device):
        self.query_instance.to(device)
        self.device = device

    def _load(self):
        dense = self.filename.replace('data', 'tmp').replace('csv', 'pickle')
        if os.path.exists(dense):
            print("load from existed files")
            with open(dense, 'rb') as f:
                data = pickle.load(f)
                self.query_instance = data['query_instance']
                self.answer_set = data['answer_set']
                self.easy_answer_set = data['easy_answer_set']
                self.hard_answer_set = data['hard_answer_set']
                self.length = len(self.query_instance)
        else:
            df = pd.read_csv(self.filename)
            self.query_instance = parse_formula(beta_query_v2[self.beta_name])
            self._parse(df)
            data = {'query_instance': self.query_instance, 'answer_set': self.answer_set,
                    'easy_answer_set': self.easy_answer_set, 'hard_answer_set': self.hard_answer_set}
            try:
                os.makedirs(os.path.dirname(dense), exist_ok=True)
                print(f"save to {dense}")
                with open(dense, 'wb') as f:
                    pickle.dump(data, f)
            except:
                print(f"can't save to {dense}")

    def __len__(self):
        return self.length

    def setup_iteration(self):
        self.idxlist = np.random.permutation(len(self))
        # self.idxlist = np.arange(len(self))

    def batch_estimation_iterator(self, estimator, batch_size):
        assert self.device == estimator.device
        i = 0
        while i < len(self):
            batch_indices = self.idxlist[i: i + batch_size].tolist()
            i += batch_size
            batch_embedding = self.query_instance.embedding_estimation(estimator=estimator, batch_indices=batch_indices)
            yield batch_embedding, batch_indices

    def _parse(self, df):
        for q in tqdm(df['query']):
            self.query_instance.additive_ground(json.loads(q))

        if 'answer_set' in df.columns:
            self.answer_set = df.answer_set.map(lambda x: list(eval(x))).tolist()
            assert len(self.query_instance) == len(self.answer_set)

        if 'easy_answer_set' in df.columns:
            self.easy_answer_set = df.easy_answer_set.map(
                lambda x: list(eval(x))).tolist()
            assert len(self.query_instance) == len(self.easy_answer_set)

        if 'hard_answer_set' in df.columns:
            self.hard_answer_set = df.hard_answer_set.map(
                lambda x: list(eval(x))).tolist()
            assert len(self.query_instance) == len(self.hard_answer_set)

        self.length = len(self.query_instance)


class TaskManager:
    def __init__(self, mode, tasks: List[Task], device):
        self.tasks = {t.query_instance.formula: t for t in tasks}
        self.task_iterators = {}
        self.mode = mode
        partition = []
        for t in self.tasks:
            self.tasks[t].to(device)
            partition.append(len(self.tasks[t]))
        p = np.asarray(partition)
        self.partition = p / p.sum()

    def build_iterators(self, estimator, batch_size):
        self.task_iterators = {}
        for i, tmf in enumerate(self.tasks):
            self.tasks[tmf].setup_iteration()
            self.task_iterators[tmf] = \
                self.tasks[tmf].batch_estimation_iterator(
                    estimator,
                    int(batch_size * self.partition[i]))

        while True:
            finish = 0
            data = defaultdict(dict)
            for tmf in self.task_iterators:
                try:
                    emb, batch_id = next(self.task_iterators[tmf])
                    data[tmf]['emb'] = emb
                    if self.mode in ['train', 'finetune']:
                        ans_sets = [self.tasks[tmf].answer_set[j] for j in batch_id]
                        data[tmf]['answer_set'] = ans_sets
                    else:
                        easy_ans_sets = [self.tasks[tmf].easy_answer_set[j] for j in batch_id]
                        data[tmf]['easy_answer_set'] = easy_ans_sets
                        hard_ans_sets = [self.tasks[tmf].hard_answer_set[j] for j in batch_id]
                        data[tmf]['hard_answer_set'] = hard_ans_sets

                except StopIteration:
                    finish += 1

            if finish == len(self.tasks):
                break

            yield data


class TestDataset(Dataset):
    def __init__(self, flattened_queries):
        # flattened_queries is a list of (query, easy_ans_set, hard_ans_set, query_structure) list
        self.len = len(flattened_queries)
        self.flattened_queries = flattened_queries

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.flattened_queries[idx]

    @staticmethod
    def collate_fn(flattened_queries):
        query = [_[0] for _ in flattened_queries]
        easy_ans_set = [_[1] for _ in flattened_queries]
        hard_ans_set = [_[2] for _ in flattened_queries]
        beta_name = [_[3] for _ in flattened_queries]
        return query, easy_ans_set, hard_ans_set, beta_name


class MyDataIterator:
    def __init__(self, tasks) -> None:
        self.tasks = tasks


class TrainDataset(Dataset):
    def __init__(self, flattened_queries):
        # flattened_queries is a list of (query, ans_set, query_structure) list
        self.len = len(flattened_queries)
        self.flattened_queries = flattened_queries

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.flattened_queries[idx]

    @staticmethod
    def collate_fn(flattened_queries):
        query = [_[0] for _ in flattened_queries]
        ans_set = [_[1] for _ in flattened_queries]
        beta_name = [_[2] for _ in flattened_queries]
        return query, ans_set, beta_name


class BenchmarkFormManager:  # A FormManager is actually managing all different normal forms of the same formula
    def __init__(self, mode, type_str, query_inform_dict: dict, filename: str, device, model, all_distance2index=None):
        # type_str: type0001
        self.type_str = type_str
        self.mode = mode
        self.query_inform_dict = query_inform_dict
        self.tasks, self.form2formula = {}, {}
        self.all_formula, self.allowed_formula = set(), set()
        self.formula_distance_config = copy.deepcopy(all_distance2index)
        self.full_name_formula_to_formula = {}
        for normal_form in all_normal_form:
            # Note that different query_class may have same formula: (p(u(p,p))) and (u,((p,p),(p,p)))
            formula = query_inform_dict[normal_form]
            full_name_formula = f'{type_str}_{formula}'
            self.form2formula[normal_form] = full_name_formula
            self.all_formula.add(full_name_formula)
            self.full_name_formula_to_formula[full_name_formula] = formula
        print(f'[data] load query from file {filename}')
        self._load(filename, model)
        self.task_iterators = {}
        for t in self.tasks:
            self.tasks[t].set_up(device, self.len)
        self.partition = [1 / len(self.tasks) for i in range(len(self.tasks))]

    def _load(self, filename, model):
        dense = filename.replace('data', 'tmp').replace('csv', 'pickle')
        if os.path.exists(dense):
            print("load from existed files")
            with open(dense, 'rb') as f:
                data = pickle.load(f)
                if self.mode in ['train', 'finetune']:
                    self.answer_set = data['answer_set']
                    self.len = len(self.answer_set)
                else:
                    self.easy_answer_set = data['easy_answer_set']
                    self.hard_answer_set = data['hard_answer_set']
                    self.len = len(self.easy_answer_set)
                for full_name_formula in self.all_formula:
                    single_formula_name = self.full_name_formula_to_formula[full_name_formula]
                    query_instance = data[full_name_formula]
                    if self.formula_distance_config:
                        count_single_distance(query_instance, self.formula_distance_config[full_name_formula])
                    try:
                        query_instance.to(model.device)
                        pred_emb = query_instance.embedding_estimation(
                            estimator=model, batch_indices=[0, 1, 2, 3])
                        assert pred_emb.ndim == 2 + ('u' in single_formula_name or 'U' in single_formula_name)
                        self.allowed_formula.add(full_name_formula)
                    except (AssertionError, RuntimeError):
                        pass
                    if full_name_formula in self.allowed_formula:
                        self.tasks[full_name_formula] = BenchmarkTask(query_instance)
                    assert len(data[full_name_formula]) == self.len
        else:
            df = pd.read_csv(filename)
            self.len = len(df)
            loaded = {full_name_formula: False for full_name_formula in self.all_formula}
            data = {}
            # todo: 'easy_answers' all change to easy_answer_set, and so does hard answers
            if self.mode in ['train', 'finetune']:
                if 'answer_set' in df.columns:
                    self.answer_set = df.answer_set.map(lambda x: list(eval(x))).tolist()
                    data = {'answer_set': self.answer_set}
            elif self.mode == 'valid' or self.mode == 'test':
                if 'easy_answers' in df.columns or 'easy_answer_set' in df.columns:
                    if 'easy_answer_set' in df.columns:
                        self.easy_answer_set = df.easy_answer_set.map(
                            lambda x: list(eval(x))).tolist()
                    else:
                        self.easy_answer_set = df.easy_answers.map(
                            lambda x: list(eval(x))).tolist()
                    assert self.len == len(self.easy_answer_set)
                if 'hard_answers' in df.columns or 'hard_answer_set' in df.columns:
                    if 'hard_answer_set' in df.columns:
                        self.hard_answer_set = df.hard_answer_set.map(
                            lambda x: list(eval(x))).tolist()
                    else:
                        self.hard_answer_set = df.hard_answers.map(
                            lambda x: list(eval(x))).tolist()
                    assert self.len == len(self.hard_answer_set)
                    data = {'easy_answer_set': self.easy_answer_set, 'hard_answer_set': self.hard_answer_set}
            else:
                assert False, 'not valid mode!'
            for normal_form in all_normal_form:
                full_name_formula = self.form2formula[normal_form]
                if not loaded[full_name_formula]:
                    single_formula_name = self.full_name_formula_to_formula[full_name_formula]
                    query_instance = parse_formula(single_formula_name)
                    for q in df[normal_form]:
                        query_instance.additive_ground(json.loads(q))
                    data[full_name_formula] = query_instance
                    query_instance.to(model.device)
                    if self.formula_distance_config:
                        count_single_distance(query_instance, self.formula_distance_config[full_name_formula])
                    try:
                        pred_emb = query_instance.embedding_estimation(
                            estimator=model, batch_indices=[0, 1, 2, 3])
                        assert pred_emb.ndim == 2 + ('u' in single_formula_name or 'U' in single_formula_name)
                        self.allowed_formula.add(full_name_formula)
                    except (AssertionError, RuntimeError):
                        pass
                    if full_name_formula in self.allowed_formula:
                        self.tasks[full_name_formula] = BenchmarkTask(query_instance)
                    loaded[full_name_formula] = True
            try:
                os.makedirs(os.path.dirname(dense), exist_ok=True)
                print(f"save to {dense}")
                with open(dense, 'wb') as f:
                    pickle.dump(data, f)
            except:
                print(f"can't save to {dense}")

    def build_iterators(self, estimator, batch_size):
        self.task_iterators = {}
        for i, tmf in enumerate(self.tasks):
            self.task_iterators[tmf] = \
                self.tasks[tmf].batch_estimation_iterator(
                    estimator,
                    int(batch_size * self.partition[i]))

        while True:
            finish = 0
            data = defaultdict(dict)
            for tmf in self.task_iterators:
                try:
                    emb, batch_id = next(self.task_iterators[tmf])
                    data[tmf]['emb'] = emb
                    easy_ans_sets = [self.easy_answer_set[j] for j in batch_id]
                    data[tmf]['easy_answer_set'] = easy_ans_sets
                    hard_ans_sets = [self.hard_answer_set[j] for j in batch_id]
                    data[tmf]['hard_answer_set'] = hard_ans_sets

                except StopIteration:
                    finish += 1

            if finish == len(self.tasks):
                break

            yield data


class BenchmarkTask:  # A Task is a formula(corresponding to a query_instance), thus it only needs idxlist
    def __init__(self, query_instance, distance_config=None):
        self.query_instance = query_instance
        self.device = None
        self.answer_set = None
        self.easy_answer_set = None
        self.hard_answer_set = None
        self.index = 0
        self.length = 0
        self.idxlist = np.arange(len(self))

    def set_up(self, device, length):
        self.length = length
        self.query_instance.to(device)
        self.device = device
        self.idxlist = np.arange(len(self))

    def setup_iteration(self):
        self.idxlist = np.random.permutation(len(self))
        self.index = 0

    def restart(self):
        self.index = 0

    def __len__(self):
        return self.length

    def batch_estimation_iterator(self, estimator, batch_size):
        assert self.device == estimator.device
        i = 0
        while i < len(self):
            batch_indices = self.idxlist[i: i + batch_size].tolist()
            i += batch_size
            batch_embedding = self.query_instance.embedding_estimation(estimator=estimator, batch_indices=batch_indices)
            yield batch_embedding, batch_indices

    def get_batch_data(self, estimator, batch_size, update_index: bool = True, freeze_dict=None, distance_config=None,
                       meta_parameters=None):
        assert self.device == estimator.device
        if self.index < self.length:
            batch_indices = self.idxlist[self.index: self.index + batch_size].tolist()
            if update_index:
                self.index += batch_size
            else:
                pass
            batch_embedding = self.query_instance.embedding_estimation(estimator=estimator, batch_indices=batch_indices,
                                                                       full_distance_config=distance_config,
                                                                       freeze_dumps=freeze_dict,
                                                                       meta_parameters=meta_parameters)
            finish = (self.index >= self.length)
            return batch_embedding, batch_indices, finish
        else:
            return None, None, None

    def get_raw_data(self, batch_size, auto_restart: bool = True):
        """
        get raw data indices, if there is no enough query left for a whole batch, the remaining will also be output
        """
        if self.index < self.length:
            batch_indices = self.idxlist[self.index: self.index + batch_size].tolist()
            self.index += batch_size
            return batch_indices
        else:
            if auto_restart:
                self.setup_iteration()
                batch_indices = self.idxlist[self.index: self.index + batch_size].tolist()
                return batch_indices
            else:
                return None

    def simply_update_index(self, batch_size):
        if self.index < self.length:
            self.index += batch_size
        finish = (self.index >= self.length)
        return finish


class BenchmarkWholeManager:  # It manages all tasks in machine learning algorithm
    def __init__(self, mode, formula_id_data, data_folder: str, interested_normal_form: list, device, model,
                 old_name: bool = False, all_distance2index=None):
        """
        The computation of query distance is done on _load of BenchmarkFormManager
        """
        self.mode = mode
        self.formula_id_data = formula_id_data
        self.interested_normal_form = interested_normal_form
        self.query_classes = {}
        self.partition = {}
        self.task_iterators = {}
        self.full_formula_to_type_str = {}
        self.full_formula_to_formula = {}
        self.all_task_length = 0
        self.finish_task = []
        self.formula2id = {}
        self.all_distance2index = copy.deepcopy(all_distance2index)

        for i in formula_id_data.index:
            type_str = formula_id_data['formula_id'][i]
            if old_name:
                filename = os.path.join(data_folder, f'data-{type_str}.csv')
            else:
                filename = os.path.join(data_folder, f'{mode}-{type_str}.csv')
            # real_index = formula_id_data.loc[formula_id_data['formula_id'] == f'{type_str}'].index[0]
            # index != formula id
            query_class_dict = formula_id_data.loc[i]
            self.query_classes[type_str] = BenchmarkFormManager(mode, type_str, query_class_dict, filename, device,
                                                                model, self.all_distance2index)

        # all types of queries are sampled together
        now_id = 0
        for i, type_str in enumerate(self.query_classes):
            interested_full_formulas = set([self.query_classes[type_str].form2formula[form] for form in
                                            self.interested_normal_form])
            final_allowed_formulas = interested_full_formulas.intersection(self.query_classes[type_str].allowed_formula)
            final_allowed_formulas_list = list(final_allowed_formulas)
            final_allowed_formulas_list.sort()  # Do such thing to ensure the exact same order within a query_classed.

            for full_formula in final_allowed_formulas_list:
                self.full_formula_to_type_str[full_formula] = type_str
                self.partition[full_formula] = len(self.query_classes[type_str].tasks[full_formula])
                self.all_task_length += self.partition[full_formula]
                self.formula2id[full_formula] = now_id
                now_id += 1
        for full_name_formula in self.formula2id:
            self.partition[full_name_formula] /= self.all_task_length
        self.set_up_whole_iteration()

    def build_iterators(self, estimator, batch_size):
        self.task_iterators = {}
        for full_name_formula in self.formula2id:
            self.query_classes[self.full_formula_to_type_str[full_name_formula]].tasks[
                full_name_formula].setup_iteration()
            self.task_iterators[full_name_formula] = self.query_classes[full_name_formula].tasks[
                full_name_formula].batch_estimation_iterator(
                estimator, ceil(batch_size * self.partition[full_name_formula]))
        while True:
            finish = 0
            data = defaultdict(dict)
            for task_formula in self.task_iterators:
                try:
                    emb, batch_id = next(self.task_iterators[task_formula])
                    data[task_formula]['emb'] = emb
                    if self.mode in ['train', 'finetune']:
                        ans_sets = [self.query_classes[self.full_formula_to_type_str[task_formula]].answer_set[j]
                                    for j in batch_id]
                        data[task_formula]['answer_set'] = ans_sets
                    else:
                        easy_ans_sets = [self.query_classes[self.full_formula_to_type_str[task_formula]]
                                             .easy_answer_set[j] for j in batch_id]
                        data[task_formula]['easy_answer_set'] = easy_ans_sets
                        hard_ans_sets = [self.query_classes[self.full_formula_to_type_str[task_formula]]
                                             .hard_answer_set[j] for j in batch_id]
                        data[task_formula]['hard_answer_set'] = hard_ans_sets
                except StopIteration:
                    finish += 1

            if finish == len(self.full_formula_to_type_str):
                break

            yield data

    def set_up_whole_iteration(self):
        for full_name_formula in self.full_formula_to_type_str:
            self.query_classes[self.full_formula_to_type_str[full_name_formula]].tasks[
                full_name_formula].setup_iteration()

    def get_whole_batch_data(self, estimator, batch_size, update_index: bool = True, freeze_dict=None,
                             distance_config=None, meta_parameters=None, selected_formulas: set = None):
        """
        This function utilizes get_batch_data in BenchmarkTask to get pred_emb and use stored answer set to give
        full data for compute logit.
        selected_formulas: A set that contains all formulas to be computed, other formula will be skipped, however,
        the partition will not be enlarged accordingly.
        """
        data = defaultdict(dict)
        for full_formula in self.formula2id:
            if selected_formulas is None or full_formula in selected_formulas:
                emb, batch_id, finish = self.query_classes[self.full_formula_to_type_str[full_formula]].tasks[
                    full_formula].get_batch_data(estimator, ceil(batch_size * self.partition[full_formula]),
                                                 update_index,
                                                 json.loads(freeze_dict[full_formula]) if freeze_dict else None,
                                                 distance_config[full_formula] if distance_config else None,
                                                 meta_parameters)
                # TODO: whether the partition should change too?
                # TODO: the set up iteration is totally changed.
                if finish:
                    self.query_classes[
                        self.full_formula_to_type_str[full_formula]].tasks[full_formula].setup_iteration()
                if batch_id:
                    data[full_formula]['emb'] = emb
                    if self.mode in ['train', 'finetune']:
                        ans_sets = [self.query_classes[self.full_formula_to_type_str[full_formula]].answer_set[j]
                                    for j in batch_id]
                        data[full_formula]['answer_set'] = ans_sets
                    else:
                        easy_ans_sets = [self.query_classes[self.full_formula_to_type_str[
                            full_formula]].easy_answer_set[j] for j in batch_id]
                        data[full_formula]['easy_answer_set'] = easy_ans_sets
                        hard_ans_sets = [self.query_classes[self.full_formula_to_type_str[
                            full_formula]].hard_answer_set[j] for j in batch_id]
                        data[full_formula]['hard_answer_set'] = hard_ans_sets

        return data

    def get_rawdata_single_task(self, batch_size, full_name_formula: str, auto_restart: bool = True):
        data = defaultdict(list)
        single_task_manager = self.query_classes[
            self.full_formula_to_type_str[full_name_formula]].tasks[full_name_formula]
        batch_indices = single_task_manager.get_raw_data(batch_size, auto_restart)
        data['batch_indices'] = batch_indices
        if self.mode in ['train', 'finetune']:
            ans_sets = [self.query_classes[self.full_formula_to_type_str[full_name_formula]].answer_set[j]
                        for j in batch_indices]
            data['answer_set'] = ans_sets
        else:
            easy_ans_sets = [self.query_classes[self.full_formula_to_type_str[full_name_formula]].easy_answer_set[j]
                             for j in batch_indices]
            data['easy_answer_set'] = easy_ans_sets
            hard_ans_sets = [self.query_classes[self.full_formula_to_type_str[full_name_formula]].hard_answer_set[j]
                             for j in batch_indices]
            data['hard_answer_set'] = hard_ans_sets
        return data

    def index2emb(self, estimator, full_name_formula: str, batch_indices, distance_config=None, freeze_dumps=None,
                  meta_parameters=None):
        single_task_manager = self.query_classes[
            self.full_formula_to_type_str[full_name_formula]].tasks[full_name_formula]
        emb = single_task_manager.query_instance.embedding_estimation(estimator, batch_indices, distance_config,
                                                                      freeze_dumps, meta_parameters)
        return emb

    def simply_update_index(self, batch_size, selected_formulas=None):
        for full_formula in self.formula2id:
            if selected_formulas is None or full_formula in selected_formulas:
                finish = self.query_classes[self.full_formula_to_type_str[full_formula]].tasks[
                             full_formula].simply_update_index(ceil(batch_size * self.partition[full_formula]))
                if finish:
                    self.query_classes[self.full_formula_to_type_str[full_formula]].tasks[
                        full_formula].setup_iteration()


def count_formula_distance(query_instance, distance_config):
    distance2index = fixed_depth_nested_dict(int, 2)
    if distance_config['use_distance'] == 'leaf':
        _, final_accumulative = query_instance.get_leaf_distance(distance_config['distance_operators'])
    elif distance_config['use_distance'] == 'root':
        final_accumulative = query_instance.get_root_distance(distance_config['distance_operators'], 0)
    else:
        final_accumulative = fixed_depth_nested_dict(int, 2)
    for operator in final_accumulative:
        now_index = 0
        for depth in final_accumulative[operator]:
            distance2index[operator][depth] = now_index
            now_index += 1
    return distance2index






