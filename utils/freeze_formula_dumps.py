from os.path import join
import pandas as pd
import json

from fol import parse_formula
from utils.independent_util import all_normal_form


def output_freeze_dumps(formula_file, store_fold):
    formula_data = pd.read_csv(formula_file)
    output_dict = {}
    for i in formula_data.index:
        single_form_data = formula_data.loc[i]
        for normal_form in all_normal_form:
            specific_formula = single_form_data[normal_form]
            formula_instance = parse_formula(specific_formula)
            empty_dumps = formula_instance.dumps
            output_dict[specific_formula] = empty_dumps
    output_data = pd.DataFrame.from_dict(output_dict, orient='index')
    output_data.to_csv(join(store_fold, 'freeze_formula_dumps.csv'))


now_train_formula_file = 'data/FB15k-237-betae/truncated_formulas.csv'
output_folder = 'data/FB15k-237-betae'
output_freeze_dumps(now_train_formula_file, output_folder)
