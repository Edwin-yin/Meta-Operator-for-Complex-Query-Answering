import os
import pandas as pd


def rename_data(folder, formula_file, old_prefix, new_prefix):
    formula_data = pd.read_csv(formula_file)
    for i, row in formula_data.iterrows():
        type_str = formula_data['formula_id'][i]
        type_filename = os.path.join(folder, f"{old_prefix}-{type_str}.csv")
        if os.path.exists(type_filename):
            os.rename(type_filename, os.path.join(folder, f"{new_prefix}-{type_str}.csv"))
        else:
            print(f'Warning! {type_str} not exist!')


if __name__ == '__main__':
    benchmark_237_folder = 'data/benchmark/FB15k-237'
    benchmark_formula_file = 'data/test_generated_formula_anchor_node=3.csv'
    finetune_folder = "/home/hyin/FirstOrderQueryEstimation/data/FB15k-237-betae-train-500/finetune_50_test_formulas.csv"
    test_formula_file = 'data/FB15k-237-betae/test_formulas.csv'
    rename_data(finetune_folder, test_formula_file, 'train', 'finetune')


