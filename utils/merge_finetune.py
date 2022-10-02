import os
import pandas as pd
import shutil


def merge_finetune_data(finetune_folder, formula_file, output_folder, cover: bool = False, skip_formula_file=None):
    """
    This function is actually not merging: since the finetune data is sampled within the train data.
    The skip_formula_list is given by running select_formula.py.
    """
    formula_data = pd.read_csv(formula_file)
    if skip_formula_file:
        skip_formula_data = pd.read_csv(skip_formula_file)
        skip_formula_id_list = list(skip_formula_data['formula_id'])
    else:
        skip_formula_id_list = []
    for i, row in formula_data.iterrows():
        type_str = formula_data['formula_id'][i]
        if type_str not in skip_formula_id_list:
            type_filename = os.path.join(finetune_folder, f"finetune-{type_str}.csv")
            if os.path.exists(type_filename):
                train_file_path = os.path.join(output_folder, f"train-{type_str}.csv")
                if os.path.exists(train_file_path):
                    if cover:
                        shutil.copy(type_filename, train_file_path)
                        print(f'{type_str} already have, cover it!')
                    else:
                        print(f'{type_str} already have, skip!')
                else:
                    shutil.copy(type_filename, train_file_path)
                    print(f'{type_str} transform finetune data to train data.')
            else:
                print(f'Warning! {type_str} not exist!')
        else:
            print(f'{type_str} skip since it is selected.')




if __name__ == '__main__':
    finetune_237_folder = "data/FB15k-237-selected/finetune_50_test_formulas.csv"
    test_formula_file = 'data/FB15k-237-betae/test_formulas.csv'
    output_folder = "data/FB15k-237-selected"
    selected_formula_file = 'data/FB15k-237-betae/selected_formulas.csv'
    merge_finetune_data(finetune_237_folder, test_formula_file, output_folder, True, selected_formula_file)


