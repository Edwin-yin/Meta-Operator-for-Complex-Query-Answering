import copy
import os
import math
from itertools import product
import yaml

base_file = "NGC_task_config/urgent/LogicE_0001_root_lr_0.008.yaml"
ckpt_path = 'EFO-1_log/urgent/LogicE_0001_root_lr_0.008.yaml220930.09:18:556a6ce9bb'
output_folder = "NGC_task_config/NGC_test_urgent"
step_list = list(range(240000, 465000, 15000))
model_name = 'LogicE'
output_config_count = 0

with open(base_file) as f:
    original_config = yaml.full_load(f)
    print(original_config)

data_folder_list = ['data/FB15k-237-betae-1p-001', 'data/FB15k-237-betae-1p-0001',
                    'data/FB15k-237-betae-20', 'data/FB15k-237-betae-10',
                    'data/FB15k-237-betae-5', 'data/FB15k-237-betae-1']
parameter_to_choose = ['adapt_lr_list', 'adapt_step_list']
model_list = ['LogicE', 'beta', 'ConE']
distance_type = ['input_binary', 'output_binary', 'input', 'output', 'root', 'leaf']
distance_binary_type = ['input_binary', 'output_binary']
distance_p_type = ['input_binary', 'output_binary', 'root', 'leaf']
#k_query_list = [7, 31]\
model_lr_dict = {
    'LogicE': [0.008, 0.004, 0.016],
    'BetaE': [0.002, 0.001, 0.0005],
    'ConE': [0.002, 0.004, 0.008]
}


def check_config(to_rectify_config):
    """
    Make sure is FB15k-237
    """
    if to_rectify_config['estimator']['embedding'] == 'ConE':
        to_rectify_config['train']['steps'], to_rectify_config['train']['warm_up_steps'] = 300000, 150000
        to_rectify_config['train']['learning_rate'] = 0.00005
    elif to_rectify_config['estimator']['embedding'] in ['logic', 'beta']:
        to_rectify_config['train']['steps'], to_rectify_config['train']['warm_up_steps'] = 450000, 225000
        to_rectify_config['train']['learning_rate'] = 0.0001
    if to_rectify_config['train']['formula_id_file'] in ['data/FB15k-237-betae/p_formulas.csv',
                                                         'data/FB15k-237-betae/1p.csv',
                                                         'data/FB15k-237-betae/1p+2p.csv']:
        to_rectify_config['train']['steps'] *= 2
        to_rectify_config['train']['warm_up_steps'] *= 2
    to_rectify_config['MetaLearning']['operator_MAML']['finetune_lr'] = \
        to_rectify_config['MetaLearning']['operator_MAML']['adaptation_lr'] * \
        to_rectify_config['MetaLearning']['operator_MAML']['adaptation_step'] / 4
    if to_rectify_config['MetaLearning']['operator_MAML']['adaptation_step'] == 1:
        to_rectify_config['MetaLearning']['operator_MAML']['momentum'], \
        to_rectify_config['MetaLearning']['operator_MAML']['weight_decay'] = 0, 0
    if to_rectify_config['MetaLearning']['operator_MAML']['adaptation_step'] > 1:
        to_rectify_config['MetaLearning']['operator_MAML']['first_order'] = True
    return to_rectify_config


adapt_step_list = [4]
momentum_weight_decay_combination = [(0, 0), (0.9, 0.001), (0.9, 0)]
for step in step_list:
    # change one of parameter_to_choose:
    config = copy.deepcopy(original_config)
    config["cuda"] = 0
    config['load']['load_model'] = 'true'
    config['load']['checkpoint_path'] = ckpt_path
    config['load']['step'] = step
    config['train']['steps'] = step + 1
    #data_folder = 'data/FB15k-237-betae-1p-00001'
    #config['data']['data_folder'] = data_folder
    data_folder_short = config['data']['data_folder'].split('-')[-1]
    #config = check_config(config)
    case_name = f'{model_name}_{data_folder_short}_eval_{config["train"]["use_distance"]}_lr_' \
                        f'{config["MetaLearning"]["operator_MAML"]["adaptation_lr"]}_step_{step}'
    config_name = case_name + '.yaml'
    output_config_count += 1
    with open(os.path.join(output_folder, config_name), 'wt') as f:
        yaml.dump(config, f)


print(output_config_count)

'''
for distance in product(distance_type):
    # change one of parameter_to_choose:
    for i, parameter_name in enumerate(parameter_to_choose):
        config = copy.deepcopy(original_config)
        config["cuda"] = 0
        config["train"]["use_distance"] = distance
        momentum, weight_decay = momentum_weight_decay
        for parameter_change in eval(parameter_name):
            if i == 0:
                config['MetaLearning']['operator_MAML']['adaptation_lr'] = parameter_change
            elif i == 1:
                config['MetaLearning']['operator_MAML']['adaptation_step'] = parameter_change
                config['MetaLearning']['operator_MAML']['adaptation_lr'] /= parameter_change
            config['MetaLearning']['operator_MAML']['finetune_lr'] = \
                config['MetaLearning']['operator_MAML']['adaptation_lr'] * \
                config['MetaLearning']['operator_MAML']['adaptation_step'] / 4
            if config['MetaLearning']['operator_MAML']['adaptation_step'] == 1:
                momentum, weight_decay = 0, 0
            config['MetaLearning']['operator_MAML']['momentum'] = momentum
            config['MetaLearning']['operator_MAML']['weight_decay'] = weight_decay
            if config['MetaLearning']['operator_MAML']['adaptation_step'] > 1:
                config['MetaLearning']['operator_MAML']['first_order'] = True
            case_name = f"{model_name}_{distance}_lr_" \
                        f"{config['MetaLearning']['operator_MAML']['adaptation_lr']}_step_" \
                        f"{config['MetaLearning']['operator_MAML']['adaptation_step']}_m_{momentum}_{weight_decay}"
            config_name = case_name + '.yaml'
            config['output']['output_path'] = f'operator_Meta-1p-01/{case_name}'
            output_config_count += 1
            with open(os.path.join(output_folder, config_name), 'wt') as f:
                yaml.dump(config, f)
                '''