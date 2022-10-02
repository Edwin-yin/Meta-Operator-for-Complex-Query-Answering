import os

config_folder = "NGC_task_config/NGC_test_urgent"

for config_file in os.listdir(config_folder):
    if not config_file.endswith('.yaml'):
        continue

    case_name = config_file.split('.y')[0]

    command = ("ngc batch run "
               f"--name ml-model.{case_name} "
               "--priority NORMAL "
               "--preempt RUNONCE "
               "--ace nv-us-west-2 "
               "--instance dgx1v.32g.1.norm "
               f'--commandline "cd /mount/efo1-meta && python3 main.py --config {config_folder}/{config_file}" '
               "--result /results "
               "--image nvidia/pytorch:22.04-py3 "
               "--org nvidian "
               "--team sae "
               "--workspace kzvK1JEYTYO00ri0sLDQVA:/mount/efo1-meta:RW "
               "--order 50")
    print(command, '\n')
    os.system(command)
    print("launched")
