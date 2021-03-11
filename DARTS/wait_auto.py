import os, subprocess
import sys
import time, re
import numpy as np
from datetime import datetime

# cmd = ['python', 'train_imagenet.py']
# cmd = ['python', 'train_imagenet_dali.py', '--arch', 'LL_acc1_ood1_UL_adv_nop_flp_mgda_abs30_gnl2_cifar100', '--save', 'LL_acc1_ood1_UL_adv_nop_flp_mgda_abs30_gnl2_cifar100']
memo_required = 30000
 
# def gpu_info():
#     gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
#     gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
#     gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
#     # print(gpu_status)
#     return gpu_power, gpu_memory

wait_path = 'config/waiting/'
run_path = 'config/running/'
done_path = 'config/done/'

def get_task():
    task_list = os.listdir(wait_path)
    print(task_list)
    ini_name = task_list[0]
    ini_path = os.path.join(wait_path, ini_name)
    os.system('mv '+ini_path+' '+run_path)
    ini_path = os.path.join(run_path, ini_name)
    cmd = ['python', 'auto.py', '--config', ini_path]
    return cmd, ini_name

def free_gpu():
    gpu_memo = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free').read()
    gpu_memo = re.findall('(\d+) MiB', gpu_memo)
    gpu_memo = [int(x) for x in gpu_memo]
    print(gpu_memo)
    gpu_id = -1
    if max(gpu_memo) > memo_required:
        gpu_id = np.argmax(gpu_memo)
    return gpu_id

def narrow_setup(interval=300):
    # gpu_power, gpu_memory = gpu_info()
    error = 1

    gpu_id = free_gpu()
    while gpu_id < 0:
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        gpu_id = free_gpu()
        time.sleep(interval)

    print('find free gpu:', gpu_id)
    cmd, ini_name = get_task()
    print('get task:', cmd)
    cmd_gpu = cmd + ['--gpu'] + [str(gpu_id)]
    print('now running:', cmd_gpu)
    ini_path = os.path.join(run_path, ini_name)

    result = subprocess.run(cmd_gpu)
    error = result.returncode
    print('run code:', result)
    if error != 0:
        print('An error occurred, exit')
        os.system('mv '+ini_path+' '+wait_path)
        exit()
    arch_name = ini_name.split('.')[0]
    plot_file = os.path.join(arch_name, arch_name+'.pdf')
    if os.path.isfile(plot_file):
        os.system('mv '+ini_path+' '+done_path)
    else:
        os.system('mv '+ini_path+' '+wait_path)
    # i = 0
    # while gpu_memory > 1000 or gpu_power > 40:  # set waiting condition
    #     gpu_power, gpu_memory = gpu_info()
    #     i = i % 5
    #     symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
    #     gpu_power_str = 'gpu power:%d W |' % gpu_power
    #     gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
    #     sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
    #     sys.stdout.flush()
    #     time.sleep(interval)
    #     i += 1
    # print('\n' + cmd)
    # os.system(cmd)
 
 
if __name__ == '__main__':
    narrow_setup()