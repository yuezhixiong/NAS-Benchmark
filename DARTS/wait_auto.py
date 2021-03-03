import os, subprocess
import sys
import time, re
import numpy as np
from datetime import datetime
 
cmd = ['python', 'auto.py']
# cmd = ['python', 'train_imagenet_dali.py', '--arch', 'LL_acc1_ood1_UL_adv_nop_flp_mgda_abs30_gnl2_cifar100', '--save', 'LL_acc1_ood1_UL_adv_nop_flp_mgda_abs30_gnl2_cifar100']
memo_required = 27000 # 17000
 
# def gpu_info():
#     gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
#     gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
#     gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
#     # print(gpu_status)
#     return gpu_power, gpu_memory
 
def free_gpu():
    gpu_memo = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free').read()
    gpu_memo = re.findall('(\d+) MiB', gpu_memo)
    gpu_memo = [int(x) for x in gpu_memo]
    print(gpu_memo)
    gpu_id = -1
    if max(gpu_memo) > memo_required:
        gpu_id = np.argmax(gpu_memo)
    return gpu_id

def narrow_setup(interval=10):
    # gpu_power, gpu_memory = gpu_info()
    error = 1
    while error != 0:
        gpu_id = free_gpu()
        while gpu_id < 0:
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            gpu_id = free_gpu()
            time.sleep(interval)

        print('find free gpu:', gpu_id)
        cmd_gpu = cmd + ['--gpu'] + [str(gpu_id)]
        print('now running:', cmd_gpu)
        # os.system(cmd_gpu)

        result = subprocess.run(cmd_gpu)
        error = result.returncode
        if error != 0:
            print('An error occurred, keep looking')

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