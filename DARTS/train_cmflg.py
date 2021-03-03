import os
import torch
import time, argparse

# declare which gpu device to use
parser = argparse.ArgumentParser()
parser.add_argument('--time', type=int, default='604800')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
cuda_device = args.gpu

def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.8)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    del x
    print('GPU{} blocking {} of {} in total'.format(cuda_device, block_mem, max_mem))
    return block_mem
    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    block_size = occumpy_mem(cuda_device)
    for _ in range(args.time):
        time.sleep(1)
        total, used = check_mem(cuda_device)
        print('GPU{}: {}/{}'.format(cuda_device, used, total), end='\r')
        if int(used) - block_size <= 1000:
            print('Protection terminated, GPU{} released'.format(cuda_device))
            break
    print('Done')