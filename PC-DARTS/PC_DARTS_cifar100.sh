save=PC_DARTS_cifar100
gpu=3

# python train.py --save $save --gpu $gpu --arch $save \
# --auxiliary --cutout \
# --dataset CIFAR100 --layers 20 #20 for CIFAR datasets, 8 for Sport8, MIT67 and flowers102

# python test_adv.py --arch $save --gpu $gpu --auxiliary --cutout \
# --attack PGD --batch_size 16 \
# --model_path "/home/yuezx/NAS-Benchmark/PC-DARTS/PC_DARTS_cifar/weights.pt"

python test_adv.py --arch $save --gpu $gpu --auxiliary --cutout \
--attack PGD --batch_size 32 --dataset cifar100 \
--model_path "/home/yuezx/NAS-Benchmark/PC-DARTS/PC_DARTS_cifar100/batchsize32_channel36_CIFAR100/weights.pt"