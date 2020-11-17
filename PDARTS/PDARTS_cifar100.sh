save=PDARTS_cifar100
gpu=2

# python train_cifar.py \
# --save $save --dataset CIFAR100 --layers 20 \
# --arch $save --gpu $gpu --auxiliary --cutout 
# layers 20 for cifar10 and cifar100, 8 for sport8, mit67 and flowers102

python test_adv.py --arch $save --batch_size 32 --gpu $gpu --auxiliary --cutout \
--attack PGD --init_channels 36 --dataset cifar100 --layers 20 \
--model_path "/home/yuezx/NAS-Benchmark/PDARTS/PDARTS_cifar100/batchsize32_channel36/weights.pt"