save=pcdarts_fgsm_nop_min
gpu=4

# python train_search.py \
# --save $save --gpu $gpu \
# --dataset CIFAR10

python train.py \
--dataset CIFAR10 --gpu $gpu --save $save --arch $save \
--auxiliary --cutout --layers 20 #20 for CIFAR datasets, 8 for Sport8, MIT67 and flowers102