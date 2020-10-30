save=pcdarts_fgsm_nop_min
gpu=4

python train_search.py \
--save $save --gpu $gpu \
--dataset CIFAR10