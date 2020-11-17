save=PC_DARTS_cifar10
gpu=6

python train_search.py \
--save $save --gpu $gpu --dataset CIFAR10 \
--original