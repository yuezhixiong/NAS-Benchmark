save=PC_DARTS_cifar10_time
gpu=4

python train_search_time.py \
--save $save --gpu $gpu --dataset CIFAR10 \
--original --batch_size 64