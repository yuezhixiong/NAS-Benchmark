save=PCDARTS_cifar10
gpu=6

# python train_search.py \
# --save $save --gpu $gpu --dataset cifar10

python train.py \
--save $save --dataset cifar10 --layers 20 \
--arch $save --gpu $gpu --auxiliary --cutout