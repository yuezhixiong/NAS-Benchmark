save=PDARTS_cifar10
gpu=0

# python train_search.py \
# --save $save --gpu $gpu \
# --layers 5 --add_layers 6 --add_layers 12 --dropout_rate 0.0 --dropout_rate 0.4 --dropout_rate 0.7

python train_cifar.py \
--save $save --dataset cifar10 --layers 20 \
--arch $save --gpu $gpu --auxiliary --cutout 