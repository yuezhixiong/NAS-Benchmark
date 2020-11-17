save=DARTS_V2_cifar10
gpu=3

python train_search.py --save $save --gpu $gpu \
--batch_size 64 --unrolled --cutout