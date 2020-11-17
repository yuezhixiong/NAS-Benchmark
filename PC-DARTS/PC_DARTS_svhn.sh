save=PC_DARTS_svhn
gpu=7

# python train_search.py \
# --save $save --gpu $gpu \
# --dataset CIFAR10

# python train.py --save $save --gpu $gpu --arch $save \
# --auxiliary --cutout \
# --dataset svhn --layers 20 #20 for CIFAR datasets, 8 for Sport8, MIT67 and flowers102

python test_adv.py --arch $save --gpu $gpu --auxiliary --cutout \
--attack PGD --batch_size 128 --dataset svhn \
--model_path "/home/yuezx/NAS-Benchmark/PC-DARTS/PC_DARTS_svhn/batchsize32_channel36_svhn/weights.pt"