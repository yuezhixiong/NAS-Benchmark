save=PDARTS_svhn
gpu=0

# python train_cifar.py \
# --save $save --dataset svhn --layers 20 \
# --arch $save --gpu $gpu --auxiliary --cutout 
# layers 20 for cifar10 and cifar100, 8 for sport8, mit67 and flowers102

python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
--attack PGD --init_channels 36 --dataset svhn --layers 20 \
--model_path "/home/yuezx/NAS-Benchmark/PDARTS/PDARTS_svhn/batchsize32_channel36/weights.pt"