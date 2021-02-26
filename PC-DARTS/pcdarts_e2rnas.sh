save=pcdarts_e2rnas
gpu=7

python train_search.py \
--save $save --gpu $gpu --dataset cifar10 --cutout \
--nop_outer --adv_outer --flp_outer --ood_lambda 1 --MGDA --constrain abs --constrain_min 3 --grad_norm l2

python train.py \
--dataset cifar10 --gpu $gpu --save $save --arch $save \
--auxiliary --cutout --layers 20 #20 for CIFAR datasets, 8 for Sport8, MIT67 and flowers102


# python test_adv.py --arch pcdarts_fgsm_nop \
# --dataset cifar10 --gpu $gpu \
# --auxiliary --cutout --layers 20 --attack PGD \
# --model_path "/home/yuezx/NAS-Benchmark/PC-DARTS/pcdarts_fgsm_nop/batchsize32_channel36_CIFAR10/weights.pt"