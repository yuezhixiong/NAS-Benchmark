save=lossNorm_pcdarts_e2rnas_cifar100
gpu=0

# python train_search.py \
# --save $save --gpu $gpu --dataset cifar100 --cutout \
# --nop_outer --adv_outer --flp_outer --ood_lambda 1 --MGDA --constrain abs --constrain_min 3 --grad_norm l2

# python copy_genotype.py --save $save

python train.py \
--dataset cifar100 --gpu $gpu --save $save --arch $save \
--auxiliary --cutout --layers 20 #20 for CIFAR datasets, 8 for Sport8, MIT67 and flowers102

model=$save/channel36_cifar10/weights.pt
python test_adv.py --arch $save --gpu $gpu \
--model_path $model

python test_ood.py --arch $save --gpu $gpu \
--model_path $model

python test_flp.py --arch $save --gpu $gpu \
--model_path $model