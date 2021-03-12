save=pcdarts_e2rnas_1
gpu=6

python train_search.py \
--save $save --gpu $gpu --dataset cifar10 --cutout \
--nop_outer --adv_outer --flp_outer --ood_lambda 1 --MGDA --constrain abs --constrain_min 3 --grad_norm l2

# python train.py \
# --dataset cifar10 --gpu $gpu --save $save --arch $save \
# --auxiliary --cutout --layers 20 #20 for CIFAR datasets, 8 for Sport8, MIT67 and flowers102

model="pcdarts_e2rnas_1/batchsize96_channel36_cifar10/weights.pt"
python test_adv.py --arch $save --gpu $gpu \
--model_path $model

python test_ood.py --arch $save --gpu $gpu \
--model_path $model

python test_flp.py --arch $save --gpu $gpu \
--model_path $model