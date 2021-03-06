save=pdarts_e2rnas_oodInOut
gpu=0

python train_search.py \
--save $save --gpu $gpu --batch_size 48 \
--layers 5 --add_layers 6 --add_layers 12 --dropout_rate 0.0 --dropout_rate 0.4 --dropout_rate 0.7 \
--ood_inner --nop_outer --adv_outer --flp_outer --ood_outer --MGDA --constrain abs --constrain_min 3 --grad_norm l2

python copy_genotype.py --save $save

python train.py \
--save $save --dataset cifar10 --layers 20 \
--arch $save --gpu $gpu --auxiliary --cutout 

model=$save/cifar10_channel36/weights.pt
python test_adv.py --arch $save --gpu $gpu \
--model_path $model

python test_ood.py --arch $save --gpu $gpu \
--model_path $model

python test_flp.py --arch $save --gpu $gpu \
--model_path $model