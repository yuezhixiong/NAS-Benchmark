save=PDARTS_cifar100
gpu=0

python train.py \
--save $save --dataset cifar100 --layers 20 \
--arch $save --gpu $gpu --auxiliary --cutout 

model="PDARTS_cifar100/cifar100_channel36/weights.pt"
python test_adv.py --arch $save --gpu $gpu \
--model_path $model --dataset cifar100

python test_ood.py --arch $save --gpu $gpu \
--model_path $model --dataset cifar100

python test_flp.py --arch $save --gpu $gpu \
--model_path $model