save=PCDARTS
gpu=0

# python train_search.py \
# --save $save --gpu $gpu --dataset cifar10

python train.py \
--save $save --dataset cifar10 --layers 20 \
--arch $save --gpu $gpu --auxiliary --cutout

model="PCDARTS/channel36_cifar10/weights.pt"
python test_adv.py --arch $save --gpu $gpu \
--model_path $model

python test_ood.py --arch $save --gpu $gpu \
--model_path $model

python test_flp.py --arch $save --gpu $gpu \
--model_path $model