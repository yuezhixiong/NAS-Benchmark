save=PCDARTS_cifar100
gpu=0

# python train_search.py \
# --save $save --gpu $gpu --dataset cifar10
#--batch_size 48
# python train.py \
# --save $save --dataset cifar100 --layers 20 \
# --arch $save --gpu $gpu --auxiliary --cutout

model="PCDARTS_cifar100/channel36_cifar100/best_model.pt"
python test_adv.py --arch $save --gpu $gpu \
--model_path $model --dataset cifar100

python test_ood.py --arch $save --gpu $gpu \
--model_path $model --dataset cifar100

python test_flp.py --arch $save --gpu $gpu \
--model_path $model --dataset cifar100