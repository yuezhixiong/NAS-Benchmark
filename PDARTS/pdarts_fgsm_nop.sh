save=pdarts_fgsm_nop
gpu=3

python train_search.py \
--save $save --gpu $gpu \
--layers 5 --add_layers 6 --add_layers 12 --dropout_rate 0.0 --dropout_rate 0.4 --dropout_rate 0.7 \
--fgsm --MGDA --constrain min

python copy_genotype.py --save $save

python train_cifar.py \
--save $save --dataset CIFAR10 --layers 20 \
--arch $save --gpu $gpu --auxiliary --cutout 