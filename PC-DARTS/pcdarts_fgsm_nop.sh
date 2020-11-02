save=pcdarts_fgsm_nop
gpu=6

python train_search.py \
--save $save --gpu $gpu --dataset CIFAR10 \
--fgsm --MGDA --constrain min

python copy_genotype.py --save $save

python train.py \
--dataset CIFAR10 --gpu $gpu --save $save --arch $save \
--auxiliary --cutout --layers 20 #20 for CIFAR datasets, 8 for Sport8, MIT67 and flowers102