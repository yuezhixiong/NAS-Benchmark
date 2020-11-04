save=PDARTS_cifar100
gpu=6

# python train_search.py \
# --save $save --gpu $gpu \
# --layers 5 --add_layers 6 --add_layers 12 --dropout_rate 0.0 --dropout_rate 0.4 --dropout_rate 0.7 \
# --fgsm --MGDA --constrain min

# python copy_genotype.py --save $save

python train_cifar.py \
--save $save --dataset CIFAR100 --layers 20 \
--arch $save --gpu $gpu --auxiliary --cutout 
# layers 20 for cifar10 and cifar100, 8 for sport8, mit67 and flowers102