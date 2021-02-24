save=pdarts_e2rnas
gpu=0

python train_search.py \
--save $save --gpu $gpu \
--layers 5 --add_layers 6 --add_layers 12 --dropout_rate 0.0 --dropout_rate 0.4 --dropout_rate 0.7 \
--nop_outer --adv_outer --flp_outer --ood_inner --MGDA --constrain abs --constrain_min 3 --grad_norm l2

python copy_genotype.py --save $save

python train_cifar.py \
--save $save --dataset cifar10 --layers 20 \
--arch $save --gpu $gpu --auxiliary --cutout 

# python test_adv.py --arch $save --gpu $gpu --auxiliary --cutout \
# --attack PGD --batch_size 32 \
# --model_path "/home/yuezx/NAS-Benchmark/PDARTS/pdarts_fgsm_nop/20201103-132845/weights.pt"