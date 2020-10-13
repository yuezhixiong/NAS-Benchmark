save=pgd_nop_max_etp_cutout_batchsize
gpu=10

# python train_search.py --save $save --gpu $gpu --adv PGD --batch_size 64

python copy_genotype.py --save $save

python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary --batch_size 96
