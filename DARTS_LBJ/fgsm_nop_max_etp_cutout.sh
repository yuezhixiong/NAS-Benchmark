save=fgsm_nop_max_etp_cutout_batchsize

# python train_search.py --save fgsm_nop_max_etp_cutout_batchsize \
# --gpu 10 --adv FGSM --cutout --batch_size 80

# python copy_genotype.py --save $save

python train.py --arch $save --save $save --gpu 8 --cutout --batch_size 96