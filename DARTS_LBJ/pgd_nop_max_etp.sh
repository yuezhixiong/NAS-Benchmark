save=pgd_nop_max_etp_cutout_batchsize
gpu=12

# python train_search.py --save $save --gpu $gpu --adv PGD --batch_size 64

# python copy_genotype.py --save $save

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary --batch_size 96

python test_adv.py --arch $save --batch_size 64 --gpu $gpu --attack PGD --cutout --auxiliary \
--model_path "/raid/zy_lab/yzx/NAS-Benchmark/DARTS_LBJ/pgd_nop_max_etp_cutout_batchsize/auxiliary0.4_cutout16_batch_size96/best_model.pt"