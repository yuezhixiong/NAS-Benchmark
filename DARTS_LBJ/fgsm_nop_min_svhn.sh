save=fgsm_nop_min_svhn
gpu=4

# python train_search.py --save $save --gpu $gpu \
# --batch_size 64 --unrolled --cutout \
# --adv FGSM --MGDA --constrain min \
# --dataset svhn

# python copy_genotype.py --save $save

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 36 --dataset svhn

# python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 36 --dataset svhn \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/fgsm_nop_min_svhn/auxiliary0.4_cutout16_batchsize96_channel36_svhn/best_model.pt"

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 30 --dataset svhn

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 21 --dataset svhn

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 39 --dataset svhn

# python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 21 --dataset svhn \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/fgsm_nop_min_svhn/auxiliary0.4_cutout16_batchsize96_channel21_svhn/best_model.pt"

# python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 30 --dataset svhn \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/fgsm_nop_min_svhn/auxiliary0.4_cutout16_batchsize96_channel30_svhn/best_model.pt"

# python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 39 --dataset svhn \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/fgsm_nop_min_svhn/auxiliary0.4_cutout16_batchsize96_channel39_svhn/best_model.pt"

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 39 --dataset svhn

python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
--attack PGD --init_channels 39 --dataset svhn \
--model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/fgsm_nop_min_svhn/auxiliary0.4_cutout16_batchsize96_channel39_svhn/best_model.pt"