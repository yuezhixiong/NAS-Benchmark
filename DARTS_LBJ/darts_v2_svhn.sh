save=darts_v2_svhn
gpu=6

# python train_search.py --save $save --gpu $gpu \
# --batch_size 64 --unrolled --cutout --dataset svhn

# python copy_genotype.py --save $save

# # python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# # --batch_size 96 --init_channels 12

# # python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# # --batch_size 96 --init_channels 20

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 36 --dataset svhn

# python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 34 --dataset svhn \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/DARTS_V2/auxiliary0.4_cutout16_batchsize96_channel34_cifar100/best_model.pt"

# python test_adv.py --arch $save --batch_size 18 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 36 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/adv_nop_min/auxiliary0.4_cutout16_batchsize50_channel36/best_model.pt"

# python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 34 --dataset svhn \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/DARTS_V2/auxiliary0.4_cutout16_batchsize96_channel34_svhn/best_model.pt"

# python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 28 --dataset svhn \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/DARTS_V2/auxiliary0.4_cutout16_batchsize96_channel28_svhn/best_model.pt"

# python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 20 --dataset svhn \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/DARTS_V2/auxiliary0.4_cutout16_batchsize96_channel20_svhn/best_model.pt"

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 28 --dataset svhn

python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
--attack PGD --init_channels 28 --dataset svhn \
--model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/darts_v2_svhn/auxiliary0.4_cutout16_batchsize96_channel28_svhn/best_model.pt"