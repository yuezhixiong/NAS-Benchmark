save=darts_v2_svhn
gpu=3

# python train_search.py --save $save --gpu $gpu \
# --batch_size 64 --unrolled --cutout --dataset svhn

# python copy_genotype.py --save $save

# # python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# # --batch_size 96 --init_channels 12

# # python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# # --batch_size 96 --init_channels 20

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 36 --dataset svhn

python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
--attack FGSM --init_channels 36 --dataset svhn \
--model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/darts_v2_svhn/auxiliary0.4_cutout16_batchsize96_channel36_svhn/best_model.pt"

python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
--attack PGD --init_channels 36 --dataset svhn \
--model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/darts_v2_svhn/auxiliary0.4_cutout16_batchsize96_channel36_svhn/best_model.pt"

# python test_adv.py --arch $save --batch_size 32 --gpu $gpu --auxiliary --cutout \
# --attack FGSM --init_channels 20 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/adv_nop_min/auxiliary0.4_cutout16_batchsize50_channel20/best_model.pt"

# python test_adv.py --arch $save --batch_size 18 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 36 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/adv_nop_min/auxiliary0.4_cutout16_batchsize50_channel36/best_model.pt"