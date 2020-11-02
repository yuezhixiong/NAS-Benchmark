save=DARTS_V2
gpu=3

# python train_search.py --save $save --gpu $gpu \
# --batch_size 20 --unrolled --cutout \
# --adv FGSM --nop --MGDA --constrain min

# python copy_genotype.py --save $save

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 9

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 12

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 15

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 20

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 28

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 48 --init_channels 36

python test_adv.py --arch $save --batch_size 16 --gpu $gpu --auxiliary --cutout \
--attack FGSM --init_channels 28 \
--model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/DARTS_V2/auxiliary0.4_cutout16_batchsize96_channel28_cifar10/best_model.pt"

python test_adv.py --arch $save --batch_size 16 --gpu $gpu --auxiliary --cutout \
--attack PGD --init_channels 28 \
--model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/DARTS_V2/auxiliary0.4_cutout16_batchsize96_channel28_cifar10/best_model.pt"

# python test_adv.py --arch $save --batch_size 16 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 36 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/DARTS_V2/auxiliary0.4_cutout16_batchsize48_channel36_cifar10/best_model.pt"

# python test_adv.py --arch $save --batch_size 32 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 20 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/adv_nop_min/auxiliary0.4_cutout16_batchsize50_channel20/best_model.pt"

# python test_adv.py --arch $save --batch_size 18 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 36 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/adv_nop_min/auxiliary0.4_cutout16_batchsize50_channel36/best_model.pt"