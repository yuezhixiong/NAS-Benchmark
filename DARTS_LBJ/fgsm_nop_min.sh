save=fgsm_nop_min
gpu=4

# python train_search.py --save $save --gpu $gpu \
# --batch_size 64 --unrolled --cutout \
# --adv FGSM --nop --MGDA --constrain min

# python copy_genotype.py --save $save

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 16

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 25

python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
--attack FGSM --init_channels 25 \
--model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/fgsm_nop_min/auxiliary0.4_cutout16_batchsize96_channel25_cifar10/best_model.pt"

python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
--attack PGD --init_channels 25 \
--model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/fgsm_nop_min/auxiliary0.4_cutout16_batchsize96_channel25_cifar10/best_model.pt"


# python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
# --attack FGSM --init_channels 16 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/fgsm_nop_min/auxiliary0.4_cutout16_batchsize96_channel16_cifar10/best_model.pt"

# python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 16 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/fgsm_nop_min/auxiliary0.4_cutout16_batchsize96_channel16_cifar10/best_model.pt"

# python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 36 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/fgsm_nop_min/auxiliary0.4_cutout16_batchsize96_channel36_cifar10/best_model.pt"

# python test_adv.py --arch $save --batch_size 32 --gpu $gpu --auxiliary --cutout \
# --attack FGSM --init_channels 20 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/adv_nop_min/auxiliary0.4_cutout16_batchsize50_channel20/best_model.pt"

# python test_adv.py --arch $save --batch_size 18 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 36 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/adv_nop_min/auxiliary0.4_cutout16_batchsize50_channel36/best_model.pt"