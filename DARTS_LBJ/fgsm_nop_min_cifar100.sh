save=fgsm_nop_min_cifar100
gpu=6

# python train_search.py --save $save --gpu $gpu \
# --batch_size 64 --unrolled --cutout \
# --adv FGSM --nop --MGDA --constrain min \
# --dataset cifar100

# python copy_genotype.py --save $save

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 36 --dataset cifar100

# python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 36 --dataset cifar100 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/fgsm_nop_min_cifar100/auxiliary0.4_cutout16_batchsize96_channel36_cifar100/best_model.pt"

# python test_adv.py --arch $save --batch_size 32 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 20 --dataset cifar100 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/fgsm_nop_min_cifar100/auxiliary0.4_cutout16_batchsize96_channel20_cifar100/best_model.pt"

# python test_adv.py --arch $save --batch_size 32 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 29 --dataset cifar100 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/fgsm_nop_min_cifar100/auxiliary0.4_cutout16_batchsize96_channel29_cifar100/best_model.pt"

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 38 --dataset cifar100

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 29 --dataset cifar100

# python test_adv.py --arch $save --batch_size 32 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 38 --dataset cifar100 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/fgsm_nop_min_cifar100/auxiliary0.4_cutout16_batchsize96_channel38_cifar100/best_model.pt"

python test_adv.py --arch $save --batch_size 32 --gpu $gpu --auxiliary --cutout \
--attack PGD --init_channels 29 --dataset cifar100 \
--model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/fgsm_nop_min_cifar100/auxiliary0.4_cutout16_batchsize96_channel29_cifar100/best_model.pt"