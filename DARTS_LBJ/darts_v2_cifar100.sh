save=darts_v2_cifar100
gpu=7

# python train_search.py --save $save --gpu $gpu \
# --batch_size 64 --unrolled --cutout --dataset cifar100

# python copy_genotype.py --save $save

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 36 --dataset cifar100

# python test_adv.py --arch $save --batch_size 64 --gpu $gpu --auxiliary --cutout \
# --attack FGSM --init_channels 36 --dataset cifar100 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/darts_v2_cifar100/auxiliary0.4_cutout16_batchsize96_channel36_cifar100/best_model.pt"

# python test_adv.py --arch $save --batch_size 64 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 36 --dataset cifar100 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/darts_v2_cifar100/auxiliary0.4_cutout16_batchsize96_channel36_cifar100/best_model.pt"

# python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 27 --dataset cifar100 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/DARTS_V2/auxiliary0.4_cutout16_batchsize96_channel27_cifar100/best_model.pt"

python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
--attack PGD --init_channels 34 --dataset cifar100 \
--model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/DARTS_V2/auxiliary0.4_cutout16_batchsize96_channel34_cifar100/best_model.pt"