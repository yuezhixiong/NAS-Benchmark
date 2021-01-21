save=nop
gpu=1

# python train_search.py --save $save --gpu $gpu \
# --batch_size 64 --unrolled --cutout \
# --nop

# python copy_genotype.py --save $save

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
# --batch_size 96 --init_channels 36

# python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
# --attack FGSM --init_channels 36 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/fgsm_nop_min8e5/auxiliary0.4_cutout16_batchsize96_channel36_cifar10/best_model.pt"

python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
--attack PGD --init_channels 36 \
--model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/nop/auxiliary0.4_cutout16_batchsize96_channel36_cifar10/best_model.pt"