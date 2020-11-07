save=fast0_nop_min1e6_outer
gpu=5

python train_search.py --save $save --gpu $gpu \
--batch_size 64 --unrolled --cutout \
--adv fast --inner_lambda 0 \
--nop_outer --MGDA --constrain min --adv_outer

python copy_genotype.py --save $save

python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary \
--batch_size 96 --init_channels 36

# python test_adv.py --arch $save --batch_size 128 --gpu $gpu --auxiliary --cutout \
# --attack PGD --init_channels 46 \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/fgsm_nop_min/auxiliary0.4_cutout16_batchsize96_channel46_cifar10/best_model.pt"