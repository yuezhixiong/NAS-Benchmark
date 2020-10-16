save=adv_nop_min
gpu=1

python train_search.py --save $save --gpu $gpu \
--batch_size 20 --unrolled --cutout \
--adv FGSM --nop --MGDA --grad_norm --constrain min

# python copy_genotype.py --save $save

# python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary --batch_size 96

# python test_adv.py --arch $save --batch_size 32 --gpu $gpu --attack PGD --auxiliary --cutout \
# --model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/adv_nop/auxiliary0.4_cutout16_batchsize96/best_model.pt"