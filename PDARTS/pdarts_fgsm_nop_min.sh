save=pdarts_fgsm_nop_min
gpu=3

# python train_search.py \
# --save $save --gpu $gpu \
# --layers 5 --add_layers 6 --add_layers 12 --dropout_rate 0.0 --dropout_rate 0.4 --dropout_rate 0.7

# python train_cifar.py \
# --save $save --dataset CIFAR10 --layers 20 \
# --arch $save --gpu $gpu --auxiliary --cutout 

python test_adv.py -arch $save --gpu $gpu --auxiliary --cutout \
--attack PGD --batch_size 128 \
--model_path "/home/yuezx/NAS-Benchmark/PDARTS/pdarts_fgsm_nop/20201103-132845/weights.pt"