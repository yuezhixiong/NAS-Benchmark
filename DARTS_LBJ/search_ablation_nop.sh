save=search_ablation_nop

# python train_search.py --adv False --gpu 1 --save search_ablation_nop > search_ablation_nop/search.txt

# python copy_genotype.py --save $save

# python train.py --arch $save --save $save --gpu 1 --cutout --batch_size 96

python test_adv.py --arch $save --batch_size 64 --gpu 1 --attack PGD \
--model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/search_ablation_nop/20201012-112350/best_model.pt"