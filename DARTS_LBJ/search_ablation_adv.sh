save=search_ablation_adv

# python train_search.py --nop False --gpu 2 --save search_ablation_adv > search_ablation_adv/search.txt

# python copy_genotype.py --save $save

# python train.py --arch $save --save $save --gpu 2 --cutout --batch_size 96

python test_adv.py --arch $save --batch_size 64 --gpu 1 --attack PGD \
--model_path "/home/yuezx/NAS-Benchmark/DARTS_LBJ/search_ablation_adv/20201012-160219/best_model.pt"