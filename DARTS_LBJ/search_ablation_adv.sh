save=search_ablation_adv

# python train_search.py --nop False --gpu 2 --save search_ablation_adv > search_ablation_adv/search.txt

python copy_genotype.py --save $save

python train.py --arch $save --save $save --gpu 2 --cutout --batch_size 96