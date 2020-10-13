save=ablation_mgda
gpu=12

python train_search.py --save $save --gpu $gpu --batch_size 64 \
--adv none --MGDA false --entropy false --max_constraint false

python copy_genotype.py --save $save

python train.py --arch $save --save $save --gpu $gpu --cutout --auxiliary --batch_size 96

# python test_adv.py --arch $save --batch_size 128 --gpu 12  \ 
# --model_path "fgsm_nop_max_etp_cutout_batchsize/20201012-104509/best_model.pt"
