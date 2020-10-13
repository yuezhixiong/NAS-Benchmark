# nohup python -u train_search.py --gpu 3 > search.out &

# echo $!
# ps -p $!
# while [ $? = 0 ]
# do
# sleep 30s
# ps -p $!
# done

# nohup python -u train_cifar.py --gpu 3 > train.out &

# echo $!
# ps -p $!
# while [ $? = 0 ]
# do
# sleep 30s
# ps -p $!
# done

save=adv_nop

python test_adv.py --arch $save --batch_size 32 --gpu 2  --attack PGD \
--model_path "/home/yuezx/NAS-Benchmark/PDARTS/adv_nop_train/weights.pt"