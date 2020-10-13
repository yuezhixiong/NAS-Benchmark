# nohup python -u train_search.py --gpu 2 > search.out &

# echo $!
# ps -p $!
# while [ $? = 0 ]
# do
# sleep 30s
# ps -p $!
# done

# nohup python -u train.py --gpu 2 > train.out &

# echo $!
# ps -p $!
# while [ $? = 0 ]
# do
# sleep 30s
# ps -p $!
# done

save=adv_nop

python test_adv.py --arch $save --batch_size 64 --gpu 2  --attack PGD \
--model_path "/home/yuezx/NAS-Benchmark/PC-DARTS/saved_models/weights.pt"