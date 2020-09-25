nohup python -u train_search.py --nop False --save ablation_nop --gpu 2 > ablation_nop/search.out &

echo $!
ps -p $!
while [ $? = 0 ]
do
sleep 30s
ps -p $!
done

nohup python -u train.py --log_save ablation_nop --gpu 2 > ablation_nop/train.out &

echo $!
ps -p $!
while [ $? = 0 ]
do
sleep 30s
ps -p $!
done
