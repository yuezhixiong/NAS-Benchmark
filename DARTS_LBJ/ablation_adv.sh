nohup python -u train_search.py --adv False --save ablation_adv --gpu 1 > ablation_adv/search.out &

echo $!
ps -p $!
while [ $? = 0 ]
do
sleep 30s
ps -p $!
done

nohup python -u train.py --log_save ablation_adv --arch ablation_adv --gpu 1 > ablation_adv/train.out &

echo $!
ps -p $!
while [ $? = 0 ]
do
sleep 30s
ps -p $!
done
