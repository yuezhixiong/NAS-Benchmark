nohup python -u train_search.py --gpu 1 > adv_nop_etp/search.out &

echo $!
ps -p $!
while [ $? = 0 ]
do
sleep 30s
ps -p $!
done

nohup python -u train.py --gpu 1 > adv_nop_etp/train.out &

echo $!
ps -p $!
while [ $? = 0 ]
do
sleep 30s
ps -p $!
done
