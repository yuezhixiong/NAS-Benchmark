nohup python -u train_search.py --adv False --nop False --save ablation_org --gpu 3 > ablation_org/search.out &

echo $!
ps -p $!
while [ $? = 0 ]
do
sleep 30s
ps -p $!
done

nohup python -u train.py --log_save ablation_org --gpu 3 > ablation_org/train.out &

echo $!
ps -p $!
while [ $? = 0 ]
do
sleep 30s
ps -p $!
done
