nohup python -u train_search.py --gpu 2 > search.out &

echo $!
ps -p $!
while [ $? = 0 ]
do
sleep 30s
ps -p $!
done

nohup python -u train.py --gpu 2 > train.out &

echo $!
ps -p $!
while [ $? = 0 ]
do
sleep 30s
ps -p $!
done
