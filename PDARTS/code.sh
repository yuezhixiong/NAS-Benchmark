nohup python -u train_search.py --gpu 3 --epochs 1 > search.out &

echo $!
ps -p $!
while [ $? = 0 ]
do
sleep 30s
ps -p $!
done

nohup python -u train_cifar.py --gpu 3 > train.out &

echo $!
ps -p $!
while [ $? = 0 ]
do
sleep 30s
ps -p $!
done
