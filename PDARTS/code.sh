nohup python -u train_search.py --dataset CIFAR10 --batch_size 64 --gpu 0 --epochs 1 > search.out &

echo $!
ps -p $!
while [ $? = 0 ]
do
sleep 30s
ps -p $!
done

nohup python -u train_cifar.py --dataset CIFAR10 --batch_size 64 --epochs 1 > train.out &

echo $!
ps -p $!
while [ $? = 0 ]
do
sleep 30s
ps -p $!
done
