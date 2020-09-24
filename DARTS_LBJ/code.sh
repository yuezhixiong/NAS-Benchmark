nohup python -u train_search.py --data ../data --batch_size 64 --gpu 0 --unrolled --epochs 1 --save EXP > search.out &

echo $!
ps -p $!
while [ $? = 0 ]
do
sleep 30s
ps -p $!
done

nohup python -u train.py --data ../data --batch_size 64 --gpu 0 --epochs 1 --log_save EXP > train.out &

echo $!
ps -p $!
while [ $? = 0 ]
do
sleep 30s
ps -p $!
done
