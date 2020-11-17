save=PDARTS_cifar10_time
gpu=1

python train_search_time.py \
--save $save --gpu $gpu \
--layers 5 --add_layers 6 --add_layers 12 --dropout_rate 0.0 --dropout_rate 0.4 --dropout_rate 0.7 \
--original --batch_size 64