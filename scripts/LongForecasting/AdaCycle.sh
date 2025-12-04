if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/AdaCycle" ]; then
    mkdir ./logs/LongForecasting/AdaCycle
fi


model_name=AdaCycle
seq_len=96
GPU=0
root=./dataset/
train_epochs=20
patience=3



data_name=ETTh1
#ETTh1
for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/ETT-small/ \
    --data_path $data_name.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size 64 \
    --d_model 128 \
    --learning_rate 0.0009 \
    --alpha 0.35 \
    --wv db1 \
    --m 3 \
    --itr 1 > logs/LongForecasting/AdaCycle/$data_name'_'$model_name'_'$pred_len.logs
done



alpha=0.35
data_name=ETTh2
#ETTh2
for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/ETT-small/ \
    --data_path $data_name.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size 64 \
    --d_model 256 \
    --learning_rate 0.0002 \
    --alpha 0.35 \
    --wv db1 \
    --m 2 \
    --itr 1 > logs/LongForecasting/AdaCycle/$data_name'_'$model_name'_'$pred_len.logs
done



data_name=ETTm1
#ETTm1
for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/ETT-small/ \
    --data_path $data_name.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size 64 \
    --d_model 128 \
    --learning_rate 0.0004 \
    --alpha 0.35 \
    --wv db1 \
    --m 1 \
    --itr 1 > logs/LongForecasting/AdaCycle/$data_name'_'$model_name'_'$pred_len.logs
done



data_name=ETTm2
#ETTm2
for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/ETT-small/ \
    --data_path $data_name.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size 64 \
    --d_model 256 \
    --learning_rate 0.00007 \
    --alpha 0.35 \
    --wv db2 \
    --m 3 \
    --itr 1 > logs/LongForecasting/AdaCycle/$data_name'_'$model_name'_'$pred_len.logs
done



data_name=weather
#weather
for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/weather/ \
    --data_path $data_name.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size 64 \
    --d_model 256 \
    --learning_rate 0.0003 \
    --alpha 0.1 \
    --wv db6 \
    --m 3 \
    --itr 1 > logs/LongForecasting/AdaCycle/$data_name'_'$model_name'_'$pred_len.logs
done



data_name=Solar
#Solar
for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/Solar/ \
    --data_path solar_AL.txt \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 137 \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size 32 \
    --d_model 256 \
    --learning_rate 0.0002 \
    --alpha 0.05 \
    --wv db4 \
    --m 2 \
    --use_revin 0 \
    --itr 1 > logs/LongForecasting/AdaCycle/$data_name'_'$model_name'_'$pred_len.logs
done



data_name=electricity
#electricity
for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/electricity/ \
    --data_path $data_name.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 321 \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size 32 \
    --d_model 512 \
    --learning_rate 0.0014 \
    --alpha 0.2 \
    --wv db1 \
    --m 2 \
    --itr 1 > logs/LongForecasting/AdaCycle/$data_name'_'$model_name'_'$pred_len.logs
done




data_name=traffic
#traffic
for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/traffic/ \
    --data_path $data_name.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 862 \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size 16 \
    --d_model 768 \
    --learning_rate 0.0005 \
    --alpha 0.35 \
    --wv db1 \
    --m 2 \
    --itr 1 > logs/LongForecasting/AdaCycle/$data_name'_'$model_name'_'$pred_len.logs
done