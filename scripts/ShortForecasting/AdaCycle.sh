if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ShortForecasting" ]; then
    mkdir ./logs/ShortForecasting
fi

if [ ! -d "./logs/ShortForecasting/AdaCycle" ]; then
    mkdir ./logs/ShortForecasting/AdaCycle
fi



model_name=AdaCycle

root_path_name=./dataset/PEMS
data_name=PEMS

seq_len=96
pred_len=12
train_epochs=10
patience=3
batch_size=32



#PEMS03
data_path_name=PEMS03.npz
model_id_name=PEMS03
python -u run.py \
  --is_training 1 \
  --root_path $root_path_name\
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 358 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --d_model 256 \
  --learning_rate 0.002 \
  --alpha 0 \
  --wv db2 \
  --dropout 0.15 \
  --m 3 \
  --itr 1 > logs/ShortForecasting/AdaCycle/$model_id_name'_'$model_name'_'$pred_len.logs




#PEMS04
data_path_name=PEMS04.npz
model_id_name=PEMS04
python -u run.py \
  --is_training 1 \
  --root_path $root_path_name\
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 307 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --d_model 256 \
  --learning_rate 0.0013 \
  --alpha 0.05 \
  --wv db4 \
  --m 1 \
  --use_revin 0 \
  --dropout 0.2 \
  --itr 1 > logs/ShortForecasting/AdaCycle/$model_id_name'_'$model_name'_'$pred_len.logs




#PEMS07
data_path_name=PEMS07.npz
model_id_name=PEMS07
python -u run.py \
  --is_training 1 \
  --root_path $root_path_name\
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 883 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --d_model 512 \
  --learning_rate 0.001 \
  --alpha 0 \
  --wv db4 \
  --m 1 \
  --use_revin 0 \
  --itr 1 > logs/ShortForecasting/AdaCycle/$model_id_name'_'$model_name'_'$pred_len.logs



#PEMS08
data_path_name=PEMS08.npz
model_id_name=PEMS08
python -u run.py \
  --is_training 1 \
  --root_path $root_path_name\
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 170 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --d_model 256 \
  --learning_rate 0.0018 \
  --alpha 0 \
  --wv db4 \
  --m 1 \
  --itr 1 > logs/ShortForecasting/AdaCycle/$model_id_name'_'$model_name'_'$pred_len.logs





