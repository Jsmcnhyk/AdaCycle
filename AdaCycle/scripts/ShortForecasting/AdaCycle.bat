@echo off
setlocal enabledelayedexpansion

if not exist ".\logs" mkdir .\logs
if not exist ".\logs\ShortForecasting" mkdir .\logs\ShortForecasting
if not exist ".\logs\ShortForecasting\AdaCycle" mkdir .\logs\ShortForecasting\AdaCycle

set model_name=AdaCycle
set data_name=PEMS
set root=./dataset/PEMS
set seq_len=96
set train_epochs=10
set patience=3
set batch_size=32
set GPU=0

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: PEMS03 experiments
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
set data_path_name=PEMS03.npz
set model_id_name=PEMS03

for %%p in (12) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/ ^
    --data_path !data_path_name! ^
    --model_id !model_id_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data !data_name! ^
    --features M ^
    --seq_len !seq_len! ^
    --pred_len %%p ^
    --enc_in 358 ^
    --train_epochs !train_epochs! ^
    --patience !patience! ^
    --batch_size !batch_size! ^
    --d_model 256 ^
    --learning_rate 0.002 ^
    --alpha 0 ^
    --wv db2 ^
    --dropout 0.15 ^
    --m 3 ^
    --itr 1 > logs\ShortForecasting\AdaCycle\!model_id_name!_!model_name!_%%p.logs
)

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: PEMS04 experiments
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
set data_path_name=PEMS04.npz
set model_id_name=PEMS04

for %%p in (12) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/ ^
    --data_path !data_path_name! ^
    --model_id !model_id_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data !data_name! ^
    --features M ^
    --seq_len !seq_len! ^
    --pred_len %%p ^
    --enc_in 307 ^
    --train_epochs !train_epochs! ^
    --patience !patience! ^
    --batch_size !batch_size! ^
    --d_model 256 ^
    --learning_rate 0.0013 ^
    --alpha 0.05 ^
    --wv db4 ^
    --m 1 ^
    --use_revin 0 ^
    --dropout 0.2 ^
    --itr 1 > logs\ShortForecasting\AdaCycle\!model_id_name!_!model_name!_%%p.logs
)

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: PEMS07 experiments
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
set data_path_name=PEMS07.npz
set model_id_name=PEMS07

for %%p in (12) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/ ^
    --data_path !data_path_name! ^
    --model_id !model_id_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data !data_name! ^
    --features M ^
    --seq_len !seq_len! ^
    --pred_len %%p ^
    --enc_in 883 ^
    --train_epochs !train_epochs! ^
    --patience !patience! ^
    --batch_size !batch_size! ^
    --d_model 512 ^
    --learning_rate 0.001 ^
    --alpha 0 ^
    --wv db4 ^
    --m 1 ^
    --use_revin 0 ^
    --itr 1 > logs\ShortForecasting\AdaCycle\!model_id_name!_!model_name!_%%p.logs
)

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: PEMS08 experiments
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
set data_path_name=PEMS08.npz
set model_id_name=PEMS08

for %%p in (12) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/ ^
    --data_path !data_path_name! ^
    --model_id !model_id_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data !data_name! ^
    --features M ^
    --seq_len !seq_len! ^
    --pred_len %%p ^
    --enc_in 170 ^
    --train_epochs !train_epochs! ^
    --patience !patience! ^
    --batch_size !batch_size! ^
    --d_model 256 ^
    --learning_rate 0.0018 ^
    --alpha 0 ^
    --wv db4 ^
    --m 1 ^
    --itr 1 > logs\ShortForecasting\AdaCycle\!model_id_name!_!model_name!_%%p.logs
)

echo All PEMS experiments completed!
endlocal
