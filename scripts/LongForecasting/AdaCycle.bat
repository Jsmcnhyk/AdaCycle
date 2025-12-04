@echo off
setlocal enabledelayedexpansion

REM Create log folders
if not exist ".\logs" mkdir .\logs
if not exist ".\logs\LongForecasting" mkdir .\logs\LongForecasting
if not exist ".\logs\LongForecasting\AdaCycle" mkdir .\logs\LongForecasting\AdaCycle

set model_name=AdaCycle
set seq_len=96
set GPU=0
set root=./dataset/
set train_epochs=20
set patience=3


REM =============================
REM ETTh1
REM =============================
set data_name=ETTh1
for %%p in (96 192 336 720) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/ETT-small/ ^
    --data_path !data_name!.csv ^
    --model_id !data_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data !data_name! ^
    --features M ^
    --seq_len !seq_len! ^
    --pred_len %%p ^
    --enc_in 7 ^
    --train_epochs !train_epochs! ^
    --patience !patience! ^
    --batch_size 64 ^
    --d_model 128 ^
    --learning_rate 0.0009 ^
    --alpha 0.35 ^
    --wv db1 ^
    --m 3 ^
    --itr 1 > logs\LongForecasting\AdaCycle\!data_name!_!model_name!_%%p.logs
)


REM =============================
REM ETTh2
REM =============================
set data_name=ETTh2
for %%p in (96 192 336 720) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/ETT-small/ ^
    --data_path !data_name!.csv ^
    --model_id !data_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data !data_name! ^
    --features M ^
    --seq_len !seq_len! ^
    --pred_len %%p ^
    --enc_in 7 ^
    --train_epochs !train_epochs! ^
    --patience !patience! ^
    --batch_size 64 ^
    --d_model 256 ^
    --learning_rate 0.0002 ^
    --alpha 0.35 ^
    --wv db1 ^
    --m 2 ^
    --itr 1 > logs\LongForecasting\AdaCycle\!data_name!_!model_name!_%%p.logs
)


REM =============================
REM ETTm1
REM =============================
set data_name=ETTm1
for %%p in (96 192 336 720) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/ETT-small/ ^
    --data_path !data_name!.csv ^
    --model_id !data_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data !data_name! ^
    --features M ^
    --seq_len !seq_len! ^
    --pred_len %%p ^
    --enc_in 7 ^
    --train_epochs !train_epochs! ^
    --patience !patience! ^
    --batch_size 64 ^
    --d_model 128 ^
    --learning_rate 0.0004 ^
    --alpha 0.35 ^
    --wv db1 ^
    --m 1 ^
    --itr 1 > logs\LongForecasting\AdaCycle\!data_name!_!model_name!_%%p.logs
)


REM =============================
REM ETTm2
REM =============================
set data_name=ETTm2
for %%p in (96 192 336 720) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/ETT-small/ ^
    --data_path !data_name!.csv ^
    --model_id !data_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data !data_name! ^
    --features M ^
    --seq_len !seq_len! ^
    --pred_len %%p ^
    --enc_in 7 ^
    --train_epochs !train_epochs! ^
    --patience !patience! ^
    --batch_size 64 ^
    --d_model 256 ^
    --learning_rate 0.00007 ^
    --alpha 0.35 ^
    --wv db2 ^
    --m 3 ^
    --itr 1 > logs\LongForecasting\AdaCycle\!data_name!_!model_name!_%%p.logs
)


REM =============================
REM Weather
REM =============================
set data_name=weather
for %%p in (96 192 336 720) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/weather/ ^
    --data_path !data_name!.csv ^
    --model_id !data_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data custom ^
    --features M ^
    --seq_len !seq_len! ^
    --pred_len %%p ^
    --enc_in 21 ^
    --train_epochs !train_epochs! ^
    --patience !patience! ^
    --batch_size 64 ^
    --d_model 256 ^
    --learning_rate 0.0003 ^
    --alpha 0.1 ^
    --wv db6 ^
    --m 3 ^
    --itr 1 > logs\LongForecasting\AdaCycle\!data_name!_!model_name!_%%p.logs
)


REM =============================
REM Solar
REM =============================
set data_name=Solar
for %%p in (96 192 336 720) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/Solar/ ^
    --data_path solar_AL.txt ^
    --model_id !data_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data Solar ^
    --features M ^
    --seq_len !seq_len! ^
    --pred_len %%p ^
    --enc_in 137 ^
    --train_epochs !train_epochs! ^
    --patience !patience! ^
    --batch_size 32 ^
    --d_model 256 ^
    --learning_rate 0.0002 ^
    --alpha 0.05 ^
    --wv db4 ^
    --m 2 ^
    --use_revin 0 ^
    --itr 1 > logs\LongForecasting\AdaCycle\!data_name!_!model_name!_%%p.logs
)


REM =============================
REM Electricity
REM =============================
set data_name=electricity
for %%p in (96 192 336 720) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/electricity/ ^
    --data_path !data_name!.csv ^
    --model_id !data_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data custom ^
    --features M ^
    --seq_len !seq_len! ^
    --pred_len %%p ^
    --enc_in 321 ^
    --train_epochs !train_epochs! ^
    --patience !patience! ^
    --batch_size 32 ^
    --d_model 512 ^
    --learning_rate 0.0014 ^
    --alpha 0.2 ^
    --wv db1 ^
    --m 2 ^
    --itr 1 > logs\LongForecasting\AdaCycle\!data_name!_!model_name!_%%p.logs
)


REM =============================
REM Traffic
REM =============================
set data_name=traffic
for %%p in (96 192 336 720) do (
  set CUDA_VISIBLE_DEVICES=!GPU!
  python -u run.py ^
    --is_training 1 ^
    --root_path !root!/traffic/ ^
    --data_path !data_name!.csv ^
    --model_id !data_name!_!seq_len!_%%p ^
    --model !model_name! ^
    --data custom ^
    --features M ^
    --seq_len !seq_len! ^
    --pred_len %%p ^
    --enc_in 862 ^
    --train_epochs !train_epochs! ^
    --patience !patience! ^
    --batch_size 16 ^
    --d_model 768 ^
    --learning_rate 0.0005 ^
    --alpha 0.35 ^
    --wv db1 ^
    --m 2 ^
    --itr 1 > logs\LongForecasting\AdaCycle\!data_name!_!model_name!_%%p.logs
)

echo All AdaCycle experiments completed!
endlocal
