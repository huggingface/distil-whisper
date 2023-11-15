#!/usr/bin/env bash

python run_finetuning.py \
  --model_name_or_path "distil-whisper/tiny-random-whisper" \
  --dataset_name "distil-whisper/librispeech_asr" \
  --dataset_config_name "all" \
  --train_split_name "train.clean.100[:1024]" \
  --eval_split_name "validation.clean[:1024]" \
  --cache_dir "/home/sanchitgandhi/cache" \
  --dataset_cache_dir "/home/sanchitgandhi/cache" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --text_column_name "text" \
  --output_dir "./" \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 4 \
  --dtype "bfloat16" \
  --num_train_epochs 2 \
  --dataloader_num_workers 16 \
  --freeze_encoder \
  --wandb_project "distil-whisper-debug" \
  --logging_steps 2 \
  --use_scan \
  --gradient_checkpointing \
  --overwrite_output_dir \
  --predict_with_generate
