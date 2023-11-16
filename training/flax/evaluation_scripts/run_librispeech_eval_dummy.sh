#!/usr/bin/env bash

python run_eval.py \
  --model_name_or_path "openai/whisper-large-v2" \
  --dataset_name "gigaspeech-l+gigaspeech-l" \
  --dataset_config_name "l+l" \
  --dataset_split_name "train+validation" \
  --text_column_name "text" \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --dataset_cache_dir "/home/sanchitgandhi/.cache" \
  --output_dir "./" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-label" \
  --wandb_name "whisper-large-v2-gigaspeech-l-with-audio" \
  --per_device_eval_batch_size 64 \
  --dtype "bfloat16" \
  --dataloader_num_workers 0 \
  --report_to "wandb" \
  --streaming \
  --max_eval_samples 1024 \
  --predict_with_generate \
  --log_audio
