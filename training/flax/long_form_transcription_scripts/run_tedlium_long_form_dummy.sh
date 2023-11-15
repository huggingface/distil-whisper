#!/usr/bin/env bash

python run_long_form_transcription.py \
  --model_name_or_path "openai/whisper-tiny" \
  --dataset_name "distil-whisper/tedlium-long-form" \
  --dataset_config_name "all" \
  --dataset_split_name "validation" \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --dataset_cache_dir "/home/sanchitgandhi/.cache" \
  --output_dir "./" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-debug" \
  --wandb_name "whisper-tiny-tedlium-long-form" \
  --per_device_eval_batch_size 64 \
  --max_eval_samples 1 \
  --dtype "bfloat16" \
  --report_to "wandb" \
  --streaming
