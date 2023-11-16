#!/usr/bin/env bash

python run_long_form_transcription.py \
  --model_name_or_path "sanchit-gandhi/large-32-2-ts-freeze-28k-wer-10" \
  --subfolder "checkpoint-15000" \
  --dataset_name "distil-whisper/tedlium-long-form" \
  --dataset_config_name "all" \
  --dataset_split_name "validation" \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --dataset_cache_dir "/home/sanchitgandhi/.cache" \
  --output_dir "./" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-long-form" \
  --wandb_name "large-32-2-ts-freeze-28k-wer-10-30k-steps" \
  --per_device_eval_batch_size 32 \
  --chunk_length_s 20 \
  --dtype "bfloat16" \
  --report_to "wandb" \
  --streaming
