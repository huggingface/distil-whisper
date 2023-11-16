#!/usr/bin/env bash

python run_eval.py \
  --model_name_or_path "./" \
  --dataset_name "distil-whisper/librispeech_asr" \
  --dataset_config_name "all" \
  --test_split_name "validation.clean[:32]+validation.other[:32]" \
  --text_column_name "text" \
  --cache_dir "/home/sanchitgandhi/cache" \
  --dataset_cache_dir "/home/sanchitgandhi/cache" \
  --output_dir "./" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-debug" \
  --per_device_eval_batch_size 4 \
  --dtype "bfloat16" \
  --do_predict \
  --preprocessing_num_workers 16 \
  --dataloader_num_workers 8 \
  --load_with_scan \
  --predict_with_generate \
  --report_to "wandb"
