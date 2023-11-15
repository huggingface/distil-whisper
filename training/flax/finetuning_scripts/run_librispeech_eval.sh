#!/usr/bin/env bash

python run_eval.py \
  --model_name_or_path "./" \
  --dataset_name "distil-whisper/librispeech_asr" \
  --dataset_config_name "all" \
  --test_split_name "validation.clean+validation.other+test.clean+test.other" \
  --text_column_name "text" \
  --cache_dir "/home/sanchitgandhi/cache" \
  --dataset_cache_dir "/home/sanchitgandhi/cache" \
  --output_dir "./" \
  --wandb_name "large-32-2-pl-freeze-librispeech-eval" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-librispeech" \
  --per_device_eval_batch_size 128 \
  --dtype "bfloat16" \
  --do_predict \
  --preprocessing_num_workers 16 \
  --dataloader_num_workers 8 \
  --load_with_scan \
  --predict_with_generate \
  --report_to "wandb"
