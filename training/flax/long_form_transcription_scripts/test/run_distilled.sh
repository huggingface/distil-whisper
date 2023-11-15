#!/usr/bin/env bash

DATASET_NAMES="distil-whisper/tedlium-long-form+distil-whisper/earnings21+distil-whisper/earnings22+distil-whisper/meanwhile+distil-whisper/rev16"
DATASET_CONFIG_NAMES="all+full+full+default+whisper_subset"
DATASET_SPLIT_NAMES="test+test+test+test+test"
TEXT_COLUMN_NAMES="text+transcription+transcription+text+transcription"

python run_long_form_transcription.py \
  --model_name_or_path "sanchit-gandhi/large-32-2-tpu-timestamped-resumed" \
  --dataset_name $DATASET_NAMES \
  --dataset_config_name $DATASET_CONFIG_NAMES \
  --dataset_split_name  $DATASET_SPLIT_NAMES \
  --text_column_name $TEXT_COLUMN_NAMES \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --dataset_cache_dir "/home/sanchitgandhi/.cache" \
  --output_dir "./" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-long-form-test" \
  --wandb_name "large-32-2" \
  --per_device_eval_batch_size 16 \
  --chunk_length_s 15 \
  --generation_max_length 128 \
  --dtype "bfloat16" \
  --report_to "wandb" \
  --streaming

python run_long_form_transcription.py \
  --model_name_or_path "sanchit-gandhi/medium-24-2-tpu-timestamped-prob-0.2" \
  --subfolder "checkpoint-45000" \
  --dataset_name $DATASET_NAMES \
  --dataset_config_name $DATASET_CONFIG_NAMES \
  --dataset_split_name  $DATASET_SPLIT_NAMES \
  --text_column_name $TEXT_COLUMN_NAMES \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --dataset_cache_dir "/home/sanchitgandhi/.cache" \
  --output_dir "./" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-long-form-test" \
  --wandb_name "medium-24-2" \
  --per_device_eval_batch_size 16 \
  --chunk_length_s 20 \
  --generation_max_length 128 \
  --dtype "bfloat16" \
  --report_to "wandb" \
  --streaming
