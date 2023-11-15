#!/usr/bin/env bash

DATASET_NAMES="distil-whisper/tedlium-long-form+distil-whisper/earnings21+distil-whisper/earnings22+distil-whisper/meanwhile+distil-whisper/rev16"
DATASET_CONFIG_NAMES="all+full+full+default+whisper_subset"
DATASET_SPLIT_NAMES="test+test+test+test+test"
TEXT_COLUMN_NAMES="text+transcription+transcription+text+transcription"

python run_long_form_transcription.py \
  --model_name_or_path "openai/whisper-tiny.en" \
  --dataset_name $DATASET_NAMES \
  --dataset_config_name $DATASET_CONFIG_NAMES \
  --dataset_split_name  $DATASET_SPLIT_NAMES \
  --text_column_name $TEXT_COLUMN_NAMES \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --dataset_cache_dir "/home/sanchitgandhi/.cache" \
  --output_dir "./" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-long-form-test" \
  --wandb_name "tiny.en" \
  --per_device_eval_batch_size 16 \
  --chunk_length_s 30 \
  --generation_max_length 128 \
  --dtype "bfloat16" \
  --report_to "wandb" \
  --streaming \
  --return_timestamps

python run_long_form_transcription.py \
  --model_name_or_path "openai/whisper-base.en" \
  --dataset_name $DATASET_NAMES \
  --dataset_config_name $DATASET_CONFIG_NAMES \
  --dataset_split_name  $DATASET_SPLIT_NAMES \
  --text_column_name $TEXT_COLUMN_NAMES \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --dataset_cache_dir "/home/sanchitgandhi/.cache" \
  --output_dir "./" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-long-form-test" \
  --wandb_name "base.en" \
  --per_device_eval_batch_size 16 \
  --chunk_length_s 30 \
  --generation_max_length 128 \
  --dtype "bfloat16" \
  --report_to "wandb" \
  --streaming \
  --return_timestamps

python run_long_form_transcription.py \
  --model_name_or_path "openai/whisper-small.en" \
  --dataset_name $DATASET_NAMES \
  --dataset_config_name $DATASET_CONFIG_NAMES \
  --dataset_split_name  $DATASET_SPLIT_NAMES \
  --text_column_name $TEXT_COLUMN_NAMES \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --dataset_cache_dir "/home/sanchitgandhi/.cache" \
  --output_dir "./" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-long-form-test" \
  --wandb_name "small.en" \
  --per_device_eval_batch_size 16 \
  --chunk_length_s 30 \
  --generation_max_length 128 \
  --dtype "bfloat16" \
  --report_to "wandb" \
  --streaming \
  --return_timestamps

python run_long_form_transcription.py \
  --model_name_or_path "openai/whisper-medium.en" \
  --dataset_name $DATASET_NAMES \
  --dataset_config_name $DATASET_CONFIG_NAMES \
  --dataset_split_name  $DATASET_SPLIT_NAMES \
  --text_column_name $TEXT_COLUMN_NAMES \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --dataset_cache_dir "/home/sanchitgandhi/.cache" \
  --output_dir "./" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-long-form-test" \
  --wandb_name "medium.en" \
  --per_device_eval_batch_size 16 \
  --chunk_length_s 30 \
  --generation_max_length 128 \
  --dtype "bfloat16" \
  --report_to "wandb" \
  --streaming \
  --return_timestamps

python run_long_form_transcription.py \
  --model_name_or_path "openai/whisper-large-v2" \
  --dataset_name $DATASET_NAMES \
  --dataset_config_name $DATASET_CONFIG_NAMES \
  --dataset_split_name  $DATASET_SPLIT_NAMES \
  --text_column_name $TEXT_COLUMN_NAMES \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --dataset_cache_dir "/home/sanchitgandhi/.cache" \
  --output_dir "./" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-long-form-test" \
  --wandb_name "large-v2" \
  --per_device_eval_batch_size 16 \
  --chunk_length_s 30 \
  --generation_max_length 128 \
  --dtype "bfloat16" \
  --report_to "wandb" \
  --streaming \
  --return_timestamps
