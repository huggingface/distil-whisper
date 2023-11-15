#!/usr/bin/env bash

DATASET_NAMES="librispeech_asr+librispeech_asr+common_voice_13_0+voxpopuli+ami-ihm+ami-sdm+peoples_speech-clean+tedlium+switchboard-data+switchboard-data+gigaspeech-l+spgispeech+chime4+google/fleurs+earnings22"
DATASET_CONFIG_NAMES="all+all+en+en+ihm+sdm+clean+release3+all+all+l+L+1-channel+en_us+chunked"
DATASET_SPLIT_NAMES="test.clean+test.other+test+test+test+test+test+test+test.switchboard+test.callhome+test+test+test+test+test"
TEXT_COLUMN_NAMES="text+text+text+text+text+text+text+text+text+text+text+text+text+transcription+transcription"

python run_eval.py \
  --model_name_or_path "openai/whisper-tiny.en" \
  --dataset_name $DATASET_NAMES \
  --dataset_config_name $DATASET_CONFIG_NAMES \
  --dataset_split_name $DATASET_SPLIT_NAMES \
  --text_column_name $TEXT_COLUMN_NAMES \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --dataset_cache_dir "/home/sanchitgandhi/.cache" \
  --output_dir "./" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-test" \
  --wandb_name "tiny.en" \
  --per_device_eval_batch_size 32 \
  --dtype "bfloat16" \
  --dataloader_num_workers 0 \
  --report_to "wandb" \
  --streaming \
  --predict_with_generate

python run_eval.py \
  --model_name_or_path "openai/whisper-base.en" \
  --dataset_name $DATASET_NAMES \
  --dataset_config_name $DATASET_CONFIG_NAMES \
  --dataset_split_name $DATASET_SPLIT_NAMES \
  --text_column_name $TEXT_COLUMN_NAMES \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --dataset_cache_dir "/home/sanchitgandhi/.cache" \
  --output_dir "./" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-test" \
  --wandb_name "base.en" \
  --per_device_eval_batch_size 32 \
  --dtype "bfloat16" \
  --dataloader_num_workers 0 \
  --report_to "wandb" \
  --streaming \
  --predict_with_generate

python run_eval.py \
  --model_name_or_path "openai/whisper-small.en" \
  --dataset_name $DATASET_NAMES \
  --dataset_config_name $DATASET_CONFIG_NAMES \
  --dataset_split_name $DATASET_SPLIT_NAMES \
  --text_column_name $TEXT_COLUMN_NAMES \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --dataset_cache_dir "/home/sanchitgandhi/.cache" \
  --output_dir "./" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-test" \
  --wandb_name "small.en" \
  --per_device_eval_batch_size 32 \
  --dtype "bfloat16" \
  --dataloader_num_workers 0 \
  --report_to "wandb" \
  --streaming \
  --predict_with_generate

python run_eval.py \
  --model_name_or_path "openai/whisper-medium.en" \
  --dataset_name $DATASET_NAMES \
  --dataset_config_name $DATASET_CONFIG_NAMES \
  --dataset_split_name $DATASET_SPLIT_NAMES \
  --text_column_name $TEXT_COLUMN_NAMES \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --dataset_cache_dir "/home/sanchitgandhi/.cache" \
  --output_dir "./" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-test" \
  --wandb_name "medium.en" \
  --per_device_eval_batch_size 32 \
  --dtype "bfloat16" \
  --dataloader_num_workers 0 \
  --report_to "wandb" \
  --streaming \
  --predict_with_generate

python run_eval.py \
  --model_name_or_path "openai/whisper-large-v2" \
  --dataset_name $DATASET_NAMES \
  --dataset_config_name $DATASET_CONFIG_NAMES \
  --dataset_split_name $DATASET_SPLIT_NAMES \
  --text_column_name $TEXT_COLUMN_NAMES \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --dataset_cache_dir "/home/sanchitgandhi/.cache" \
  --output_dir "./" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-test" \
  --wandb_name "large-v2" \
  --per_device_eval_batch_size 16 \
  --dtype "bfloat16" \
  --dataloader_num_workers 0 \
  --report_to "wandb" \
  --streaming \
  --predict_with_generate
