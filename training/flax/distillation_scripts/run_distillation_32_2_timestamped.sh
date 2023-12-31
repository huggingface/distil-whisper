#!/usr/bin/env bash

TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000 python3 run_distillation.py \
  --model_name_or_path "distil-whisper/large-32-2" \
  --teacher_model_name_or_path "openai/whisper-large-v2" \
  --train_dataset_config_name "all+all+all+en+en+ihm+sdm+clean+release3+all+l+all+all+all+release3" \
  --train_dataset_samples "2.9+10.4+14.9+89+18.2+10.9+10.9+288+26.8+371.2+226.6+2.9+10.4+14.9+26.8" \
  --train_dataset_name "librispeech_asr-timestamped+librispeech_asr-timestamped+librispeech_asr-timestamped+common_voice_13_0-timestamped+voxpopuli-timestamped+ami-ihm-timestamped+ami-sdm-timestamped+peoples_speech-clean-timestamped+tedlium-timestamped+switchboard-data+gigaspeech-l-timestamped+librispeech_asr-prompted+librispeech_asr-prompted+librispeech_asr-prompted+tedlium-prompted" \
  --train_split_name "train.clean.100+train.clean.360+train.other.500+train+train+train+train+train+train+train+train+train.clean.100+train.clean.360+train.other.500+train" \
  --eval_dataset_name "distil-whisper/gigaspeech-l+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset" \
  --eval_dataset_config_name "l+librispeech+librispeech+common_voice+common_voice+voxpopuli+voxpopuli+tedlium+tedlium+spgispeech+spgispeech+ami+ami" \
  --eval_split_name "validation+clean+other+clean+other+clean+other+clean+other+clean+other+clean+other" \
  --eval_text_column_name "text+ortho_transcript+ortho_transcript+ortho_transcript+ortho_transcript+ortho_transcript+ortho_transcript+ortho_transcript+ortho_transcript+ortho_transcript+ortho_transcript+ortho_transcript+ortho_transcript" \
  --eval_steps 5000 \
  --save_steps 5000 \
  --warmup_steps 500 \
  --learning_rate 0.0001 \
  --logging_steps 25 \
  --save_total_limit 1 \
  --max_steps 80000 \
  --wer_threshold 10 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --dtype "bfloat16" \
  --dataloader_num_workers 16 \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --dataset_cache_dir "/home/sanchitgandhi/.cache" \
  --output_dir "./" \
  --wandb_name "large-32-2-tpu-timestamped" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper" \
  --do_train \
  --do_eval \
  --use_scan \
  --gradient_checkpointing \
  --overwrite_output_dir \
  --predict_with_generate \
  --freeze_encoder \
  --streaming \
  --use_auth_token \
  --push_to_hub
