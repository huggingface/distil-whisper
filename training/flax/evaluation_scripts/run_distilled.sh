#!/usr/bin/env bash

python run_eval.py \
  --model_name_or_path "sanchit-gandhi/large-32-2-ts-freeze-28k-wer-10" \
  --subfolder "checkpoint-15000" \
  --dataset_name "librispeech_asr+librispeech_asr+common_voice_13_0+voxpopuli+ami-ihm+ami-sdm+peoples_speech-clean+tedlium+switchboard-data+gigaspeech-l+spgispeech+chime4+google/fleurs+sanchit-gandhi/earnings22_split_resampled" \
  --dataset_config_name "all+all+en+en+ihm+sdm+clean+release3+all+l+L+1-channel+en_us+default" \
  --dataset_split_name "validation.clean+validation.other+validation+validation+validation+validation+validation+validation+validation+validation+validation+validation+validation+validation" \
  --text_column_name "text+text+text+text+text+text+text+text+text+text+text+text+transcription+sentence" \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --dataset_cache_dir "/home/sanchitgandhi/.cache" \
  --output_dir "./" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-eval" \
  --wandb_name "large-32-2-ts-freeze-28k-wer-10-30k-steps" \
  --per_device_eval_batch_size 64 \
  --dtype "bfloat16" \
  --dataloader_num_workers 0 \
  --report_to "wandb" \
  --streaming \
  --predict_with_generate
