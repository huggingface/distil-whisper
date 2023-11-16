#!/usr/bin/env bash

DATASET_NAMES="librispeech_asr+librispeech_asr+common_voice_13_0+voxpopuli+ami-ihm+ami-sdm+peoples_speech-clean+tedlium+switchboard-data+gigaspeech-l+spgispeech+chime4+google/fleurs+sanchit-gandhi/earnings22_split_resampled"
DATASET_CONFIG_NAMES="all+all+en+en+ihm+sdm+clean+release3+all+l+L+1-channel+en_us+default"
DATASET_SPLIT_NAMES="validation.clean+validation.other+validation+validation+validation+validation+validation+validation+validation+validation+validation+validation+validation+validation"
TEXT_COLUMN_NAMES="text+text+text+text+text+text+text+text+text+text+text+text+transcription+sentence"

python run_long_form_transcription.py \
  --model_name_or_path "sanchit-gandhi/large-32-2-ts-28k-wer-10-converted-context-20s" \
  --dataset_name $DATASET_NAMES \
  --dataset_config_name $DATASET_CONFIG_NAMES \
  --dataset_split_name  $DATASET_SPLIT_NAMES \
  --text_column_name $TEXT_COLUMN_NAMES \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --dataset_cache_dir "/home/sanchitgandhi/.cache" \
  --output_dir "./" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-eval" \
  --wandb_name "large-32-2-ts-freeze-28k-wer-10-30k-steps-chunk-length-15-context-20" \
  --per_device_eval_batch_size 1 \
  --chunk_length_s 15 \
  --dtype "bfloat16" \
  --report_to "wandb" \
  --streaming
