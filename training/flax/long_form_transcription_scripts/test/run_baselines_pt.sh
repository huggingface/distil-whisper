#!/usr/bin/env bash

DATASET_NAMES="distil-whisper/tedlium-long-form+distil-whisper/earnings21+distil-whisper/earnings22+distil-whisper/meanwhile+distil-whisper/rev16"
DATASET_CONFIG_NAMES="all+full+full+default+whisper_subset"
DATASET_SPLIT_NAMES="test+test+test+test+test"
TEXT_COLUMN_NAMES="text+transcription+transcription+text+transcription"

python run_pt_long_form_transcription.py \
  --model_name_or_path "facebook/wav2vec2-large-960h" \
  --dataset_name $DATASET_NAMES \
  --dataset_config_name $DATASET_CONFIG_NAMES \
  --dataset_split_name  $DATASET_SPLIT_NAMES \
  --text_column_name $TEXT_COLUMN_NAMES \
  --output_dir "./" \
  --wandb_project "distil-whisper-long-form-test" \
  --wandb_name "wav2vec2-large-960h" \
  --per_device_eval_batch_size 32 \
  --chunk_length_s 20 \
  --dtype "float16" \
  --report_to "wandb" \
  --streaming
