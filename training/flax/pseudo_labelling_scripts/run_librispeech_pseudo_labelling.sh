#!/usr/bin/env bash

python run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-large-v2" \
  --dataset_name "sanchit-gandhi/librispeech_asr_clean" \
  --dataset_config_name "clean" \
  --data_split_name "train.100" \
  --text_column_name "text" \
  --cache_dir "/home/sanchitgandhi/cache" \
  --dataset_cache_dir "/home/sanchitgandhi/cache" \
  --output_dir "./transcriptions-streaming" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-debug" \
  --wandb_name "whisper-large-v2-beam-libri-train.clean.100" \
  --per_device_eval_batch_size 16 \
  --max_label_length 256 \
  --dtype "bfloat16" \
  --preprocessing_num_workers 16 \
  --report_to "wandb" \
  --dataloader_num_workers 16 \
  --streaming False \
  --generation_num_beams 1
