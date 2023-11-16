#!/usr/bin/env bash

python run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-tiny" \
  --dataset_name "distil-whisper/librispeech_asr" \
  --dataset_config_name "all" \
  --data_split_name "validation.clean+validation.other" \
  --text_column_name "text" \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --dataset_cache_dir "/home/sanchitgandhi/.cache" \
  --output_dir "./transcriptions-streaming" \
  --wandb_dir "/home/sanchitgandhi/.cache" \
  --wandb_project "distil-whisper-debug" \
  --per_device_eval_batch_size 1 \
  --dtype "bfloat16" \
  --dataloader_num_workers 16 \
  --logging_steps 2 \
  --report_to "wandb" \
  --streaming \
  --max_samples_per_split 256 \
  --max_label_length 256 \
  --return_timestamps \
  --decode_token_ids False
