#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="0" python ./run_speed_pt.py \
  --dataset_name "distil-whisper/earnings22" \
  --wandb_name "[Earnings] RTX 4090 - large-v2-32-2" \
  --model_name_or_path "patrickvonplaten/whisper-large-v2-32-2" \
  --wandb_project "distil-whisper-speed-benchmark" \
  --dataset_config_name "chunked" \
  --dataset_split_nam "test" \
  --text_column_name "transcription" \
  --batch_size 1
