#!/usr/bin/env bash

TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000 python convert_train_state_to_hf.py \
  --model_name_or_path "distil-whisper/large-32-2" \
  --output_dir "./" \
  --resume_from_checkpoint "checkpoint-15000" \
  --cache_dir "/home/sanchitgandhi/.cache" \
  --use_scan
