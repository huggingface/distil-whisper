#!/usr/bin/env bash

TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000 python create_student_model.py \
  --teacher_checkpoint "openai/whisper-small.en" \
  --decoder_layers 2 \
  --save_dir "./"
