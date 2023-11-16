#!/usr/bin/env bash

TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000 python create_student_model.py \
  --teacher_checkpoint "distil-whisper/tiny-random-whisper" \
  --decoder_layers 1 \
  --save_dir "./"
