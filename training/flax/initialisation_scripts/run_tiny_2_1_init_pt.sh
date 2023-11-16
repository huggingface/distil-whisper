#!/usr/bin/env bash

python create_student_model_pt.py \
  --teacher_checkpoint "distil-whisper/tiny-random-whisper" \
  --decoder_layers 1 \
  --save_dir "./"
