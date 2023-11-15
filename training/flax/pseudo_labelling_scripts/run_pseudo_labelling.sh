#!/usr/bin/env bash

MODEL_NAME="openai/whisper-large-v3"
CACHE_DIR="/home/sanchitgandhi/.cache"
OUTPUT_DIR="./transcriptions-streaming"
WANDB_DIR="/home/sanchitgandhi/.cache"
WANDB_PROJECT="distil-whisper-label"
BATCH_SIZE=64
NUM_BEAMS=1
MAX_LABEL_LENGTH=256
LOGGING_STEPS=500
NUM_WORKERS=64
RETURN_TIMESTAMPS=False

python run_pseudo_labelling.py \
  --model_name_or_path $MODEL_NAME \
  --dataset_name "distil-whisper/librispeech_asr" \
  --dataset_config_name "all" \
  --data_split_name "train.other.500+validation.clean+validation.other+test.clean+test.other" \
  --wandb_name "whisper-large-v2-librispeech_asr" \
  --cache_dir $CACHE_DIR \
  --dataset_cache_dir $CACHE_DIR \
  --output_dir $OUTPUT_DIR \
  --wandb_dir $WANDB_DIR \
  --wandb_project $WANDB_PROJECT \
  --per_device_eval_batch_size $BATCH_SIZE \
  --generation_num_beams $NUM_BEAMS \
  --max_label_length $MAX_LABEL_LENGTH \
  --logging_steps $LOGGING_STEPS \
  --dataloader_num_workers $NUM_WORKERS \
  --dtype "bfloat16" \
  --report_to "wandb" \
  --streaming True \
  --push_to_hub \
  --return_timestamps $RETURN_TIMESTAMPS \
  --compilation_cache $CACHE_DIR

python run_pseudo_labelling.py \
  --model_name_or_path $MODEL_NAME \
  --dataset_name "distil-whisper/peoples_speech-clean" \
  --dataset_config_name "clean" \
  --data_split_name "train+validation+test" \
  --wandb_name "whisper-large-v2-peoples_speech-clean" \
  --cache_dir $CACHE_DIR \
  --dataset_cache_dir $CACHE_DIR \
  --output_dir $OUTPUT_DIR \
  --wandb_dir $WANDB_DIR \
  --wandb_project $WANDB_PROJECT \
  --per_device_eval_batch_size $BATCH_SIZE \
  --generation_num_beams $NUM_BEAMS \
  --max_label_length $MAX_LABEL_LENGTH \
  --logging_steps $LOGGING_STEPS \
  --dataloader_num_workers $NUM_WORKERS \
  --dtype "bfloat16" \
  --report_to "wandb" \
  --streaming True \
  --push_to_hub \
  --return_timestamps $RETURN_TIMESTAMPS \
  --compilation_cache $CACHE_DIR
