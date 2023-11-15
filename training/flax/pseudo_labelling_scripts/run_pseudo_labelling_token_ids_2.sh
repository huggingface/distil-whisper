#!/usr/bin/env bash

MODEL_NAME="openai/whisper-large-v3"
CACHE_DIR="/home/sanchitgandhi/.cache"
OUTPUT_DIR="./transcriptions-streaming"
WANDB_DIR="/home/sanchitgandhi/.cache"
WANDB_PROJECT="distil-whisper-label"
SPLITS="train+validation+test"
BATCH_SIZE=16
NUM_BEAMS=1
MAX_LABEL_LENGTH=256
LOGGING_STEPS=500
NUM_WORKERS=64
RETURN_TIMESTAMPS=False
DECODE_TOKEN_IDS=False

DATASET_NAMES=("distil-whisper/common_voice_13_0" "distil-whisper/voxpopuli" "distil-whisper/tedlium" "distil-whisper/ami-ihm" "distil-whisper/ami-sdm" "distil-whisper/spgispeech" "distil-whisper/gigaspeech-l")
CONFIGS=("en" "en" "release3" "ihm" "sdm" "L" "l")

for i in "${!DATASET_NAMES[@]}"; do
  python run_pseudo_labelling.py \
  --model_name_or_path $MODEL_NAME \
  --dataset_name "${DATASET_NAMES[i]}" \
  --dataset_config_name "${CONFIGS[i]}" \
  --data_split_name "$SPLITS" \
  --wandb_name "whisper-large-v2-${DATASET_NAMES[i]}-token-ids" \
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
  --compilation_cache $CACHE_DIR \
  --decode_token_ids $DECODE_TOKEN_IDS
done
