#!/usr/bin/env bash

DATASET_NAME="librispeech_asr-noise+librispeech_asr-noise+librispeech_asr-noise+librispeech_asr-noise+librispeech_asr-noise+librispeech_asr-noise+librispeech_asr-noise+librispeech_asr-noise+librispeech_asr-noise+librispeech_asr-noise+librispeech_asr-noise"
DATASET_CONFIG_NAME=("test-white-noise" "test-pub-noise")
DATASET_SPLIT_NAME="40+35+30+25+20+15+10+5+0+minus5+minus10"

for i in "${!DATASET_CONFIG_NAME[@]}"; do
  python run_eval.py \
    --model_name_or_path "sanchit-gandhi/large-32-2-tpu-timestamped-resumed" \
    --dataset_name $DATASET_NAME \
    --dataset_config_name "${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}" \
    --dataset_split_name $DATASET_SPLIT_NAME \
    --cache_dir "/home/sanchitgandhi/cache" \
    --dataset_cache_dir "/home/sanchitgandhi/cache" \
    --output_dir "./" \
    --wandb_dir "/home/sanchitgandhi/cache" \
    --wandb_project "distil-whisper-noise-test" \
    --wandb_name "large-32-2-tpu-timestamped-${DATASET_CONFIG_NAME[i]}" \
    --per_device_eval_batch_size 64 \
    --dtype "bfloat16" \
    --dataloader_num_workers 16 \
    --report_to "wandb" \
    --streaming \
    --predict_with_generate
    
    python run_eval.py \
    --model_name_or_path "sanchit-gandhi/medium-24-2-tpu-timestamped-prob-0.2" \
    --subfolder "checkpoint-45000" \
    --dataset_name $DATASET_NAME \
    --dataset_config_name "${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}+${DATASET_CONFIG_NAME[i]}" \
    --dataset_split_name $DATASET_SPLIT_NAME \
    --cache_dir "/home/sanchitgandhi/cache" \
    --dataset_cache_dir "/home/sanchitgandhi/cache" \
    --output_dir "./" \
    --wandb_dir "/home/sanchitgandhi/cache" \
    --wandb_project "distil-whisper-noise-test" \
    --wandb_name "medium-24-2-tpu-timestamped-prob-0.2-${DATASET_CONFIG_NAME[i]}" \
    --per_device_eval_batch_size 64 \
    --dtype "bfloat16" \
    --dataloader_num_workers 16 \
    --report_to "wandb" \
    --streaming \
    --predict_with_generate
done
