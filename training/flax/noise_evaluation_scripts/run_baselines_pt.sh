#!/usr/bin/env bash

MODEL_IDs=("facebook/wav2vec2-base-960h" "facebook/wav2vec2-large-960h" "facebook/wav2vec2-large-960h-lv60-self" "facebook/wav2vec2-large-robust-ft-libri-960h" "facebook/wav2vec2-conformer-rel-pos-large-960h-ft" "facebook/wav2vec2-conformer-rope-large-960h-ft" "facebook/hubert-large-ls960-ft" "facebook/hubert-xlarge-ls960-ft" "facebook/mms-1b-all" "facebook/mms-1b-fl102" "facebook/data2vec-audio-large-960h" "facebook/data2vec-audio-base-960h")
DATASET_NAME="librispeech_asr-noise+librispeech_asr-noise+librispeech_asr-noise+librispeech_asr-noise+librispeech_asr-noise+librispeech_asr-noise+librispeech_asr-noise+librispeech_asr-noise+librispeech_asr-noise+librispeech_asr-noise+librispeech_asr-noise"
DATASET_CONFIG_NAME=("test-white-noise" "test-pub-noise")
DATASET_SPLIT_NAME="40+35+30+25+20+15+10+5+0+minus5+minus10"

for i in "${!MODEL_IDs[@]}"; do
  for j in "${!DATASET_CONFIG_NAME[@]}"; do
    python run_pt_long_form_transcription.py \
        --model_name_or_path "${MODEL_IDs[i]}" \
        --dataset_name $DATASET_NAME \
        --dataset_config_name "${DATASET_CONFIG_NAME[j]}+${DATASET_CONFIG_NAME[j]}+${DATASET_CONFIG_NAME[j]}+${DATASET_CONFIG_NAME[j]}+${DATASET_CONFIG_NAME[j]}+${DATASET_CONFIG_NAME[j]}+${DATASET_CONFIG_NAME[j]}+${DATASET_CONFIG_NAME[j]}+${DATASET_CONFIG_NAME[j]}+${DATASET_CONFIG_NAME[j]}+${DATASET_CONFIG_NAME[j]}" \
        --dataset_split_name $DATASET_SPLIT_NAME \
        --cache_dir "/home/sanchit/.cache" \
        --dataset_cache_dir "/home/sanchit/.cache" \
        --output_dir "./" \
        --wandb_dir "/home/sanchit/.cache" \
        --wandb_project "distil-whisper-noise-test" \
        --wandb_name "${MODEL_IDs[i]}-${DATASET_CONFIG_NAME[j]}" \
        --per_device_eval_batch_size 16 \
        --dtype "float16" \
        --report_to "wandb" \
        --streaming \
        --predict_with_generate
  done
done
