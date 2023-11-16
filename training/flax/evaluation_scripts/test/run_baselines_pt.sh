#!/usr/bin/env bash

DATASET_NAMES="librispeech_asr+librispeech_asr+common_voice_13_0+voxpopuli+ami-ihm+ami-sdm+peoples_speech-clean+tedlium+switchboard-data+switchboard-data+gigaspeech-l+spgispeech+chime4+google/fleurs+earnings22"
DATASET_CONFIG_NAMES="all+all+en+en+ihm+sdm+clean+release3+all+all+l+L+1-channel+en_us+chunked"
DATASET_SPLIT_NAMES="test.clean+test.other+test+test+test+test+test+test+test.switchboard+test.callhome+test+test+test+test+test"
TEXT_COLUMN_NAMES="text+text+text+text+text+text+text+text+text+text+text+text+text+transcription+transcription"

python run_pt_long_form_transcription.py \
  --model_name_or_path "facebook/wav2vec2-large-960h" \
  --wandb_name "facebook/wav2vec2-large-960h" \
  --dataset_name $DATASET_NAMES \
  --dataset_config_name $DATASET_CONFIG_NAMES \
  --dataset_split_name $DATASET_SPLIT_NAMES \
  --text_column_name $TEXT_COLUMN_NAMES \
  --output_dir "./" \
  --wandb_project "distil-whisper-test" \
  --per_device_eval_batch_size 32 \
  --dtype "float16" \
  --dataloader_num_workers 0 \
  --report_to "wandb" \
  --streaming \
  --predict_with_generate
