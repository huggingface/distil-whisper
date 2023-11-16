#!/usr/bin/env bash
names=("openai/whisper-large-v2" "openai/whisper-medium.en" "openai/whisper-small.en" "openai/whisper-base.en" "openai/whisper-tiny.en")
names=("openai/whisper-small.en" "openai/whisper-base.en" "openai/whisper-tiny.en")
# names=("patrickvonplaten/whisper-large-v2-32-2" "patrickvonplaten/whisper-medium-24-2")

# chunk_lengths=("15.0" "30.0")
# --return_timestamps \
# --assistant_model_name_or_path "patrickvonplaten/whisper-large-v2-32-2" \
# --attn_type "flash2"  \

# Double loop
for name in "${names[@]}"; do
  CUDA_VISIBLE_DEVICES="1" python ./run_speed_pt.py \
    --dataset_name "distil-whisper/earnings21+distil-whisper/earnings22+distil-whisper/meanwhile+distil-whisper/rev16" \
    --wandb_name "A100-${name}-Longform-Orig" \
    --model_name_or_path ${name} \
    --wandb_project "distil-whisper-speed-bench-long-form-orig-32" \
    --dataset_config_name "full+full+default+whisper_subset" \
    --dataset_split_name "test+test+test+test" \
    --text_column_name "transcription+transcription+text+transcription" \
    --use_orig_whisper \
    --max_label_length "1000000" \
    --samples_per_dataset "32" \
    --batch_size "1"
done
