#!/usr/bin/env bash
# batch_sizes=(1 4)
batch_sizes=(1)
names=("openai/whisper-large-v2" "openai/whisper-large-v2" "openai/whisper-medium.en" "openai/whisper-medium.en")
assistant_names=("patrickvonplaten/whisper-large-v2-32-2" "openai/whisper-small" "patrickvonplaten/whisper-medium-24-2" "openai/whisper-base.en")

# --assistant_model_name_or_path "patrickvonplaten/whisper-large-v2-32-2" \
# --use_pipeline \

# Double loop

for (( i=0; i<${#names[*]}; ++i)); do
  name=${names[$i]}
  assistant_name=${assistant_names[$i]}

  for batch_size in "${batch_sizes[@]}"; do
      CUDA_VISIBLE_DEVICES="0" python ./run_speed_pt.py \
        --dataset_name "distil-whisper/chime4+distil-whisper/earnings22+google/fleurs+kensho/spgispeech" \
        --wandb_name "FP16-RTX-4090-bsz${batch_size}-${name}-${assistant_name}" \
        --model_name_or_path ${name} \
        --wandb_project "distil-whisper-speed-bench-check-spec-dec-final" \
        --dataset_config_name "1-channel+chunked+en_us+test" \
        --dataset_split_name "test+test+test+test" \
        --text_column_name "text+transcription+transcription+transcript" \
        --attn_type "flash2" \
        --assistant_model_name_or_path ${assistant_name} \
        --samples_per_dataset "10" \
        --batch_size ${batch_size}
    done
done
