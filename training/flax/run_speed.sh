#!/usr/bin/env bash
# --wandb_project "distil-whisper-speed-bench-1024-no-timestamps" \
batch_sizes=(1 16)
names=("openai/whisper-large-v2" "openai/whisper-medium.en" "openai/whisper-small.en" "openai/whisper-base.en" "openai/whisper-tiny.en" "patrickvonplaten/whisper-large-v2-32-2" "patrickvonplaten/whisper-medium-24-2")

# Double loop
for name in "${names[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
      CUDA_VISIBLE_DEVICES="1" python ./run_speed_pt.py \
        --dataset_name "google/fleurs+distil-whisper/chime4+distil-whisper/earnings22+kensho/spgispeech" \
        --wandb_name "T4-bsz${batch_size}-${name}" \
        --model_name_or_path ${name} \
        --wandb_project "beam-search-distil-whisper-speed-bench-256-no-timestamps" \
        --dataset_config_name "en_us+1-channel+chunked+test" \
        --dataset_split_name "test+test+test+test" \
        --text_column_name "transcription+text+transcription+transcript" \
        --samples_per_dataset "256" \
	--attn_type "flash2" \
	--num_beams 5 \
        --batch_size ${batch_size}
    done
done
