#!/usr/bin/env python3
# make sure to use branch: https://github.com/huggingface/transformers/pull/26701
import copy
import time

import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    WhisperForConditionalGeneration,
)


DEVICE = "cuda"
DTYPE = torch.float16
SAMPLING_RATE = 16_000
BATCH_SIZE = 1
USE_FLASH_ATTN_2 = True

# TO DEBUG
GAMMAS = [5, 7, 6, 5, 4, 3, 5]
COUNT = 0

# local loading is faster
teacher = WhisperForConditionalGeneration.from_pretrained(
    "/home/patrick/distil_whisper/",
    torch_dtype=DTYPE,
    variant="fp16",
    low_cpu_mem_usage=True,
    use_flash_attention_2=USE_FLASH_ATTN_2,
)
student = WhisperForConditionalGeneration.from_pretrained(
    "/home/patrick/distil_whisper_student/",
    torch_dtype=DTYPE,
    variant="fp16",
    low_cpu_mem_usage=True,
    use_flash_attention_2=USE_FLASH_ATTN_2,
)
# student = WhisperForCausalLM.from_pretrained("/home/patrick/distil_whisper_student", torch_dtype=DTYPE, variant="fp16", low_cpu_mem_usage=True, use_flash_attention_2=USE_FLASH_ATTN_2)

student.generation_config = copy.deepcopy(teacher.generation_config)
student.generation_config.num_assistant_tokens_schedule = "constant"

# teacher = WhisperForConditionalGeneration.from_pretrained(
#     "openai/whisper-large-v2", torch_dtype=DTYPE, variant="fp16", low_cpu_mem_usage=True
# )
# student = WhisperForConditionalGeneration.from_pretrained(
#     "sanchit-gandhi/large-32-2-gpu-flat-lr", torch_dtype=DTYPE, variant="fp16", low_cpu_mem_usage=True
# )
#
teacher.to(DEVICE)
student.to(DEVICE)

processor = AutoProcessor.from_pretrained("sanchit-gandhi/large-32-2-gpu-flat-lr")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

total_time_default = 0
total_time_spec = 0
total_time_spec_2 = 0

input_values = ds[0]["audio"]["array"]
inputs = processor(input_values, return_tensors="pt", sampling_rate=SAMPLING_RATE)
input_features = inputs.input_features.to(device=DEVICE, dtype=DTYPE)

_ = teacher.generate(input_features, max_length=100)

end_idx = ds.shape[0]
for audio_idx in range(0, end_idx, BATCH_SIZE):
    input_values = ds[audio_idx : audio_idx + BATCH_SIZE]
    input_values = [i["array"] for i in input_values["audio"]]

    inputs = processor(input_values, return_tensors="pt", sampling_rate=SAMPLING_RATE)
    input_features = inputs.input_features.to(device=DEVICE, dtype=DTYPE)

    start_time = time.time()
    out = teacher.generate(input_features, max_length=100)
    run_time = time.time() - start_time
    print(f"Normal Decoding: {run_time}")
    total_time_default += run_time

    default_out = processor.batch_decode(out, skip_special_tokens=True)
    # print("Output", default_out)

    # start_time = time.time()
    # with torch.no_grad():
    #     encoder_outputs = teacher.get_encoder()(input_features).last_hidden_state

    # out, ratio = speculative_decoding(teacher, student, encoder_outputs, max_length=100, gamma=5)
    # run_time = time.time() - start_time
    # print(20 * "=")
    # print(f"Speculative Decoding: {run_time}")
    # total_time_spec += run_time

    # spec_out = processor.batch_decode(out)

    start_time = time.time()
    with torch.no_grad():
        encoder_outputs = teacher.get_encoder()(input_features)

    out = teacher.generate(
        assistant_model=student,
        assistant_encoder_outputs=encoder_outputs,
        encoder_outputs=encoder_outputs,
        max_length=100,
    )
    run_time = time.time() - start_time

    spec_out_2 = processor.batch_decode(out, skip_special_tokens=True)

    print(f"Speculative Decoding 2: {run_time}")
    total_time_spec_2 += run_time

    if spec_out_2 != default_out:
        COUNT += 1
        print(f"Audio {audio_idx} does not match. Spec: {spec_out_2}, True: {default_out}")


print(20 * "=")
print("Total time", total_time_default)
print(f"Overall speed-up spec 2 {total_time_default / total_time_spec_2}")
# print(f"Overall speed-up {total_time_default / total_time_spec}")
