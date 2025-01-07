# Distil-Whisper

[[Paper]](https://arxiv.org/abs/2311.00430)
[[Models]](https://huggingface.co/collections/distil-whisper/distil-whisper-models-65411987e6727569748d2eb6)
[[Colab]](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/Distil_Whisper_Benchmark.ipynb)
[[Training Code]](training)

Distil-Whisper is a distilled version of Whisper for English speech recognition that is **6 times faster**, 49% smaller, and performs **within 1% word 
error rate (WER)** on out-of-distribution evaluation sets:

| Model                                                                      | Params / M | Rel. Latency ↑ | Short-Form WER ↓ | Long-Form WER ↓ |
|----------------------------------------------------------------------------|------------|----------------|------------------|-----------------|
| [large-v3](https://huggingface.co/openai/whisper-large-v3)                 | 1550       | 1.0            | **8.4**          | 11.0            |
|                                                                            |            |                |                  |                 |
| [distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3)   | 756        | 6.3            | 9.7              | **10.8**        |
| [distil-large-v2](https://huggingface.co/distil-whisper/distil-large-v2)   | 756        | 5.8            | 10.1             | 11.6            |
| [distil-medium.en](https://huggingface.co/distil-whisper/distil-medium.en) | 394        | **6.8**        | 11.1             | 12.4            |
| [distil-small.en](https://huggingface.co/distil-whisper/distil-small.en)   | **166**    | 5.6            | 12.1             | 12.8            |

For most applications, we recommend the latest [distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) checkpoint,
since it is the most performant distilled checkpoint and compatible across all Whisper libraries. The only exception is 
resource-constrained applications with very little memory, such as on-device or mobile applications, where the 
[distil-small.en](https://huggingface.co/distil-whisper/distil-small.en) is a great choice, since it is only 166M 
parameters and performs within 4% WER of Whisper large-v3.

> [!NOTE]  
> Distil-Whisper is only available for English speech recognition. For multilingual speech recognition, we recommend using the [Whisper Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) checkpoint, which was released by OpenAI and leverages the same principles as Distil-Whisper. For details, refer to the Whisper turbo [release statement](https://github.com/openai/whisper/discussions/2363).

## 1. Usage

Distil-Whisper is supported in Hugging Face 🤗 Transformers from version 4.35 onwards. To run the model, first 
install the latest version of the Transformers library. For this example, we'll also install 🤗 Datasets to load a toy 
audio dataset from the Hugging Face Hub:

```bash
pip install --upgrade pip
pip install --upgrade transformers accelerate datasets[audio]
```

### Short-Form Transcription

Short-form transcription is the process of transcribing audio samples that are less than 30-seconds long, which is the 
maximum receptive field of the Whisper models. This means the entire audio clip can be processed in one go without the 
need for chunking.

First, we load Distil-Whisper via the convenient [`AutoModelForSpeechSeq2Seq`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSpeechSeq2Seq) and [`AutoProcessor`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoProcessor) classes.

We load the model in `float16` precision and make sure that loading time takes as little time as possible by passing `low_cpu_mem_usage=True`.
In addition, we want to make sure that the model is loaded in [`safetensors`](https://github.com/huggingface/safetensors) format by passing `use_safetensors=True`:

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
```

The model and processor can then be passed to the [`pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline).
Note that if you would like to have more control over the generation process, you can directly make use of model + processor API as shown below.

```python
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)
```

Next, we load an example short-form audio from the LibriSpeech corpus:

```python
from datasets import load_dataset

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = dataset[0]["audio"]
```

Finally, we can call the pipeline to transcribe the audio:

```python
result = pipe(sample)
print(result["text"])
```

To transcribe a local audio file, simply pass the path to your audio file when you call the pipeline:

```python
result = pipe("audio.mp3")
print(result["text"])
```

For more information on how to customize the automatic speech recognition pipeline, please refer to the ASR pipeline [docs](https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline).
We also provide an end-to-end [Google Colab](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/Distil_Whisper_Benchmark.ipynb) that benchmarks Whisper against Distil-Whisper.

<details>
<summary> For more control over the generation parameters, use the model + processor API directly: </summary>

Ad-hoc generation arguments can be passed to `model.generate`, including `num_beams` for beam-search, `return_timestamps` 
for segment-level timestamps, and `prompt_ids` for prompting. See the [docstrings](https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate)
for more details.

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import Audio, load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(processor.feature_extractor.sampling_rate))
sample = dataset[0]["audio"]

input_features = processor(
  sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
).input_features

input_features = input_features.to(device, dtype=torch_dtype)

gen_kwargs = {
  "max_new_tokens": 128,
  "num_beams": 1,
  "return_timestamps": False,
}

pred_ids = model.generate(input_features, **gen_kwargs)
pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=gen_kwargs["return_timestamps"])

print(pred_text)
```

</details>

### Sequential Long-Form

The latest [distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) checkpoint is specifically designed 
to be compatible with OpenAI's sequential long-form transcription algorithm. This algorithm uses a sliding window for 
buffered inference of long audio files (> 30-seconds), and returns more accurate transcriptions compared to the 
[chunked long-form algorithm](#chunked-long-form).

The sequential long-form algorithm should be used in either of the following scenarios:
1. Transcription accuracy is the most important factor, and latency is less of a consideration
2. You are transcribing **batches** of long audio files, in which case the latency of sequential is comparable to chunked, while being up to 0.5% WER more accurate

If you are transcribing single long audio files and latency is the most important factor, you should use the chunked algorithm
described [below](#chunked-long-form). For a detailed explanation of the different algorithms, refer to Sections 5 of 
the [Distil-Whisper paper](https://arxiv.org/pdf/2311.00430.pdf).

We start by loading the model and processor as before:

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
```

The model and processor can then be passed to the [`pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline).
Note that if you would like to have more control over the generation process, you can directly make use of `model.generate(...)` API as shown below.

```python
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)
```

Next, we load a long-form audio sample. Here, we use an example of concatenated samples from the LibriSpeech corpus:

```python
from datasets import load_dataset

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]
```

Finally, we can call the pipeline to transcribe the audio:

```python
result = pipe(sample)
print(result["text"])
```

To transcribe a local audio file, simply pass the path to your audio file when you call the pipeline:

```python
result = pipe("audio.mp3")
print(result["text"])
```

<details>

<summary> For more control over the generation parameters, use the model + processor API directly: </summary>

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import Audio, load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(processor.feature_extractor.sampling_rate))
sample = dataset[0]["audio"]

inputs = processor(
    sample["array"],
    sampling_rate=sample["sampling_rate"],
    return_tensors="pt",
    truncation=False,
    padding="longest",
    return_attention_mask=True,
)
inputs = inputs.to(device, dtype=torch_dtype)

gen_kwargs = {
    "max_new_tokens": 448,
    "num_beams": 1,
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "return_timestamps": True,
}

pred_ids = model.generate(**inputs, **gen_kwargs)
pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)

print(pred_text)
```

</details>

### Chunked Long-Form

distil-large-v3 remains compatible with the Transformers chunked long-form algorithm. This algorithm should be used when 
a single large audio file is being transcribed and the fastest possible inference is required. In such circumstances, 
the chunked algorithm is up to 9x faster than OpenAI's sequential long-form implementation (see Table 7 of the 
[Distil-Whisper paper](https://arxiv.org/pdf/2311.00430.pdf)).

We can load the model and processor as before:

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
```

To enable chunking, pass the `chunk_length_s` parameter to the `pipeline`. For distil-large-v3, a chunk length of 25-seconds
is optimal. To activate batching, pass the argument `batch_size`:

```python
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=25,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
)
```

The argument `max_new_tokens` controls the maximum number of generated tokens *per-chunk*. In the typical speech setting,
we have no more than 3 words spoken per-second. Therefore, for a 30-second input, we have at most 90 words (approx 128 tokens).
We set the maximum number of generated tokens per-chunk to 128 to truncate any possible hallucinations that occur at the 
end of the segment. These tokens get removed at the chunk borders using the long-form chunking transcription algorithm, 
so it is more efficient to truncate them directly during generation to avoid redundant generation steps in the decoder.

Now, let's load a long-form audio sample. Here, we use an example of concatenated samples from the LibriSpeech corpus:

```python
from datasets import load_dataset

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]
```

Finally, we can call the pipeline to transcribe the audio:

```python
result = pipe(sample)
print(result["text"])
```

For more information on how to customize the automatic speech recognition pipeline, please refer to the ASR pipeline [docs](https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline).

### Speculative Decoding

Distil-Whisper can be used as an assistant model to Whisper for [speculative decoding](https://huggingface.co/blog/whisper-speculative-decoding). 
Speculative decoding mathematically ensures the exact same outputs as Whisper are obtained while being 2 times faster. 
This makes it the perfect drop-in replacement for existing Whisper pipelines, since the same outputs are guaranteed.

For speculative decoding, we need to load both the teacher: [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3).
As well as the assistant (*a.k.a* student) [`distil-whisper/distil-large-v3`](https://huggingface.co/distil-whisper/distil-large-v3).

Let's start by loading the teacher model and processor. We do this in much the same way we loaded the Distil-Whisper 
model in the previous examples:

```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
```

Now let's load the assistant. Since Distil-Whisper shares exactly same encoder as the teacher model, we only need 
to load the 2-layer decoder as a "Decoder-only" model:

```python
from transformers import AutoModelForCausalLM
assistant_model_id = "distil-whisper/distil-large-v2"

assistant_model = AutoModelForCausalLM.from_pretrained(
    assistant_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
assistant_model.to(device)
```

The assistant model shares the same processor as the teacher, so there's no need to load a student processor.

We can now pass the assistant model to the pipeline to be used for speculative decoding. We pass it as a `generate_kwarg`
with the key [`"assistant_model"`](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate.assistant_model) 
so that speculative decoding is enabled:

```python
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    generate_kwargs={"assistant_model": assistant_model},
    torch_dtype=torch_dtype,
    device=device,
)
```

As before, we can pass any sample to the pipeline to be transcribed:

```python
from datasets import load_dataset

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])
```

**Note:** speculative decoding should be on average 2x faster than using "only" Whisper large-v2 at a mere 8% increase 
in VRAM memory usage while mathematically ensuring the same results. This makes it the perfect replacement for Whisper large-v2
in existing speech recognition pipelines.

For more details on speculative decoding, refer to the following resources:
* [Speculative decoding for 2x faster Whisper inference](https://huggingface.co/blog/whisper-speculative-decoding) blog post by Sanchit Gandhi
* [Assisted Generation: a new direction toward low-latency text generation](https://huggingface.co/blog/assisted-generation) blog post by Joao Gante
* [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) paper by Leviathan et. al.

### Additional Speed & Memory Improvements

You can apply additional speed and memory improvements to Distil-Whisper which we cover in the following.

#### Flash Attention

We recommend using [Flash Attention 2](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#flashattention-2) if your GPU allows for it.
To do so, you first need to install [Flash Attention](https://github.com/Dao-AILab/flash-attention):

```
pip install flash-attn --no-build-isolation
```

You can then pass `use_flash_attention_2=True` to `from_pretrained` to enable Flash Attention 2:

```diff
- model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
+ model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, use_flash_attention_2=True)
```

#### Torch Scale-Product-Attention (SDPA)

If your GPU does not support Flash Attention, we recommend making use of [BetterTransformers](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#bettertransformer).
To do so, you first need to install optimum:

```
pip install --upgrade optimum
```

And then convert your model to a "BetterTransformer" model before using it:

```diff
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
+ model = model.to_bettertransformer()
```

### Exporting to Other Libraries

Distil-Whisper has support in the following libraries with the original "sequential" long-form transcription algorithm. 
Click the links in the table to see the relevant code-snippets for each:

| Library         | distil-small.en                                                                                 | distil-medium.en                                                                                 | distil-large-v2                                                                                 |
|-----------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| OpenAI Whisper  | [link](https://huggingface.co/distil-whisper/distil-small.en#running-whisper-in-openai-whisper) | [link](https://huggingface.co/distil-whisper/distil-medium.en#running-whisper-in-openai-whisper) | [link](https://huggingface.co/distil-whisper/distil-large-v2#running-whisper-in-openai-whisper) |
| Whisper cpp     | [link](https://huggingface.co/distil-whisper/distil-small.en#whispercpp)                        | [link](https://huggingface.co/distil-whisper/distil-medium.en#whispercpp)                        | [link](https://huggingface.co/distil-whisper/distil-large-v2#whispercpp)                        |
| Transformers js | [link](https://huggingface.co/distil-whisper/distil-small.en#transformersjs)                    | [link](https://huggingface.co/distil-whisper/distil-medium.en#transformersjs)                    | [link](https://huggingface.co/distil-whisper/distil-large-v2#transformersjs)                    |
| Candle (Rust)   | [link](https://huggingface.co/distil-whisper/distil-small.en#candle)                            | [link](https://huggingface.co/distil-whisper/distil-medium.en#candle)                            | [link](https://huggingface.co/distil-whisper/distil-large-v2#candle)                            |

Updates will be posted here with the integration of the "chunked" long-form transcription algorithm into the respective 
libraries.

For the 🤗 Transformers code-examples, refer to the sections [Short-Form](#short-form-transcription) and [Long-Form](#long-form-transcription) Transcription.

## 2. Why use Distil-Whisper? ⁉️

Distil-Whisper is designed to be a drop-in replacement for Whisper on English speech recognition. Here are 5 reasons for making the
switch to Distil-Whisper:

1. **Faster inference:** 6 times faster inference speed, while performing to within 1% WER of Whisper on out-of-distribution audio:

<p align="center">
  <img src="https://huggingface.co/datasets/distil-whisper/figures/resolve/main/main_table.png?raw=true" width="600"/>
</p>

2. **Robustness to noise:** demonstrated by strong WER performance at low signal-to-noise ratios:

<p align="center">
  <img src="https://huggingface.co/datasets/distil-whisper/figures/resolve/main/noise.png?raw=true" width="600"/>
</p>

3. **Robustness to hallucinations:** quantified by 1.3 times fewer repeated 5-gram word duplicates (5-Dup.) and 2.1% lower insertion error rate (IER) than Whisper:

<p align="center">
  <img src="https://huggingface.co/datasets/distil-whisper/figures/resolve/main/hallucination.png?raw=true" width="600"/>
</p>

4. **Designed for speculative decoding:** Distil-Whisper can be used as an assistant model to Whisper, giving 2 times faster inference speed while mathematically ensuring the same outputs as the Whisper model.
5. **Permissive license:** Distil-Whisper is [MIT licensed](./LICENSE), meaning it can be used for commercial applications.

## 3. Approach ✍️

To distill Whisper, we copy the entire encoder module and freeze it during training. We copy only two decoder layers, 
which are initialised from the first and last decoder layers from Whisper. All other decoder layers from Whisper
are discarded:

<p align="center">
  <img src="https://huggingface.co/datasets/distil-whisper/figures/resolve/main/architecture.png?raw=true" width="600"/>
</p>

Distil-Whisper is trained on a *knowledge distillation* objective. Specifically, it is trained to minimise the KL divergence
between the distilled model and the Whisper model, as well as the cross-entropy loss on pseudo-labelled audio data.

We train Distil-Whisper on a total of 22k hours of pseudo-labelled audio data, spanning 10 domains with over 18k speakers:

<p align="center">
  <img src="https://huggingface.co/datasets/distil-whisper/figures/resolve/main/datasets.png?raw=true" width="600"/>
</p>

This diverse audio dataset is paramount to ensuring robustness of Distil-Whisper to different datasets and domains. 

In addition, we use a WER filter to discard pseudo-labels where Whisper mis-transcribes or hallucinates. This greatly 
improves WER performance of the downstream distilled model.

For full details on the distillation set-up and evaluation results, refer to the [Distil-Whisper paper](https://arxiv.org/abs/2311.00430).

## 4. Training Code

Training code to reproduce Distil-Whisper can be found in the directory [training](training). This code has been adapted 
be general enough to distill Whisper for multilingual speech recognition, facilitating anyone in the community to distill 
Whisper on their choice of language.

## 5. Acknowledgements
* OpenAI for the Whisper [model](https://huggingface.co/openai/whisper-large-v3) and [original codebase](https://github.com/openai/whisper)
* Hugging Face 🤗 [Transformers](https://github.com/huggingface/transformers) for the model integration
* Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) program for Cloud TPU v4s

## 6. Citation

If you use this model, please consider citing the Distil-Whisper paper:
```
@misc{gandhi2023distilwhisper,
      title={Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling}, 
      author={Sanchit Gandhi and Patrick von Platen and Alexander M. Rush},
      year={2023},
      eprint={2311.00430},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

And also the Whisper paper:
```
@misc{radford2022robust,
      title={Robust Speech Recognition via Large-Scale Weak Supervision}, 
      author={Alec Radford and Jong Wook Kim and Tao Xu and Greg Brockman and Christine McLeavey and Ilya Sutskever},
      year={2022},
      eprint={2212.04356},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
