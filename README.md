# Distil-Whisper

[[Paper]](https://arxiv.org/abs/2311.00430)
[[Models]](https://huggingface.co/collections/distil-whisper/distil-whisper-models-65411987e6727569748d2eb6)
[[wandb]](https://wandb.ai/sanchit-gandhi/distil-whisper/workspace?workspace=user-sanchit-gandhi)

Distil-Whisper is a distilled version of Whisper that is **6 times faster**, 49% smaller, and performs **within 1% word error rate (WER)** on 
out-of-distribution evaluation sets.

| Model                                                                      | Params / M | Rel. Latency | Short-Form WER | Long-Form WER |
|----------------------------------------------------------------------------|------------|--------------|----------------|---------------|
| [whisper-large-v2](https://huggingface.co/openai/whisper-large-v2)         | 1550       | 1.0          | **9.1**        | 11.7          |
|                                                                            |            |              |                |               |
| [distil-large-v2](https://huggingface.co/distil-whisper/distil-large-v2)   | 756        | 5.8          | 10.1           | **11.6**      |
| [distil-medium.en](https://huggingface.co/distil-whisper/distil-medium.en) | **394**    | **6.8**      | 11.1           | 12.4          |

## 1. Usage

Distil-Whisper is supported in Hugging Face 🤗 Transformers from version 4.35 onwards. To run the model, first 
install the latest version of the Transformers library. For this example, we'll also install 🤗 Datasets to load toy 
audio dataset from the Hugging Face Hub:

```bash
pip install --upgrade pip
pip install --upgrade transformers accelerate datasets[audio]
```

### Short-Form Transcription

The model can be used with the [`pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline)
class to transcribe short-form audio files as follows:

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])
```

To transcribe a local audio file, simply pass the path to your audio file when you call the pipeline:
```diff
- result = pipe(sample)
+ result = pipe("audio.mp3")
```

### Long-Form Transcription

Distil-Whisper uses a chunked algorithm to transcribe long-form audio files. In practice, this chunked long-form algorithm 
is 9x faster than the sequential algorithm proposed by OpenAI in the Whisper paper (see Table 7 of the [Distil-Whisper paper](https://arxiv.org/abs/2311.00430)).

To enable chunking, pass the `chunk_length_s` parameter to the `pipeline`. For Distil-Whisper, a chunk length of 15-seconds
is optimal. To activate batching, pass the argument `batch_size`:

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "default", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])
```

<!---
**Tip:** The pipeline can also be used to transcribe an audio file from a remote URL, for example:

```python
result = pipe("https://huggingface.co/datasets/sanchit-gandhi/librispeech_long/resolve/main/audio.wav")
```
--->

### Speculative Decoding

Distil-Whisper can be used as an assistant model to Whisper for speculative decoding. Speculative decoding mathematically
ensures the exact same outputs as Whisper are obtained while being 2 times faster. This makes it the perfect drop-in 
replacement for existing Whisper pipelines, since the same outputs are guaranteed.

In the following code-snippet, we load the assistant Distil-Whisper model standalone to the main Whisper pipeline. We then
specify it as the "assistant model" for generation:

```python
from transformers import pipeline, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

assistant_model_id = "distil-whisper/distil-large-v2"

assistant_model = AutoModelForCausalLM.from_pretrained(
    assistant_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
assistant_model.to(device)

model_id = "openai/whisper-large-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

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

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])
```

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

Training code to reproduce Distil-Whisper will be published here shortly. We will also release more general code to distill
Whisper for multilingual speech recognition, facilitating anyone in the community to distill Whisper on their choice of 
language.

## 5. Acknowledgements
* OpenAI for the Whisper [model](https://huggingface.co/openai/whisper-large-v2) and [original codebase](https://github.com/openai/whisper)
* Hugging Face 🤗 [Transformers](https://github.com/huggingface/transformers) for the model integration
* Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) programme for Cloud TPU v4s

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
