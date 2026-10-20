# Distil-Whisper

[English](README.md) | [简体中文](README.zh.md)

[[论文]](https://arxiv.org/abs/2311.00430)
[[模型集合]](https://huggingface.co/collections/distil-whisper/distil-whisper-models-65411987e6727569748d2eb6)
[[Colab 评测]](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/Distil_Whisper_Benchmark.ipynb)
[[训练代码]](training)

Distil-Whisper 是面向英语语音识别的 Whisper 知识蒸馏版本，其推理速度**提升 6 倍**，体积缩减 49%，在分布外（out-of-distribution）评测集上的**词错误率（WER）差距在 1% 以内**：

| 模型 | 参数量 (M) | 相对延迟 ↑ | 短音频 WER ↓ | 长音频 WER ↓ |
| :--- | :--- | :--- | :--- | :--- |
| [large-v3](https://huggingface.co/openai/whisper-large-v3) | 1550 | 1.0 | **8.4** | 11.0 |
| | | | | |
| [distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) | 756 | 6.3 | 9.7 | **10.8** |
| [distil-large-v2](https://huggingface.co/distil-whisper/distil-large-v2) | 756 | 5.8 | 10.1 | 11.6 |
| [distil-medium.en](https://huggingface.co/distil-whisper/distil-medium.en) | 394 | **6.8** | 11.1 | 12.4 |
| [distil-small.en](https://huggingface.co/distil-whisper/distil-small.en) | **166** | 5.6 | 12.1 | 12.8 |

对于大多数应用场景，我们推荐使用最新的 [distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) 检查点，因为它是性能最强劲的蒸馏检查点，且兼容所有 Whisper 库生态。唯一的例外是资源极度受限且内存非常小的场景（如端侧设备或移动应用），在此类场景下 [distil-small.en](https://huggingface.co/distil-whisper/distil-small.en) 是极佳的选择——它仅有 166M 参数，且与 Whisper large-v3 的 WER 差距在 4% 以内。

> [!NOTE]
> Distil-Whisper 目前仅支持英语语音识别。对于多语言语音识别，我们推荐使用 OpenAI 官方发布并沿用了 Distil-Whisper 相同原理构建的 [Whisper Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) 检查点。有关详细信息，请参阅 Whisper Turbo 的[发布公告](https://github.com/openai/whisper/discussions/2363)。

## 1. 使用方法

从 4.35 版本起，Hugging Face 🤗 Transformers 已原生支持 Distil-Whisper。要运行该模型，首先安装最新版本的 Transformers 库。在此示例中，我们还将安装 🤗 Datasets，以便从 Hugging Face Hub 加载示例音频数据集：

```bash
pip install --upgrade pip
pip install --upgrade transformers accelerate datasets[audio]
```

### 短音频转录

短音频转录是指转录长度小于 30 秒的音频样本的过程，30 秒也是 Whisper 模型的最大感受野。这意味着无需分块（chunking），整个音频片段即可一次性处理完成。

首先，我们通过便捷的 [`AutoModelForSpeechSeq2Seq`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSpeechSeq2Seq) 和 [`AutoProcessor`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoProcessor) 类加载 Distil-Whisper。

我们以 `float16` 精度加载模型，并通过传入 `low_cpu_mem_usage=True` 确保加载时间尽可能短。此外，通过传入 `use_safetensors=True` 确保模型以 [`safetensors`](https://github.com/huggingface/safetensors) 格式加载：

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

随后将模型与处理器传入 [`pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline)。如果希望对生成过程进行更多控制，也可以直接使用模型 + 处理器的原生 API。

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

接下来，从 LibriSpeech 语料库加载示例短音频：

```python
from datasets import load_dataset

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = dataset[0]["audio"]
```

最后，调用 pipeline 进行音频转录：

```python
result = pipe(sample)
print(result["text"])
```

要转录本地音频文件，只需在调用 pipeline 时传入本地文件路径即可：

```python
result = pipe("audio.mp3")
print(result["text"])
```

关于如何自定义自动语音识别流水线的更多信息，请参阅 ASR pipeline [官方文档](https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline)。
我们还提供了一个端到端的 [Google Colab](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/Distil_Whisper_Benchmark.ipynb)，对 Whisper 与 Distil-Whisper 进行基准评测对比。

<details>
<summary> 如需对生成参数进行更多精细控制，可直接使用模型 + 处理器 API： </summary>

可向 `model.generate` 传入自定义生成参数，包括用于束搜索的 `num_beams`、用于分段级时间戳的 `return_timestamps` 以及用于 Prompt 引导的 `prompt_ids`。详见[文档字符串](https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate)。

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

### 顺序长音频转录

最新的 [distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) 检查点专为兼容 OpenAI 原生顺序长音频转录算法而设计。该算法采用滑动窗口对长音频文件（> 30 秒）进行缓冲推理，与[分块长音频算法](#分块长音频转录)相比，转录精度更高。

以下两种场景建议使用顺序长音频算法：
1. 转录精度是最关键的考量因素，对延迟要求相对次要
2. 正在进行长音频文件的**批量转录（batches）**，此时顺序算法的延迟与分块算法相当，而 WER 精度最高可提升 0.5%

如果是单条长音频文件转录且延迟是最核心的要求，则应使用[下文](#分块长音频转录)介绍的分块算法。有关不同算法的详细解释，请参阅 [Distil-Whisper 论文](https://arxiv.org/pdf/2311.00430.pdf) 第 5 节。

我们同样首先加载模型与处理器：

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

随后将模型与处理器传入 [`pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline)。如果希望对生成过程进行更多控制，也可以直接使用 `model.generate(...)` API。

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

接下来，加载长音频样本。这里使用 LibriSpeech 语料库中拼接的示例音频：

```python
from datasets import load_dataset

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]
```

最后，调用 pipeline 进行音频转录：

```python
result = pipe(sample)
print(result["text"])
```

要转录本地音频文件，只需在调用 pipeline 时传入本地文件路径即可：

```python
result = pipe("audio.mp3")
print(result["text"])
```

<details>

<summary> 如需对生成参数进行更多精细控制，可直接使用模型 + 处理器 API： </summary>

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
    "compression_ratio_threshold": 1.35,  # zlib 压缩率阈值（在 Token 空间）
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

### 分块长音频转录

distil-large-v3 保持了与 Transformers 分块长音频算法的兼容性。当需要转录单个大型音频文件并追求尽可能极致的推理速度时，应使用该算法。在这种情况下，分块算法相比 OpenAI 的顺序长音频实现最高可提升 9 倍速度（见 [Distil-Whisper 论文](https://arxiv.org/pdf/2311.00430.pdf) 表 7）。

我们可以如前加载模型与处理器：

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

通过向 `pipeline` 传入 `chunk_length_s` 参数启用分块。对于 distil-large-v3，25 秒的分块长度是最优的。通过传入参数 `batch_size` 激活批处理：

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

参数 `max_new_tokens` 控制**每个分块**生成的最大 Token 数量。在常见的语音场景中，每秒语速通常不超过 3 个词。因此对于 30 秒的输入，最多产生约 90 个词（约 128 个 Token）。我们将每块最大生成 Token 数设置为 128，以截断段尾可能出现的幻觉。长音频分块转录算法会在分块边界处移除这些冗余 Token，因此在生成期间直接截断更为高效，可避免解码器产生多余的生成步数。

接下来加载长音频样本：

```python
from datasets import load_dataset

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]
```

最后，调用 pipeline 进行音频转录：

```python
result = pipe(sample)
print(result["text"])
```

关于如何自定义自动语音识别流水线的更多信息，请参阅 ASR pipeline [官方文档](https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline)。

### 投机解码

Distil-Whisper 可作为 Whisper 的辅助模型（Assistant Model）进行[投机解码（Speculative Decoding）](https://huggingface.co/blog/whisper-speculative-decoding)。
投机解码在数学上保证获得与原生 Whisper 完全一致的输出结果，同时速度提升 2 倍。这使其成为现有 Whisper 流水线的完美即插即用替代方案，因为输出一致性得到了绝对保证。

在投机解码中，我们需要同时加载教师模型 [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) 以及辅助学生模型 [`distil-whisper/distil-large-v3`](https://huggingface.co/distil-whisper/distil-large-v3)。

首先加载教师模型与处理器，加载方式与前面示例类似：

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

接着加载辅助模型。由于 Distil-Whisper 与教师模型共享完全相同的编码器，我们只需将 2 层解码器作为“仅解码器（Decoder-only）”模型进行加载：

```python
from transformers import AutoModelForCausalLM
assistant_model_id = "distil-whisper/distil-large-v2"

assistant_model = AutoModelForCausalLM.from_pretrained(
    assistant_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
assistant_model.to(device)
```

辅助模型与教师模型共享完全相同的处理器，因此无需单独加载学生处理器。

现在我们可以将辅助模型传递给 pipeline 用于投机解码。将其作为 `generate_kwargs` 中的 [`"assistant_model"`](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate.assistant_model) 键传入，即可启用投机解码：

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

与前面一样，将音频样本传入 pipeline 进行转录：

```python
from datasets import load_dataset

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])
```

**注意**：投机解码在仅增加约 8% 显存（VRAM）占用的情况下，平均可比“单用”Whisper large-v2 快 2 倍，同时在数学上保证输出完全一致。这使其成为现有语音识别流水线中替换 Whisper large-v2 的绝佳选择。

更多关于投机解码的细节，请参阅以下资源：
* Sanchit Gandhi 的博客文章：[投机解码实现 2 倍速 Whisper 推理](https://huggingface.co/blog/whisper-speculative-decoding)
* Joao Gante 的博客文章：[辅助生成：迈向低延迟文本生成的新方向](https://huggingface.co/blog/assisted-generation)
* Leviathan 等人的学术论文：[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)

### 额外的速度与显存优化

您可以对 Distil-Whisper 应用以下额外的速度与显存优化：

#### Flash Attention

如果您的 GPU 支持，推荐使用 [Flash Attention 2](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#flashattention-2)。
为此，首先需要安装 [Flash Attention](https://github.com/Dao-AILab/flash-attention)：

```
pip install flash-attn --no-build-isolation
```

随后在 `from_pretrained` 中传入 `use_flash_attention_2=True` 启用 Flash Attention 2：

```diff
- model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
+ model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, use_flash_attention_2=True)
```

#### Torch 缩放点积注意力 (SDPA)

如果您的 GPU 不支持 Flash Attention，建议使用 [BetterTransformers](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#bettertransformer)。
为此，首先安装 Optimum：

```
pip install --upgrade optimum
```

并在使用前将模型转换为 BetterTransformer 模型：

```diff
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
+ model = model.to_bettertransformer()
```

### 导出至其他生态库

Distil-Whisper 在以下生态库中原生支持原生“顺序”长音频转录算法。点击表格中的链接可查看各自的代码示例：

| 库 | distil-small.en | distil-medium.en | distil-large-v2 |
| :--- | :--- | :--- | :--- |
| OpenAI Whisper | [链接](https://huggingface.co/distil-whisper/distil-small.en#running-whisper-in-openai-whisper) | [链接](https://huggingface.co/distil-whisper/distil-medium.en#running-whisper-in-openai-whisper) | [链接](https://huggingface.co/distil-whisper/distil-large-v2#running-whisper-in-openai-whisper) |
| Whisper cpp | [链接](https://huggingface.co/distil-whisper/distil-small.en#whispercpp) | [链接](https://huggingface.co/distil-whisper/distil-medium.en#whispercpp) | [链接](https://huggingface.co/distil-whisper/distil-large-v2#whispercpp) |
| Transformers js | [链接](https://huggingface.co/distil-whisper/distil-small.en#transformersjs) | [链接](https://huggingface.co/distil-whisper/distil-medium.en#transformersjs) | [链接](https://huggingface.co/distil-whisper/distil-large-v2#transformersjs) |
| Candle (Rust) | [链接](https://huggingface.co/distil-whisper/distil-small.en#candle) | [链接](https://huggingface.co/distil-whisper/distil-medium.en#candle) | [链接](https://huggingface.co/distil-whisper/distil-large-v2#candle) |

随着“分块”长音频转录算法集成到对应库中，相关更新将在此处同步发布。

有关 🤗 Transformers 的代码示例，请参阅[短音频转录](#短音频转录)和[长音频转录](#顺序长音频转录)章节。

## 2. 为什么选择 Distil-Whisper？⁉️

Distil-Whisper 专为在英语语音识别任务中无缝替换 Whisper 而设计。选择切换到 Distil-Whisper 的 5 大理由：

1. **更快的推理速度**：推理速度提升 6 倍，在分布外音频上的 WER 与 Whisper 相差在 1% 以内：

<p align="center">
  <img src="https://huggingface.co/datasets/distil-whisper/figures/resolve/main/main_table.png?raw=true" width="600"/>
</p>

2. **抗噪鲁棒性**：在低信噪比环境下展现出强劲的 WER 表现：

<p align="center">
  <img src="https://huggingface.co/datasets/distil-whisper/figures/resolve/main/noise.png?raw=true" width="600"/>
</p>

3. **抗幻觉鲁棒性**：量化测试显示，相比 Whisper，5-gram 重复率（5-Dup.）降低 1.3 倍，插入错误率（IER）降低 2.1%：

<p align="center">
  <img src="https://huggingface.co/datasets/distil-whisper/figures/resolve/main/hallucination.png?raw=true" width="600"/>
</p>

4. **专为投机解码设计**：Distil-Whisper 可作为 Whisper 的辅助模型，在数学上保证输出完全一致的同时提供 2 倍的推理速度。
5. **友好的开源许可**：Distil-Whisper 采用 [MIT 许可证](./LICENSE)，意味着可自由用于商业应用。

## 3. 技术方案 ✍️

为了蒸馏 Whisper，我们完整复制了编码器（Encoder）模块并在训练期间予以冻结。我们仅复制了两个解码器（Decoder）层，分别从 Whisper 的第一层与最后一层解码器初始化。Whisper 中的所有其他解码器层均被舍弃：

<p align="center">
  <img src="https://huggingface.co/datasets/distil-whisper/figures/resolve/main/architecture.png?raw=true" width="600"/>
</p>

Distil-Whisper 基于*知识蒸馏（Knowledge Distillation）*目标进行训练。具体而言，模型被训练为最小化蒸馏模型与 Whisper 模型之间的 KL 散度，以及在伪标签（pseudo-labelled）音频数据上的交叉熵损失。

我们在总计 2.2 万小时的伪标签音频数据上训练 Distil-Whisper，涵盖 10 个领域以及超过 1.8 万名说话人：

<p align="center">
  <img src="https://huggingface.co/datasets/distil-whisper/figures/resolve/main/datasets.png?raw=true" width="600"/>
</p>

这种高度多样化的音频数据集对于确保 Distil-Whisper 在不同数据集与领域间的鲁棒性至关重要。

此外，我们使用 WER 过滤器剔除了 Whisper 发生转录错误或幻觉的伪标签。这大幅提升了下游蒸馏模型的 WER 表现。

有关蒸馏环境设置和评测结果的完整详情，请参阅 [Distil-Whisper 论文](https://arxiv.org/abs/2311.00430)。

## 4. 训练代码

复现 Distil-Whisper 的训练代码位于 [training](training) 目录中。该代码经过调整，具备足够通用性以将 Whisper 蒸馏用于多语言语音识别，方便社区中的任何人将其蒸馏到自己所选的语言上。

## 5. 致谢

* 感谢 OpenAI 提供 Whisper [模型](https://huggingface.co/openai/whisper-large-v3) 与 [原始代码库](https://github.com/openai/whisper)
* 感谢 Hugging Face 🤗 [Transformers](https://github.com/huggingface/transformers) 进行模型集成
* 感谢 Google 的 [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) 计划提供 Cloud TPU v4 算力支持

## 6. 论文引用

如果您使用了该模型，请考虑引用 Distil-Whisper 论文：
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

以及 Whisper 原始论文：
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

---

> **免责声明**：本文档由社区志愿者进行本地化翻译，旨在促进跨语言的技术交流与学习。如翻译内容与[英文原版](README.md)存在出入，请以英文原版为准。
