## Training Distil-Whisper

This sub-folder contains all the scripts required to train a Distil-Whisper model in your choice of language. They are 
slightly modified from the original scripts used to distill Whisper for English ASR (as-per the [Distil-Whisper paper](https://arxiv.org/abs/2311.00430)).
The main difference is that these scripts are written in [PyTorch](https://pytorch.org), whereas the original scripts 
are in [JAX](https://jax.readthedocs.io/en/latest/#)/[Flax](https://flax.readthedocs.io/en/latest/). These scripts are 
also made to be easier to run end-to-end, whereas the original scripts require more steps and are somewhat hard-coded 
for English ASR. Both sets of scripts achieve equivalent downstream results when the hyper-parameters are set equal.

If you are interested in reproducing the original Distil-Whisper checkpoints, we refer you to the sub-folder [Flax Training](./flax/README.md).
Otherwise, if you wish to distill Whisper on your own language/dataset, we recommend you use these scripts for ease of use
and the configurability they provide.

Reproducing the Distil-Whisper project requires three stages to be completed in successive order:

1. [Pseudo-labelling](#1-pseudo-labelling)
2. [Initialisation](#2-initialisation)
3. [Training](#3-training)

This README is partitioned according to the three stages. Each section provides a minimal example for running the
scripts used in the project. We will use a running example of distilling the Whisper model for Hindi speech recognition
on the Common Voice dataset.

## Overview of Training Methods

### 1. Fine-Tuning

For fine-tuning, we take the original Whisper checkpoint and train it on one or more datasets using the standard 
cross-entropy loss. As such, there is no involvement from the teacher checkpoint during training, and so the fine-tuned 
model is permitted to *overfit* to the distribution of the training data we provide. This makes it appealing for "low-resource" 
languages where the original Whisper model performs poorly, since we can boost the performance of the model on a single 
language by *overfitting* to that distribution of data. Note that this means the fine-tuned model is prone to loosing 
its robustness to different audio distributions, which is the trade-off with improving performance on a specified dataset.

As a rule of thumb, fine-tuning is appropriate for languages where the original Whisper model performs > 20% WER, and we 
have a relatively small quantity of training data available (< 100 hours). With fine-tuning, we require as little as **10 hours**
of training data to significantly boost the performance of the Whisper model. For an in-depth guide to fine-tuning Whisper,
the reader is advised to refer to the blog post: [Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper).

### 2. Shrink and Fine-Tune

For shrink and fine-tune (SFT), we first *shrink* the teacher model to a smaller student model by copying maximally 
spaced layers, and then *fine-tune* the student model on the cross-entropy loss as described above. Typically, we retain 
the full encoder from the Whisper model and only shrink the decoder. Retaining the entire encoder helps significantly with
maintaining Whisper's robustness to different audio distributions (_c.f._ Section 9.3 of the [Distil-Whisper paper](https://arxiv.org/abs/2311.00430)).

We can either train the student model on a dataset of (audio, text) pairs as above. Or, we can use the pre-trained 
Whisper model to generate *pseudo-labels* for our audio data, and train on the (audio, pseudo-label) pairs.

Pseudo-labels can be used when either:
1. The original text transcriptions are normalised (lower-cased or no punctuation): the Whisper generated pseudo-labels contain both punctuation and casing, and so can be used as a substitute for the normalised transcriptions
2. The pre-trained Whisper model achieves < 20% WER on the languages: we then know the majority of the pseudo-labels will be accurate enough for us to train on.

They are not recommended when both of the following are true:
1. The original text is punctuated and cased
2. The pre-trained Whisper model achieves > 20% WER on the languages: in this case, we want to overfit to the particular distribution of the language, and so train directly on the original text data

To discard inaccurate pseudo-labels during training, we employ a simple WER heuristic to filter our pseudo-labelled 
training data. We first normalise the original text and the pseudo-labelled text using the Whisper normaliser. If the 
WER between the normalised text exceeds a 10% WER threshold, we discard the training sample. Else, we retain it for training.
Section 9.1 of the Distil-Whisper [paper](https://arxiv.org/abs/2311.00430) demonstrates the importance of using this 
threshold for training.

### 3. KL Divergence

In the KL Divergence setting, the student model is initialised by shrinking the teacher as before, and then trained to 
match the predictions of the teacher during training. 

### Summary of Methods

The following table summarises the three training methods and the minimum suggested WER / training data for each:

| Method      | Pre-Trained WER / % | Training Data / h |
|-------------|---------------------|-------------------|
| Fine-tuning | > 20                | < 100             |
| PL          | < 20                | > 100             |
| KD          | < 20                | > 100             |

## Requirements

The Distil-Whisper training code is written in [PyTorch](https://pytorch.org) and [Accelerate](https://huggingface.co/docs/accelerate/index). 
It heavily leverages the Whisper implementation in [🤗 Transformers](https://github.com/huggingface/transformers) for both 
training and inference.

The instructions for installing the package are as follows:
1. Install PyTorch from the [official instructions](https://pytorch.org/get-started/locally/), ensuring you install the correct version for your hardware and CUDA version.
2. Fork the `distil-whisper` repository by clicking on the [fork](https://github.com/huggingface/distil-whisper/fork) button on the reopsitory's page
3. Clone the `distil_whisper` repository and add the base repository as a remote. This will allow you to "pull" any upstream changes that are made to the base repository:

```bash
git clone https://github.com/<your GitHub handle>/distil-whisper.git
cd distil-whisper
git remote add upstream https://github.com/huggingface/distil-whisper.git
```
4. pip install the required packages from the [requirements.txt](#requirements) file:
```bash
cd training
pip install -r requirements.txt
cd ../..
```

5. Configure Accelerate by running the following command. Note that you should set the number of GPUs you wish to use for distillation, and also the data type (dtype) to your preferred dtype for training/inference (e.g. `bfloat16` on A100 GPUs, `float16` on V100 GPUs, etc.):

```bash
accelerate config
```

6. The last thing we need to do is link our Hugging Face account so that we can pull/push model repositories on the Hub. This will allow us to save our final distilled weights on the Hub so that we can share them with the community. Run the command:

```bash
git config --global credential.helper store
huggingface-cli login
```
And then enter an authentication token from https://huggingface.co/settings/tokens. Create a new token if you do not have one already. You should make sure that this token has "write" privileges.

To confirm that you have a working environment, you can run the following code cell to stream one sample of data from the 
LibriSpeech dataset, and check that you can perform a forward pass of the "tiny" Whisper model:

```python
import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration
from datasets import load_dataset

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny", low_cpu_mem_usage=True)
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

model.to("cuda")

librispeech_asr = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)

inputs = feature_extractor(next(iter(librispeech_asr))["audio"]["array"], sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features

decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
logits = model(input_features.to("cuda"), decoder_input_ids=decoder_input_ids.to("cuda")).logits

print("Environment set up successful?", logits.shape[-1] == 51865)
```

## 1. Pseudo-Labelling

The python script [`run_pseudo_labelling.py`](run_pseudo_labelling.py) is a flexible inference script that can be used
to generate pseudo-labels under a range of settings, including using both greedy and beam-search. It is also compatible
with [🤗 Datasets](https://github.com/huggingface/datasets) *streaming mode*, allowing users to load massive audio
datasets with **no disk space requirements**. For more information on streaming mode, the reader is referred to the 
blog post: [A Complete Guide to Audio Datasets](https://huggingface.co/blog/audio-datasets#streaming-mode-the-silver-bullet).

The following script demonstrates how to pseudo-label the Hindi split of the Common Voice 13 dataset with greedy sampling:

```bash
#!/usr/bin/env bash

accelerate launch run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-large-v2" \
  --dataset_name "mozilla-foundation/common_voice_13_0" \
  --dataset_config_name "hi" \
  --data_split_name "train+validation+test" \
  --text_column_name "sentence" \
  --id_column_name "path" \
  --output_dir "./common_voice_13_0_hi_pseudo_labelled" \
  --wandb_project "distil-whisper-labelling" \
  --per_device_eval_batch_size 64 \
  --dtype "bfloat16" \
  --dataloader_num_workers 16 \
  --preprocessing_num_workers 16 \
  --logging_steps 500 \
  --max_label_length 128 \
  --report_to "wandb" \
  --language "hi" \
  --task "transcribe" \
  --return_timestamps \
  --attn_type "flash_attn" \
  --streaming False \
  --generation_num_beams 1 \
  --decode_token_ids False \
  --push_to_hub
```

On an 80 GB A100 GPU, the following script takes approximately 20 minutes to transcribe a total of 20 hours of audio data.
The WER of the pre-trained model is 23.8% on the test split. 

There are a few noteworthy arguments that can be configured:
1. `language`: explicitly setting the language token during inference substantially improves the generation performance of the Whisper model, since the model is forced always to predict in the given language. We recommend you set the language to the language you wish to distil the Whisper model on. The only exception is when distilling an English-only model (i.e. where the model id is appended with an `.en`, e.g. `small.en`), the language argument should be set to None, since there is no language token used during training/inference.
2. `return_timestamps`: whether or not to predict timestamps in the pseudo-labels. Timestamp prediction is required should you want your distilled model to be able to predict timestamps at inference time (e.g. for the original OpenAI long-form transcription algorithm). However, the pseudo-labels are marginally less accurate than not using timestamps. We recommend pseudo-labelling **with** timestamps to ensure the distilled model is as general as possible.
3. `attn_type`: which attention implementation to use for inference. Set to `flash_attn` for [PyTorch SDPA](https://huggingface.co/docs/transformers/v4.35.2/en/perf_infer_gpu_one#bettertransformer), or `flash_attn_2` if your hardware supports Flash Attention 2 and you have the [package installed](https://github.com/Dao-AILab/flash-attention).
4. `streaming`: whether or not to use Datasets' streaming mode. If enabled, the audio data will be streamed from the Hugging Face Hub with no disk space requirements. However, the user is then responsible for adding the pseudo-labels to the dataset script in a follow-up step (see [Using Streaming Mode](#TODO)). If set to `False`, the audio data will be downloaded and pre-processed offline. At the end of pseudo-labelling, the pseudo-labels will be automatically appended to the original dataset, meaning the dataset is ready to be used for the subsequent training step without any additional steps.
5. `generation_num_beams`: how many beams to use while decoding. In practice, we found the distilled model to perform comparably when the data was pseudo-labelled with `generation_num_beams=1` (greedy) or `generation_num_beams>1` (beam). This is likely because the WER filter compensates for the lower quality pseudo-labels obtained using greedy search. However, using `generation_num_beams=1` gives substantially faster inference time for the pseudo-labelling step, and so we recommend this configuration.
6. `decode_token_ids`: whether or not to decode the generated token ids to text transcriptions. If set to `False`, the token ids generated by the pre-trained Whisper model will be used as the targets during training. If set to `True`, the token ids will first be converted to their corresponding text transcriptions, and then re-tokenised to give the token id targets during training. The Whisper tokenizer does not preserve the same token ids across encoding/decoding steps, that is `encode(decode(token_ids)) != token_ids`. This means that decoding the token ids to text transcriptions changes the distribution of the token ids used as the targets during training. In our experiments, we found that setting `decode_token_ids=True` resulting in better downstream performance of the distilled model, but aligned worse with the original teacher model for speculative decoding. This is likely due to a ['label smoothing'](https://arxiv.org/abs/1906.02629) effect from changing the distribution of the token ids. Should your goal be training a fast assistant model for speculative decoding, we recommend you set this argument to `True`. If your goal is a fast standalone model, you can try setting this to `False` as we have done above.

Should you have your own audio dataset, you can first [convert it](https://huggingface.co/docs/datasets/audio_dataset) to 
Hugging Face Datasets format and push it to the Hugging Face Hub. You can then pseudo-label it using the script above, 
replacing the `--dataset_name` with the name of your dataset on the Hub.

Otherwise, you may wish to use an open-source dataset already available on the Hugging Face Hub. We provide a summary of 
the three most popular multilingual datasets in the table below. For more details, refer to the blog post: [A Complete Guide to Audio Datasets](https://huggingface.co/blog/audio-datasets#multilingual-speech-recognition).

| Dataset                                                                                       | Languages | Domain                                | Speaking Style | License   | Text Column  | ID Column    |
|-----------------------------------------------------------------------------------------------|-----------|---------------------------------------|----------------|-----------|--------------|--------------|
| [Multilingual LibriSpeech](https://huggingface.co/datasets/facebook/multilingual_librispeech) | 6         | Audiobooks                            | Narrated       | CC-BY-4.0 | `"sentence"` | `"path"`     |
| [Common Voice 13](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0)       | 108       | Wikipedia text & crowd-sourced speech | Narrated       | CC0-1.0   | `"raw_text"` | `"audio_id"` |
| [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli)                               | 15        | European Parliament recordings        | Spontaneous    | CC0       | `"text"`     | `"id"`       |

To achieve *robustness* to different distributions of audio data, it is recommended to train on multiple datasets where possible.
For example, the above three datasets all have splits for the German language. Thus, if distilling a Whisper model for German,
it would be wise to use a combination of the three datasets during training, in order to cover at least three distinct domains
(audiobooks, crowd-sourced speech, parliament recordings). You may wish to use a combination of open-source datasets, or 
a combination of open-source and individually owned datasets to cover multiple distributions and domains.

## 2. Initialisation

The script [`create_student_model.py`](create_student_model.py) can be used to initialise a small student model
from a large teacher model. When initialising a student model with fewer layers than the teacher model, the student is 
initialised by copying maximally spaced layers from the teacher, as per the [DistilBart](https://arxiv.org/abs/2010.13002)
recommendations.

First, we need to create a model repository on the Hugging Face Hub. This repository will contain all the required files 
to reproduce the training run, alongside model weights, training logs and a README.md card. You can either create a model 
repository directly on the Hugging Face Hub using the link: https://huggingface.co/new. Or, via the CLI, as we'll show here.

Let's pick a name for our distilled model: `distil-whisper-large-v2-hi`. We can run the following command to create a repository under this name:

```bash
huggingface-cli repo create distil-whisper-large-v2-hi
```

We can now see the model on the Hub, e.g. under https://huggingface.co/sanchit-gandhi/distil-whisper-large-v2-hi

Let's clone the repository so that we can place our training script and model weights inside:

```bash
git lfs install
git clone https://huggingface.co/sanchit-gandhi/distil-whisper-large-v2-hi
```

Be sure to change the repo address to https://huggingface.co/<your-user-name>/<your-repo-name>.

We can now copy the relevant training scrips to the repository:
```bash
cd distil-whisper-large-v2-hi

cp ../distil-whisper/training/create_student_model.py .
cp ../distil-whisper/training/run_distillation.py .
```

The following command demonstrates how to initialise a student model from the Whisper [large-v2](https://huggingface.co/openai/whisper-large-v2) 
checkpoint, with all 32 encoder layer and 2 decoder layers. The 2 student decoder layers are copied from teacher layers 
1 and 32 respectively, as the maximally spaced layers:

```bash
#!/usr/bin/env bash

python create_student_model.py \
  --teacher_checkpoint "openai/whisper-large-v2" \
  --encoder_layers 32 \
  --decoder_layers 2 \
  --save_dir "./distil-large-v2-init"
```

The initialised model will be saved to the sub-directory `distil-large-v2-init` in our model repository. 

## 3. Training

The script [`run_distillation.py`](run_distillation.py) is an end-to-end script for loading multiple
datasets, a student model, a teacher model, and performing teacher-student distillation. It uses the loss formulation
from the [Distil-Whisper paper](https://arxiv.org/abs/2311.00430), which is a weighted sum of the cross-entropy and 
KL-divergence loss terms.

The following command takes the Common Voice dataset that was pseudo-labelled in the first stage and trains the 
2-layer decoder model intialised in the previous step. Note that multiple training datasets and splits can be loaded 
by separating the dataset arguments by `+` symbols. Thus, the script generalises to any number of training datasets.

In this example, we will combine the train and validation splits to give our training set, and evaluate on the test split 
only. This is purely to demonstrate how to combine multiple pseudo-labelled datasets for training, rather than recommended 
advice for defining train/validation splits. We advise that you train on the train splits of your dataset, evaluate and 
tune hyper-parameters on the validation split, and only test the final checkpoint on the test split.

```bash
#!/usr/bin/env bash

accelerate launch run_distillation.py \
  --model_name_or_path "./distil-large-v2-init" \
  --teacher_model_name_or_path "openai/whisper-large-v2" \
  --train_dataset_name "../common_voice_13_0_hi_pseudo_labelled+../common_voice_13_0_hi_pseudo_labelled" \
  --train_dataset_config_name "hi+hi" \
  --train_split_name "train+validation" \
  --text_column_name "sentence+sentence" \
  --train_dataset_samples "10+5" \
  --eval_dataset_name "../common_voice_13_0_hi_pseudo_labelled" \
  --eval_dataset_config_name "hi" \
  --eval_split_name "test" \
  --eval_text_column_name "sentence" \
  --eval_steps 1000 \
  --save_steps 1000 \
  --warmup_steps 50 \
  --learning_rate 0.0001 \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --save_total_limit 1 \
  --max_steps 5000 \
  --wer_threshold 10 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --dataloader_num_workers 16 \
  --preprocessing_num_workers 16 \
  --dtype "bfloat16" \
  --output_dir "./" \
  --do_train \
  --do_eval \
  --gradient_checkpointing \
  --overwrite_output_dir \
  --predict_with_generate \
  --freeze_encoder \
  --streaming False \
  --push_to_hub

```

The above training script will take approximately TODO hours to complete on an 80 GB A100 GPU and yield a final WER of TODO%.

Training logs will be reported to TensorBoard and WandB, provided the relevant packages are available. An example of a 
saved checkpoint pushed to the Hugging Face Hub can be found here: [TODO](TODO).

There are a few noteworthy arguments that can be configured to give optimal training performance:
* `train_dataset_samples`: defines the number of training samples in each dataset. Used to calculate the sampling probabilities in the dataloader. A good starting point is setting the samples to the number of hours of audio data in each split. A more refined strategy is setting it to the number of training samples in each split, however this might require downloading the dataset offline to compute these statistics.
* `wer_threshold`: sets the WER threshold between the normalised pseudo-labels and normalised ground truth labels. Any samples with WER > `wer_threshold` are discarded from the training data. This is beneficial to avoid training the student model on pseudo-labels where Whisper hallucinated or got the predictions grossly wrong.
* `freeze_encoder`: whether to freeze the entire encoder of the student model during training. Beneficial when the student encoder is copied exactly from the teacher encoder. In this case, the encoder hidden-states from the teacher model are re-used for the student model. Stopping the gradient computation through the encoder and sharing the encoder hidden-states provides a significant memory saving, and can enable up to 2x batch sizes. 
* `dtype`: data type (dtype) in which the model computation should be performed. Note that this only controls the dtype of the computations (forward and backward pass), and not the dtype of the parameters or optimiser states.
* `lr_scheduler_stype`: defines the learning rate schedule, one of `constant_with_warmup` or `linear`. When experimenting with a training set-up or training for very few steps (< 5k), using `constant_with_warmup` is typically beneficial, since the learning rate remains high over the short training run. When performing long training runs (> 5k), using a `linear` schedule generally results in superior downstream performance of the distilled model

