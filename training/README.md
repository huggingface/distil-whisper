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

Reproducing the Distil-Whisper project requires four stages to be completed in successive order:

1. [Pseudo-labelling](#1-pseudo-labelling)
2. [Initialisation](#2-initialisation)
3. [Training](#3-training)
4. [Evaluation](#4-evaluation)

This README is partitioned according to the four stages. Each section provides a minimal example for running the
scripts used in the project. We will use a running example of distilling the Whisper model for Hindi speech recognition
on the Common Voice dataset. Note that this dataset only contains ~20 hours of audio data. Thus, it can be run extremely
quickly, but does not provide sufficient data to achieve optimal performance. We recommend training on upwards of 1000 
hours of data should you want to match the performance of Whisper on high-resource languages.

## Requirements

The Distil-Whisper training code is written in [PyTorch](https://pytorch.org) and [Accelerate](https://huggingface.co/docs/accelerate/index). 
It heavily leverages the Whisper implementation in [🤗 Transformers](https://github.com/huggingface/transformers) for both 
training and inference.

The instructions for installing the package are as follows:
1. Install PyTorch from the [official instructions](https://pytorch.org/get-started/locally/), ensuring you install the correct version for your hardware and CUDA version.
2. Fork the `distil-whisper` repository by clicking on the [fork](https://github.com/huggingface/distil-whisper/fork) button on the reopsitory's page
3. Clone the `distil-whisper` repository and add the base repository as a remote. This will allow you to "pull" any upstream changes that are made to the base repository:

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

To confirm that you have a working environment, first accept the terms of use of the Common Voice 13 dataset on the Hub: https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0 

You can run the following code cell to stream one sample of data from the Common Voice dataset, and check that you can 
perform inference using the "tiny" Whisper model:

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny", low_cpu_mem_usage=True)
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

model.to("cuda")

common_voice = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="validation", streaming=True)
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

inputs = processor(next(iter(common_voice))["audio"]["array"], sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features

generated_ids = model.generate(input_features.to("cuda"), max_new_tokens=128)
pred_text = processor.decode(generated_ids[0], skip_special_tokens=True)

print("Pred text:", pred_text)
print("Environment set up successful?", generated_ids.shape[-1] == 19)
```

## 1. Pseudo-Labelling

The python script [`run_pseudo_labelling.py`](run_pseudo_labelling.py) is a flexible inference script that can be used
to generate pseudo-labels under a range of settings, including using both greedy and beam-search. It is also compatible
with [🤗 Datasets](https://github.com/huggingface/datasets) *streaming mode*, allowing users to load massive audio
datasets with **no disk space requirements**. For more information on streaming mode, the reader is referred to the 
blog post: [A Complete Guide to Audio Datasets](https://huggingface.co/blog/audio-datasets#streaming-mode-the-silver-bullet).

> As of the latest Distil-Whisper release, [`distil-large-v3`](https://huggingface.co/distil-whisper/distil-large-v3), this
pseudo-labelling script also performs the added operation of concatenating (or packing) the audio inputs to 30-seconds. 
Not only does this lead to a WER improvement when using sequential long-form decoding algorithm, but concatenating audios 
to 30-seconds also improves the throughput during training, since the amount of zero-padding on the audio inputs is minimised.

The following script demonstrates how to pseudo-label the Hindi split of the Common Voice 16.1 dataset with greedy sampling:

```bash
#!/usr/bin/env bash

accelerate launch run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-large-v3" \
  --dataset_name "mozilla-foundation/common_voice_16_1" \
  --dataset_config_name "hi" \
  --dataset_split_name "train+validation+test" \
  --text_column_name "sentence" \
  --id_column_name "path" \
  --output_dir "./common_voice_16_1_hi_pseudo_labelled" \
  --wandb_project "distil-whisper-labelling" \
  --per_device_eval_batch_size 64 \
  --dtype "bfloat16" \
  --attn_implementation "sdpa" \
  --logging_steps 500 \
  --max_label_length 256 \
  --concatenate_audio \
  --preprocessing_batch_size 500 \
  --preprocessing_num_workers 8 \
  --dataloader_num_workers 8 \
  --report_to "wandb" \
  --language "hi" \
  --task "transcribe" \
  --return_timestamps \
  --streaming False \
  --generation_num_beams 1 \
  --push_to_hub
```

On an 80 GB A100 GPU, the following script takes approximately 5 minutes to concatenate and pre-process the 20 hours of 
audio data, and a further 10 minutes to transcribe the pseudo-labels. The pseudo-labelled dataset corresponding to this
script is available on the Hugging Face Hub under [sanchit-gandhi/common_voice_16_1_hi_pseudo_labelled](https://huggingface.co/datasets/sanchit-gandhi/common_voice_16_1_hi_pseudo_labelled).
The WER of the pre-trained Whisper large-v3 model is 17.2% on the test split. We will compare the performance of our distilled model against this number.

There are two noteworthy arguments that configure the dataset concatenation (or packing) process:
1. `concatenate_audio`: whether or not to concatenate (or pack) the audios to 30-second chunks. The latest Distil-Whisper model, [`distil-large-v3`](https://huggingface.co/distil-whisper/distil-large-v3#differences-with-distil-large-v2), highlights the WER improvements obtained using the sequential long-form decoding algorithm when concatenated audios are used. Concatenating audios to 30-seconds also improves the throughput during training, since the amount of zero-padding on the audio inputs is minimised. Hence, it is highly recommended to set `--concatenate_audio=True`.
2. `preprocessing_batch_size`: the batch size to use when concatenating (or packing) the audios. Using a larger batch size results in a greater portion of audio samples being packed to 30-seconds, at the expense of higher memory consumption. If you exceed your system's RAM when performing the concatenation operation, reduce the `preprocessing_batch_size` by a factor of 2 to 250 or even 125.
3. `preprocessing_num_workers`: the number of multiprocessing workers to use when concatenating the audios. Using more workers will result in faster pre-processing, at the expense of higher memory consumption. Ensure you do not exceed the maximum number of CPUs on your device.

In addition, the following arguments configure the inference of the Whisper model:
1. `language`: explicitly setting the language token during inference substantially improves the generation performance of the Whisper model, since the model is forced always to predict in the given language. We recommend you set the language to the language you wish to distil the Whisper model on. The only exception is when distilling an English-only model (i.e. where the model id is appended with an `.en`, e.g. `small.en`), the language argument should be set to None, since there is no language token used during training/inference.
2. `return_timestamps`: whether or not to predict timestamps in the pseudo-labels. Timestamp prediction is required should you want your distilled model to be able to predict timestamps at inference time (e.g. for the original OpenAI long-form transcription algorithm). However, the pseudo-labels are marginally less accurate than not using timestamps. We recommend pseudo-labelling **with** timestamps to ensure the distilled model is as general as possible.
3. `attn_implementation`: which attention implementation to use for inference. Set to `sdpa` for [PyTorch SDPA](https://huggingface.co/docs/transformers/v4.35.2/en/perf_infer_gpu_one#bettertransformer), or `flash_attn_2` if your hardware supports Flash Attention 2 and you have the [package installed](https://github.com/Dao-AILab/flash-attention).
4. `streaming`: whether or not to use Datasets' streaming mode. If enabled, the audio data will be streamed from the Hugging Face Hub with no disk space requirements. However, the user is then responsible for adding the pseudo-labels to the dataset script in a follow-up step (see [Using Streaming Mode](#TODO)). If set to `False`, the audio data will be downloaded and pre-processed offline. At the end of pseudo-labelling, the pseudo-labels will be automatically appended to the original dataset, meaning the dataset is ready to be used for the subsequent training step without any additional steps.
5. `generation_num_beams`: how many beams to use while decoding. In practice, we found the distilled model to perform comparably when the data was pseudo-labelled with `generation_num_beams=1` (greedy) or `generation_num_beams>1` (beam). This is likely because the WER filter compensates for the lower quality pseudo-labels obtained using greedy search. However, using `generation_num_beams=1` gives substantially faster inference time for the pseudo-labelling step, and so we recommend this configuration.

Should you have your own audio dataset, you can first [convert it](https://huggingface.co/docs/datasets/audio_dataset) to 
Hugging Face Datasets format and push it to the Hugging Face Hub. You can then pseudo-label it using the script above, 
replacing the `--dataset_name` with the name of your dataset on the Hub.

Otherwise, you may wish to use an open-source dataset already available on the Hugging Face Hub. We provide a summary of 
the three most popular multilingual datasets in the table below. For more details, refer to the blog post: [A Complete Guide to Audio Datasets](https://huggingface.co/blog/audio-datasets#multilingual-speech-recognition).

| Dataset                                                                                       | Languages | Domain                                | Speaking Style | License   | Text Column  | ID Column    |
|-----------------------------------------------------------------------------------------------|-----------|---------------------------------------|----------------|-----------|--------------|--------------|
| [Multilingual LibriSpeech](https://huggingface.co/datasets/facebook/multilingual_librispeech) | 6         | Audiobooks                            | Narrated       | CC-BY-4.0 | `"sentence"` | `"path"`     |
| [Common Voice 16](https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1)       | 120       | Wikipedia text & crowd-sourced speech | Narrated       | CC0-1.0   | `"raw_text"` | `"audio_id"` |
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

Let's pick a name for our distilled model: `distil-whisper-large-v3-hi`. We can run the following command to create a repository under this name:

```bash
huggingface-cli repo create distil-whisper-large-v3-hi
```

We can now see the model on the Hub, e.g. under https://huggingface.co/sanchit-gandhi/distil-whisper-large-v3-hi

Let's clone the repository so that we can place our training script and model weights inside:

```bash
git lfs install
git clone https://huggingface.co/sanchit-gandhi/distil-whisper-large-v3-hi
```

Be sure to change the repo address to `https://huggingface.co/<your-user-name>/<your-repo-name>`

We can now copy the relevant training scrips to the repository:
```bash
cd distil-whisper-large-v3-hi

cp ../distil-whisper/training/create_student_model.py .
cp ../distil-whisper/training/run_distillation.py .
```

The following command demonstrates how to initialise a student model from the Whisper [large-v3](https://huggingface.co/openai/whisper-large-v3) 
checkpoint, with all 32 encoder layer and 2 decoder layers. The 2 student decoder layers are copied from teacher layers 
1 and 32 respectively, as the maximally spaced layers:

```bash
#!/usr/bin/env bash

python create_student_model.py \
  --teacher_checkpoint "openai/whisper-large-v3" \
  --encoder_layers 32 \
  --decoder_layers 2 \
  --save_dir "./distil-large-v3-init"
```

The initialised model will be saved to the sub-directory `distil-large-v3-init` in our model repository. 

## 3. Training

The script [`run_distillation.py`](run_distillation.py) is an end-to-end script for loading multiple
datasets, a student model, a teacher model, and performing teacher-student distillation. It uses the loss formulation
from the [Distil-Whisper paper](https://arxiv.org/abs/2311.00430), which is a weighted sum of the cross-entropy and 
KL-divergence loss terms.

The following command takes the Common Voice dataset that was pseudo-labelled in the first stage and trains the 
2-layer decoder model intialised in the previous step. We pass the local path to the pseudo-labelled Common Voice dataset
(`../common_voice_16_1_hi_pseudo_labelled`), which you can change to the path where your local pseudo-labelled dataset is 
saved.

In this example, we will combine the train and validation splits to give our training set, and evaluate on the test split 
only. This is purely to demonstrate how to combine multiple pseudo-labelled datasets for training, rather than recommended 
advice for defining train/validation splits. We advise that you train on the train splits of your dataset, evaluate and 
tune hyper-parameters on the validation split, and only test the final checkpoint on the test split. Note how multiple 
training datasets and splits can be loaded by separating the dataset arguments by `+` symbols. Thus, the script generalises 
to any number of training datasets.

```bash
#!/usr/bin/env bash

accelerate launch run_distillation.py \
  --model_name_or_path "./distil-large-v3-init" \
  --teacher_model_name_or_path "openai/whisper-large-v3" \
  --train_dataset_name "./common_voice_16_1_hi_pseudo_labelled+./common_voice_16_1_hi_pseudo_labelled" \
  --train_split_name "train+validation" \
  --text_column_name "sentence+sentence" \
  --train_dataset_samples "7+4" \
  --eval_dataset_name "./common_voice_16_1_hi_pseudo_labelled" \
  --eval_split_name "test" \
  --eval_text_column_name "sentence" \
  --eval_steps 1000 \
  --save_steps 1000 \
  --warmup_steps 50 \
  --learning_rate 0.0001 \
  --lr_scheduler_type "constant_with_warmup" \
  --timestamp_probability 0.2 \
  --condition_on_prev_probability 0.2 \
  --language "hi" \
  --task "transcribe" \
  --logging_steps 25 \
  --save_total_limit 1 \
  --max_steps 5000 \
  --wer_threshold 20 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --dataloader_num_workers 8 \
  --preprocessing_num_workers 8 \
  --ddp_timeout 7200 \
  --dtype "bfloat16" \
  --attn_implementation "sdpa" \
  --output_dir "./" \
  --do_train \
  --do_eval \
  --gradient_checkpointing \
  --overwrite_output_dir \
  --predict_with_generate \
  --freeze_encoder \
  --freeze_embed_positions \
  --streaming False \
  --push_to_hub

```

The above training script will take approximately 1 hour to complete on an 80 GB A100 GPU and yield a final WER of TODO%.
This is reasonable for 1000 training steps and just 15 hours of un-filtered training data, but 12% higher than the error rate of the 
pre-trained model. As mentioned above, using upwards of 1000 hours of data and training for 10k steps will likely yield
more competitive performance. For the [Distil-Whisper paper](https://arxiv.org/abs/2311.00430), we trained on 21k hours
of audio data for 80k steps. We found that upwards of 13k hours of audio data was required to reach convergence on English 
ASR (see Section 9.2 of the [paper](https://arxiv.org/abs/2311.00430)), so the more data you have, the better!

Scaling to multiple GPUs using [distributed data parallelism (DDP)](https://pytorch.org/tutorials/beginner/ddp_series_theory.html)
is trivial: simply run `accelerate config` and select the multi-GPU option, specifying the IDs of the GPUs you wish to use. The 
above script can then be run using DDP with no code changes. 

Training logs will be reported to TensorBoard and WandB, provided the relevant packages are available. An example of a 
saved checkpoint pushed to the Hugging Face Hub can be found here: [sanchit-gandhi/distil-whisper-large-v3-hi](https://huggingface.co/sanchit-gandhi/distil-whisper-large-v3-hi).

There are a few noteworthy data arguments:
1. `train_dataset_samples`: defines the number of training samples in each dataset. Used to calculate the sampling probabilities in the dataloader. A good starting point is setting the samples to the number of hours of audio data in each split. A more refined strategy is setting it to the number of training samples in each split, however this might require downloading the dataset offline to compute these statistics.
2. `wer_threshold`: sets the WER threshold between the normalised pseudo-labels and normalised ground truth labels. Any samples with WER > `wer_threshold` are discarded from the training data. This is beneficial to avoid training the student model on pseudo-labels where Whisper hallucinated or got the predictions grossly wrong. In our English distillation experiments, we found a WER threshold of 10% provides the optimal trade-off between ensuring high-quality transcriptions, and not filtering unnecessary amounts of training data. For multilingual distillation, the threshold should be set in accordance with the WER achieved by the pre-trained model on the test set.
3. `streaming`: whether or not to use Datasets' streaming mode. Recommended for large datasets, where the audio data can be streamed from the Hugging Face Hub with no disk space requirements.
4. `timestamp_probability`: the per-sample probability for retaining timestamp tokens in the labels (should they contain them). Retaining some portion of timestamp tokens in the training data is required to ensure the distilled model can predict timestamps at inference time. In our experiments, we found that training on timestamps with high-probability hurts the distilled model's transcription performance. Thus, we recommend setting this to a value below 0.5. Typically, a value of 0.2 works well, giving good transcription and timestamp performance.
5. `condition_on_prev_probability`: the per-sample probability for conditioning on previous labels. Conditioning on previous tokens is required to ensure the distilled model can be used with the "sequential" long-form transcription algorithm at inference time. We did not experiment with this parameter, but found values around 0.2 to provide adequate performance. OpenAI pre-trained Whisper on with a 50% probability for conditioning on previous tokens. Thus, you might wish to try higher values.

As well as a few noteworthy model arguments that can be configured to give optimal training performance:
1. `freeze_encoder`: whether to freeze the entire encoder of the student model during training. Beneficial when the student encoder is copied exactly from the teacher encoder. In this case, the encoder hidden-states from the teacher model are re-used for the student model. Stopping the gradient computation through the encoder and sharing the encoder hidden-states provides a significant memory saving, and can enable up to 2x batch sizes. 
2. `freeze_embed_positions`: whether to freeze the student model's decoder positional embeddings. Using the same embed positions as the teacher model, which is designed to handle context lengths up to 448 tokens, helps the student model retain its input id representation up to the full max input length. 
3. `dtype`: data type (dtype) in which the model computation should be performed. Note that this only controls the dtype of the computations (forward and backward pass), and not the dtype of the parameters or optimiser states.

And finally, a few noteworthy training arguments:
1. `max_steps`: defines the total number of optimisation steps (forward + backward pass) during training. To reach convergence, you should use a dataset of at least 1k hours and train for a minimum of 50k steps.
2. `lr_scheduler_stype`: defines the learning rate schedule, one of `constant_with_warmup` or `linear`. When experimenting with a training set-up or training for very few steps (< 5k), using `constant_with_warmup` is typically beneficial, since the learning rate remains high over the short training run. When performing long training runs (> 5k), using a `linear` schedule generally results in superior downstream performance of the distilled model.

TODO:
- [ ] Template for model cards

## 4. Evaluation

There are four types of evaluation performed in Distil-Whisper:
1. Short form: evaluation on audio samples less than 30s in duration. Examples include typical ASR test sets, such as the LibriSpeech validation set.
2. Sequential long form: evaluation on audio samples longer than 30s in duration using the original "sequential" long-form algorithm. Examples include entire TED talks or earnings calls.
3. Chunked long form: evaluation on audio samples longer than 30s in duration using the Transformers "chunked" long-form algorithm.
4. Speculative decoding: evaluation on audio samples less than 30s in duration, where a faster, distilled model is used as the assistant to a slower, teacher model. 

All four forms of evaluation are performed using the script [`run_eval.py`](run_eval.py). Unlike the pseudo-labelling
and training scripts, the evaluation script assumes that only one GPU accelerator is used. We can copy the corresponding 
evaluation script to the model repository using the following command:

```bash
cp ../distil-whisper/training/run_eval.py .
```

Models are assessed jointly using:
1. The *word-error rate (WER)* metric: measures the numer of substitution, deletion and insertion errors relative to the total number of words. A lower WER indicates a more accurate model.
2. The *inverse real-time factor (RTFx)* metric: measures the ratio of `audio input time : model compute time`. A higher RTFx indicates a faster model.

In all cases, it is particularly important to evaluate the final model on data that is *out-of-distribution (OOD)* with 
the training data. Evaluating on OOD data provides insight as to how well the distilled model is likely to generalise to 
different audio distributions at inference time. In our example, the Common Voice test set is *in-distribution (ID)* 
with our training data, since it is taken from the same distribution as the Common Voice training set. Whereas the FLEURS 
test set is OOD, since it is not used as part of the training set.

### Short Form

The script [`run_eval.py`](run_eval.py) can be used to evaluate a trained student model over multiple short-form 
validation sets. The following example demonstrates how to evaluate the student model trained in the previous step on 
the Common Voice `test` set (ID) and also the FLEURS `test` set (OOD). Again, it leverages streaming mode to bypass 
the need to download the data offline:

```bash
#!/usr/bin/env bash

python run_eval.py \
  --model_name_or_path "./" \
  --dataset_name "../common_voice_16_1_hi_pseudo_labelled+google/fleurs" \
  --dataset_config_name "default+hi_in" \
  --dataset_split_name "test+test" \
  --text_column_name "sentence+transcription" \
  --batch_size 16 \
  --dtype "bfloat16" \
  --generation_max_length 256 \
  --language "hi" \
  --attn_implementation "sdpa" \
  --streaming

```

The student model achieves an average WER of TODO% with an RTFx of TODO for a batch size of 16. We can easily adapt the above
script to evaluate the teacher model, simply by switching the `model_name_or_path` to `openai/whisper-large-v3`, which 
achieves an average WER of TODO% with an RTFx of TODO. Therefore, for a batch size of 16, the student model is a factor of TODO
times faster than the teacher. The WER gap can be closed by training on more data (at least 1k hours) for more training
steps (at least 50k).

### Sequential Long Form

The original Whisper paper presents a long-form transcription algorithm that sequentially transcribes 30-second segments 
of audio and shifts the sliding window according to the timestamps predicted by the model. This style of sequential 
inference is performed directly using the [`.generate`](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate) 
method in Transformers.

The script [`run_eval.py`](run_eval.py) can be used to evaluate the trained student model on an arbitrary number of 
long-form evaluation sets using the sequential algorithm. Since we don't have a long-form validation set for Hindi to hand, 
in this example we'll evaluate the official Distil-Whisper model [`distil-large-v3`](https://huggingface.co/distil-whisper/distil-large-v3) 
on the TED-LIUM validation set:

```bash
#!/usr/bin/env bash

accelerate launch run_eval.py \
  --model_name_or_path "distil-whisper/distil-large-v3" \
  --dataset_name "distil-whisper/tedlium-long-form" \
  --dataset_config_name "all" \
  --dataset_split_name "validation" \
  --text_column_name "text" \
  --batch_size 16 \
  --dtype "bfloat16" \
  --generation_max_length 256 \
  --language "en" \
  --attn_implementation "sdpa" \
  --streaming

```

### Chunked Long Form

Chunked long form evaluation runs on the premise that a single long audio file can be *chunked* into smaller segments and 
inferred in parallel. The resulting transcriptions are then joined at the boundaries to give the final text prediction. 
A small overlap (or *stride*) is used between adjacent segments to ensure a continuous transcription across chunks.

This style of chunked inference is performed using the [`pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines)
class, which provides a wrapper around the [`.generate`](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate) 
function for long-form inference.

The script [`run_eval.py`](run_eval.py) can be used to evaluate the trained student model on an arbitrary number of 
long-form evaluation sets using the pipeline class. Again, in this example we'll evaluate distil-large-v3 on the 
TED-LIUM validation set:

```bash
#!/usr/bin/env bash

python run_eval.py \
  --model_name_or_path "openai/whisper-large-v3" \
  --dataset_name "distil-whisper/tedlium-long-form" \
  --dataset_config_name "all" \
  --dataset_split_name "validation" \
  --text_column_name "text" \
  --use_pipeline \
  --chunk_length_s 25.0 \
  --language "en" \
  --return_timestamps \
  --dtype "bfloat16" \
  --streaming

```

The argument `chunk_length_s` controls the length of the chunked audio samples. It should be set to match the typical
length of audio the student model was trained on. If unsure about what value of `chunk_length_s` is optimal for your case,
it is recommended to run a *sweep* over all possible values. A template script for running a [WandB sweep](https://docs.wandb.ai/guides/sweeps) 
can be found under [`run_chunk_length_s_sweep.yaml`](flax/long_form_transcription_scripts/run_chunk_length_s_sweep.yaml).

### Speculative Decoding

Speculative decoding, or assisted generation, relies on the premise that a faster, assistant model can be used to speed-up
the generation of a slower, assistant model. Speculative decoding mathematically ensures that exactly the same outputs as 
Whisper are obtained, while being ~2 times faster. This makes it the perfect drop-in replacement for existing Whisper 
pipelines, since exactly the same outputs are guaranteed.

Distil-Whisper checkpoints can be designed to be efficient assistant models to Whisper for speculative decoding. More precisely,
by freezing the encoder during training, the distilled model can share the same encoder weights as Whisper during inference, since
the encoder weights are un-changed. In doing so, only the distilled 2-layer decoder has to be loaded in addition to the 
original Whisper model, which is approximately an 8% increase to the total parameter count, with up to 2x faster inference 
for low batch sizes. For more details on speculative decoding, the reader is advised to refer to the following blog post:
[Speculative Decoding for 2x Faster Whisper Inference](https://huggingface.co/blog/whisper-speculative-decoding).

In the example below, we use our distilled model as an assistant to the large-v3 teacher model during inference:

```bash
#!/usr/bin/env bash

python run_eval.py \
  --model_name_or_path "openai/whisper-large-v3" \
  --assistant_model_name_or_path "./" \
  --dataset_name "../common_voice_16_1_hi_pseudo_labelled+google/fleurs" \
  --dataset_config_name "default+hi_in" \
  --dataset_split_name "test+test" \
  --text_column_name "sentence+transcription" \
  --batch_size 16 \
  --dtype "bfloat16" \
  --generation_max_length 256 \
  --language "hi" \
  --attn_implementation "sdpa" \
  --streaming

```

We see that we achieve a WER of TODO%, the same as what we obtained with the large-v3 model, but with an RTFx of TODO, 
a factor of TODO faster than using the large-v3 model alone. The RTFx value can be improved by training the student on 
more data and for more training steps, since this will improve the number of predicted tokens that match the teacher 
predictions.

## Overview of Training Methods

### 1. Fine-Tuning

For fine-tuning, we take the original Whisper checkpoint and train it on one or more datasets using the standard 
cross-entropy loss. As such, there is no involvement from the teacher checkpoint during training, and so the fine-tuned 
model is permitted to *overfit* to the distribution of the training data we provide. This makes it appealing for "low-resource" 
languages where the original Whisper model performs poorly, since we can boost the performance of the model on a single 
language by *overfitting* to that distribution of data. Note that this means the fine-tuned model is prone to loosing 
its robustness to different audio distributions, which is the trade-off with improving performance on a specified dataset.

As a rule of thumb, fine-tuning is appropriate for languages where the original Whisper model performs > 20% WER, and we 
have a relatively small quantity of training data available (< 1000 hours). With fine-tuning, we require as little as **10 hours**
of training data to significantly boost the performance of the Whisper model. For an in-depth guide to fine-tuning Whisper,
the reader is advised to refer to the blog post: [Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper).

### 2. Shrink and Fine-Tune

Shrink and fine-tune (SFT) is a knowledge distillation (KD) technique in which we first *shrink* the teacher model to a 
smaller student model by copying maximally spaced layers, and then *fine-tune* the student model on the cross-entropy loss 
as described above. Typically, we retain the full encoder from the Whisper model and only shrink the decoder. Retaining 
the entire encoder helps significantly with maintaining Whisper's robustness to different audio distributions (_c.f._ 
Section 9.3 of the [Distil-Whisper paper](https://arxiv.org/abs/2311.00430)).

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

The following table summarises the two training paradigms: fine-tuning and knowledge distillation (KD). It suggests 
minimum values for the pre-trained WER / training data to achieve reasonable performance:

| Method      | Pre-Trained WER / % | Training Data / h |
|-------------|---------------------|-------------------|
| Fine-tuning | > 20                | < 1000            |
| KD          | < 20                | > 1000            |

## Acknowledgements

* OpenAI for the Whisper [model](https://huggingface.co/openai/whisper-large-v3) and [original codebase](https://github.com/openai/whisper)
* Hugging Face 🤗 [Transformers](https://github.com/huggingface/transformers) for the Whisper model implementation
* Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) program for Cloud TPU v4s used to train the official Distil-Whisper models
* The Hugging Face 🤗 cluster for enabling experimentation with the PyTorch scripts

## Citation

If you use this code-base, please consider citing the Distil-Whisper paper:

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
