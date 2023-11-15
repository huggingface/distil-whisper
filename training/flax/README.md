## Reproducing Distil-Whisper

This sub-folder contains all the training and inference scripts to reproduce the Distil-Whisper project. Distil-Whisper
is written in JAX to leverage the fast training and inference speed offered by TPU v4 hardware. However, it also works
efficiently on GPU hardware without any additional code changes.

Reproducing the Distil-Whisper project requires four stages to be completed in successive order:

1. [Pseudo-labelling](#pseudo-labelling)
2. [Initialisation](#initialisation)
3. [Training](#training)
4. [Evaluation](#evaluation)

This README is partitioned according to the four stages. Each section provides a minimal example for running the
scripts used in the project. The final scripts used to train the model are referenced in-line.

It is worth noting that the experiments performed to date have been on English ASR only. We will shortly publish a Colab 
notebook that runs through these distillation steps sequentially and provides a multilingual example, facilitating anyone
to run Whisper distillation on a language of their choice.

## Requirements

Distil-Whisper is written in Python, JAX and Flax, and heavily leverages the Flax Whisper implementation in 
[🤗 Transformers](https://github.com/huggingface/transformers). The instructions for installing the package are as follows:
1. Install JAX from the [official instructions](https://github.com/google/jax#installation), ensuring you install the correct version for your hardware (GPU or TPU).
2. Install the `distil_whisper` package by cloning the repository and performing an editable installation:

```bash
git clone https://github.com/huggingface/distil-whisper.git
cd distil-whisper/training
pip install -e .
```

## Pseudo-Labelling

Pseudo-labelling is the process of generating target text predictions for the input audio data using the teacher model.
The generated text labels then replace the ground truth text labels when performing distillation. The rationale for 
using pseudo-labels instead of ground truth labels is to circumvent the issue of inconsistent transcription formatting 
across datasets.

The python script [`run_pseudo_labelling.py`](run_pseudo_labelling.py) is a flexible inference script that can be used
to generate pseudo-labels under a range of settings, including using both greedy and beam-search. It is also compatible
with [🤗 Datasets](https://github.com/huggingface/datasets) *streaming mode*, allowing users to load massive audio
datasets with **no disk space requirements**. For more information on streaming mode, the reader is referred to the 
blog post: [A Complete Guide to Audio Datasets](https://huggingface.co/blog/audio-datasets#streaming-mode-the-silver-bullet).

The following script demonstrates how to pseudo-label the [LibriSpeech 960h](https://huggingface.co/datasets/librispeech_asr)
dataset with greedy sampling and streaming mode: 

```bash
#!/usr/bin/env bash

python run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-large-v2" \
  --dataset_name "librispeech_asr" \
  --dataset_config_name "all" \
  --data_split_name "train.clean.100+train.clean.360+train.other.500" \
  --text_column_name "text" \
  --output_dir "./transcriptions" \
  --per_device_eval_batch_size 16 \
  --max_label_length 256 \
  --dtype "bfloat16" \
  --report_to "wandb" \
  --dataloader_num_workers 16 \
  --streaming \
  --push_to_hub \
  --generation_num_beams 1  # for greedy, set >1 for beam

```

The script will save the generated pseudo-labels alongside the file ids to the output directory `output_dir`. Adding the
`--push_to_hub` argument uploads the generated pseudo-labels to the Hugging Face Hub on save.

The directory [`pseudo_labelling_scripts`](pseudo_labelling_scripts) contains a collection of bash scripts for 
pseudo-labelling all 10 audio datasets used in the project. The datasets with the Whisper generated transcriptions
can be found on the Hugging Face Hub under the [Distil Whisper organisation](https://huggingface.co/datasets?sort=trending&search=distil-whisper%2F).
They can be re-used should you wish to bypass the data labelling stage of the reproduction.

<!--- TODO(SG): Combine PS with source audio to create dataset --->

## Initialisation

The script [`create_student_model.py`](create_student_model.py) can be used to initialise a small student model
from a large teacher model. When initialising a student model with fewer layers than the teacher model, the student is 
initialised by copying maximally spaced layers from the teacher, as per the [DistilBart](https://arxiv.org/abs/2010.13002)
recommendations.

The following command demonstrates how to initialise a student model from the [large-v2](https://huggingface.co/openai/whisper-large-v2) 
checkpoint, with all 32 encoder layer and 2 decoder layers. The 2 student decoder layers are copied from teacher layers 
1 and 32 respectively, as the maximally spaced layers.

```bash
#!/usr/bin/env bash

python create_student_model.py \
  --teacher_checkpoint "openai/whisper-large-v2" \
  --encoder_layers 32 \
  --decoder_layers 2 \
  --save_dir "./large-32-2" \
  --push_to_hub
```


## Training

The script [`run_distillation.py`](run_distillation.py) is an end-to-end script for loading multiple
datasets, a student model, a teacher model, and performing teacher-student distillation. It uses the loss formulation
from [DistilBart](https://arxiv.org/abs/2010.13002), which is a combination of a cross-entropy, KL-divergence and 
mean-square error (MSE) loss:

https://github.com/huggingface/distil-whisper/blob/4dd831543e6c40b1159f1ec951db7f4fe0e86850/run_distillation.py#L1725

The weight assigned to the MSE loss is configurable. The others are fixed to the values from the DistilBART paper.

The following command takes the LibriSpeech 960h dataset that was pseudo-labelled in the first stage and trains the 
2-layer decoder model intialised in the previous step. Note that multiple training datasets and splits can be loaded 
by separating the dataset arguments by `+` symbols. Thus, the script generalises to any number of training datasets.

```bash
#!/usr/bin/env bash

python3 run_distillation.py \
  --model_name_or_path "./large-32-2" \
  --teacher_model_name_or_path "openai/whisper-large-v2" \
  --train_dataset_name "librispeech_asr+librispeech_asr+librispeech_asr" \
  --train_dataset_config_name "all+all+all" \
  --train_split_name "train.clean.100+train.clean.360+train.other.500" \
  --train_dataset_samples "100+360+500" \
  --eval_dataset_name "librispeech_asr" \
  --eval_dataset_config_name "all" \
  --eval_split_name "validation.clean" \
  --eval_steps 5000 \
  --save_steps 5000 \
  --warmup_steps 500 \
  --learning_rate 0.0001 \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --save_total_limit 1 \
  --max_steps 20000 \
  --wer_threshold 10 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --dataloader_num_workers 16 \
  --dtype "bfloat16" \
  --output_dir "./" \
  --do_train \
  --do_eval \
  --use_scan \
  --gradient_checkpointing \
  --overwrite_output_dir \
  --predict_with_generate \
  --freeze_encoder \
  --streaming \
  --use_auth_token \
  --push_to_hub

```

The above training script will take approximately 20 hours to complete on a TPU v4-8 and yield a final WER of 2.3%.

Training logs will be reported to TensorBoard and WandB, provided the relevant packages are available. An example of a 
saved checkpoint pushed to the Hugging Face Hub can be found here: [large-32-2](https://huggingface.co/distil-whisper/large-32-2).

There are a few noteworthy arguments that can be configured to give optimal training performance:
* `train_dataset_samples`: defines the number of training samples in each dataset. Used to calculate the sampling probabilities in the dataloader. A good starting point is setting the samples to the number of hours of audio data in each split. A more refined strategy is setting it to the number of training samples in each split, however this might require downloading the dataset offline to compute these statistics.
* `wer_threshold`: sets the WER threshold between the normalised pseudo-labels and normalised ground truth labels. Any samples with WER > `wer_threshold` are discarded from the training data. This is beneficial to avoid training the student model on pseudo-labels where Whisper hallucinated or got the predictions grossly wrong.
* `freeze_encoder`: whether to freeze the entire encoder of the student model during training. Beneficial when the student encoder is copied exactly from the teacher encoder. In this case, the encoder hidden-states from the teacher model are re-used for the student model. Stopping the gradient computation through the encoder and sharing the encoder hidden-states provides a significant memory saving, and can enable up to 2x batch sizes. 
* `dtype`: data type (dtype) in which the model computation should be performed. Note that this only controls the dtype of the computations (forward and backward pass), and not the dtype of the parameters or optimiser states.

The Distil Whisper project extends the above script to train on a combined dataset formed from 12 open-source ASR datasets,
totalling 22k hours and over 50k speakers. Template scripts to run training on this composite dataset can be found 
in the directory [`distillation_scripts`](distillation_scripts).

## Evaluation

There are two types of evaluation performed in Distil-Whisper:
1. Short form: evaluation on audio samples less than 30s in duration. Examples include typical ASR test sets, such as the LibriSpeech validation set.
2. Long form: evaluation on audio samples longer than 30s in duration. Examples include entire TED talks or earnings calls.

Both forms of evaluation are performed using the *word-error rate (WER)* metric.

### Short Form

The script [`run_eval.py`](run_eval.py) can be used to evaluate a trained student model over multiple validation sets.
The following example demonstrates how to evaluate the student model trained in the previous step on the LibriSpeech
`validation.clean` and `validation.other` dev sets. Again, it leverages streaming mode to bypass the need to download
the data offline:

```bash
#!/usr/bin/env bash

python run_eval.py \
  --model_name_or_path "./large-32-2" \
  --dataset_name "librispeech_asr+librispeech_asr" \
  --dataset_config_name "all+all" \
  --dataset_split_name "validation.clean+validation.other" \
  --output_dir "./large-32-2" \
  --per_device_eval_batch_size 64 \
  --dtype "bfloat16" \
  --dataloader_num_workers 16 \
  --report_to "wandb" \
  --streaming \
  --predict_with_generate

```

### Long Form

Long form evaluation runs on the premise that a single long audio file can be *chunked* into smaller segments and 
inferred in parallel. The resulting transcriptions are then joined at the boundaries to give the final text prediction. 
A small overlap (or *stride*) is used between adjacent segments to ensure a continuous transcription across chunks.

This style of chunked inference is performed using the [`FlaxWhisperPipeline`](https://github.com/huggingface/distil-whisper/blob/6426022e3b3a0a498b4150a636b54e2e3898bf1a/distil_whisper/pipeline.py#L61)
class, which is heavily inspired from [Whisper JAX](https://github.com/sanchit-gandhi/whisper-jax/tree/main#pipeline-usage).

The script [`run_long_form_transcription.py`](run_long_form_transcription.py) can be used to evaluate the trained 
student model on an arbitrary number of long-form evaluation sets. The following script demonstrates how to evaluate
the example student model on two such test sets, [Earnings 21](https://huggingface.co/datasets/distil-whisper/earnings21) 
and [Earnings 22](https://huggingface.co/datasets/distil-whisper/earnings22):

```bash
#!/usr/bin/env bash

python run_long_form_transcription.py \
  --model_name_or_path "./large-32-2" \
  --dataset_name "distil-whisper/earnings21+distil-whisper/earnings22" \
  --dataset_config_name "default+default" \
  --dataset_split_name "test+test+test+test" \
  --text_column_name "transcription+transcription" \
  --output_dir "./large-32-2" \
  --per_device_eval_batch_size 64 \
  --chunk_length_s 15 \
  --dtype "bfloat16" \
  --report_to "wandb" \
  --streaming

```

The argument `chunk_length_s` controls the length of the chunked audio samples. It should be set to match the typical
length of audio the student model was trained on. If unsure about what value of `chunk_length_s` is optimal for your case,
it is recommended to run a *sweep* over all possible values. A template script for running a [WandB sweep](https://docs.wandb.ai/guides/sweeps) 
can be found under [`run_chunk_length_s_sweep.yaml`](long_form_transcription_scripts/run_chunk_length_s_sweep.yaml).

### 1. Pseudo Labelling

#### Greedy vs Beam

We found there to be little-to-no difference in the downstream performance of the distilled model after pseudo labelling
using either greedy or beam-search. We attribute this to the minimal difference in performance of the pre-trained Whisper
model under greedy and beam-search decoding, giving pseudo-labelled transcriptions of similar quality. We encourage
users to generate pseudo-labels using greedy decoding given it runs significantly faster. Beam search is only advised if 
the pre-trained model is hallucinating significantly on the audio inputs, in which case it helps reduce the frequency and
severity of hallucinations. If using beam search, the number of beams can be kept low: even 2 beams helps reduce the 
amount of hallucinations significantly.

#### Timestamps

Whisper is trained on a timestamp prediction task as part of the pre-training set-up. Here, a fixed proportion of the 
pre-training data includes sequence-level *timestamps* as part of the transcription labels:

```bash
<|0.00|> Hey, this is a test transcription. <|3.42|>
```

Timestamp prediction is useful for enriching the transcriptions with timing information for downstream tasks, such as
aligning the Whisper transcription with the output of a speaker diarization system, and also reduces the frequency of 
hallucinations.

The pseudo-labelling scrip [`run_pseudo_labelling.py`](run_pseudo_labelling.py) can be extended to predict timestamp
information in the audio data by appending the `--return_timestamps` flag to the launch command. The timestamped labelled
data can be passed to the training script in exactly the same way as the non-timestamped version, and the pre-processing
function will take care of encoding the timestamps and appending the required task tokens.

#### Previous Context

Whisper is also pre-trained on a prompting task, where the transcription for the preceding utterance is fed as context 
to the current one:

```bash
<|startofprev|> This is the previous context from the preceding utterance.<|startoftranscript|> And this is the current utterance.<|endoftranscript|>
```

Annotating the transcriptions with previous context labels is only possible for datasets where we have consecutive files 
and unique speaker ids, since we need to ensure segment `i` directly follows on from segment `i-1` if we use it as the 
prompt.

As per the Whisper paper, we mask out the loss over the previous context tokens. At inference time, we can replace the 
previous context with a “prompt” to encourage the model to generate text in the style of the prompt (i.e. for specific 
named entities, or styles of transcription)

## Acknowledgements

* 🤗 Hugging Face Transformers for the base Whisper implementation
* Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) programme for their generous provision of Cloud TPUs
