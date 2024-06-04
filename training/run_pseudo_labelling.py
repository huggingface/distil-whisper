#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pseudo-labelling audio data using the Whisper model in preparation for distillation.
"""
# You can also adapt this script for your own pseudo-labelling tasks. Pointers for this are left as comments.

import csv
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
import evaluate
import numpy as np
import torch
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datasets import (
    DatasetDict,
    IterableDatasetDict,
    load_dataset,
)
from huggingface_hub import HfFolder, create_repo, get_full_repo_name, snapshot_download, upload_folder
from torch.utils.data import DataLoader
from tqdm import tqdm
from soundfile import LibsndfileError
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.34.0.dev0")

require_version("datasets>=2.14.6", "To fix: `pip install --upgrade datasets`")

logger = get_logger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to distill from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained Whisper model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    feature_extractor_name: Optional[str] = field(
        default=None,
        metadata={"help": "feature extractor name or path if not the same as model_name"},
    )
    processor_name: Optional[str] = field(
        default=None,
        metadata={"help": "processor name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    subfolder: str = field(
        default="",
        metadata={
            "help": "In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can"
            "specify the folder name here."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "The data type (dtype) in which to load the model weights. One of `float32` (full-precision), "
                "`float16` or `bfloat16` (both half-precision)."
            )
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Which attention implementation to use in the encoder and decoder attention layers. Can be one of:\n"
                "1. `eager` or `None`: default Transformers attention implementation.\n"
                "2. `sdpa`: Flash Attention through PyTorch SDPA. Requires `torch>=2.1`. Recommended for hardware where Flash Attention 2 is not supported, e.g. Turing GPUs, (T4, RTX 2080).\n"
                "3. `flash_attn_2`: Flash Attention 2 through the Flash Attention package https://github.com/Dao-AILab/flash-attention. **Always** recommended on supported hardware (Ampere, Ada, or Hopper GPUs, e.g., A100, RTX 3090, RTX 4090, H100)."
            )
        },
    )
    attn_type: Optional[str] = field(
        default=None,
        metadata={"help": "Deprecated. Use `attn_implementation` instead."},
    )

    def __post_init__(self):
        if self.attn_type is not None and self.attn_implementation is None:
            # set attn_implementation in a backwards compatible way
            if self.attn_type == "flash_attn":
                self.attn_implementation = "sdpa"
            elif self.attn_type == "flash_attn_2":
                self.attn_implementation = "flash_attention_2"
            elif self.attn_type in [None, "eager", "sdpa", "flash_attention_2"]:
                self.attn_implementation = self.attn_type
            else:
                raise ValueError(
                    f"Argument `--attn_type` is deprecated, and set to an invalid option `{self.attn_type}`. You should omit the argument `--attn_type`, and instead set `-attention_implementation` to one of the following:\n"
                    "1. `eager` or `None`: default Transformers attention implementation.\n"
                    "2. `sdpa`: Flash Attention through PyTorch SDPA. Requires `torch>=2.1`. Recommended for hardware where Flash Attention 2 is not supported, e.g. Turing GPUs, (T4, RTX 2080).\n"
                    "3. `flash_attn_2`: Flash Attention 2 through the Flash Attention package https://github.com/Dao-AILab/flash-attention. **Always** recommended on supported hardware (Ampere, Ada, or Hopper GPUs, e.g., A100, RTX 3090, RTX 4090, H100)."
                )
            warnings.warn(
                f"Argument `--attn_type` is deprecated. Use `--attn_implementation` instead. Inferring `--attn_implementation={self.attn_implementation} from argument `--attn_type={self.attn_type}`."
            )
        elif self.attn_type is not None and self.attn_implementation is not None:
            raise ValueError(
                "`--attn_type` and `--attn_implementation` are both specified. Only the argument `--attn_implementation`."
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to cache directory for saving and loading datasets"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessing_batch_size: Optional[int] = field(
        default=500,
        metadata={"help": "The batch size to use for the dataset pre-processing."},
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'."},
    )
    id_column_name: str = field(
        default="id",
        metadata={"help": "The name of the dataset column containing the id data. Defaults to 'id'"},
    )
    speaker_id_column_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset column containing the speaker id data. Defaults to None."},
    )
    max_duration_in_seconds: float = field(
        default=30.0,
        metadata={"help": "Filter audio files that are longer than `max_duration_in_seconds` seconds"},
    )
    max_label_length: int = field(
        default=256,
        metadata={"help": "Truncate transcriptions that are longer `max_label_length` tokens."},
    )
    concatenate_audio: bool = field(
        default=True,
        metadata={"help": "Whether or not to concatenate the audio samples to `max_duration_in_seconds`."},
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is"
                " especially useful when data preprocessing errors out in distributed"
                " training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with"
                " `preprocessing_only=True` so that the cached datasets can"
                " consequently be loaded in distributed training"
            )
        },
    )
    dataset_split_name: str = field(
        default="train+validation+test",
        metadata={
            "help": (
                "The name of the data set splits to use (via the datasets library)."
                " Defaults to 'train+validation+test'. Multiple splits can be passed by splitting a"
                " list through the '+' character, e.g. 'train+validation' will"
                " pseudo-label both the 'train' and 'validation' splits sequentially."
            )
        },
    )
    wandb_project: str = field(
        default="distil-whisper",
        metadata={"help": "The name of the wandb project."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to use dataset's streaming mode to load and pre-process the data."},
    )
    max_samples_per_split: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples per split to this value if set."},
    )
    return_timestamps: bool = field(
        default=False,
        metadata={
            "help": "Whether to return the timestamps with the text. This enables the `FlaxWhisperTimestampsLogitsProcessor`."
        },
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language for multilingual distillation. This argument should be set for multilingual distillation "
                "only. For English speech recognition, it should be left as `None`."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={
            "help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."
            "This argument should be set for multilingual distillation only. For English speech recognition, it should be left as `None`."
        },
    )
    decode_token_ids: bool = field(
        default=True,
        metadata={"help": "Deprecated. The predicted token ids should always be decoded to text transcriptions."},
    )
    private_dataset: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a private dataset for the pseudo-labelled data."},
    )

    def __post_init__(self):
        if not self.decode_token_ids:
            raise ValueError(
                "The argument `--decode_token_ids` is deprecated. The token ids are now always decoded to "
                "their corresponding text string. This is following a fix to the merges of the Whisper tokenizer"
                "on the Hugging Face Hub: https://huggingface.co/openai/whisper-large-v2/discussions/100. "
                "You should either omit the argument `--decode_token_ids`, or set it to True explicitly."
            )


def shift_tokens_right(label_ids: np.array, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift label ids one token to the right.
    """
    shifted_label_ids = np.zeros_like(label_ids)
    shifted_label_ids[:, 1:] = label_ids[:, :-1]
    shifted_label_ids[:, 0] = decoder_start_token_id

    return shifted_label_ids


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`Wav2Vec2Processor`])
            The processor used for proccessing the data.
        decoder_start_token_id (:obj: `int`)
            The start-of-sequence token id of the decoder.
        input_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned input sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        target_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned target sequences (according to the model's padding side and padding index).
            See above for details.
        max_target_length (:obj:`int`, `optional`):
            Maximum length of the ``labels`` of the returned list and optionally padding length (see above).
    """

    processor: Any
    decoder_start_token_id: int
    input_padding: Union[bool, str] = "max_length"
    target_padding: Union[bool, str] = "max_length"
    max_target_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]

        # dataloader returns a list of features which we convert to a dict
        input_features = {model_input_name: [feature[model_input_name] for feature in features]}
        label_features = {"input_ids": [feature["labels"] for feature in features]}

        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.input_padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            max_length=self.max_target_length,
            padding=self.target_padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore correctly when computing the loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def log_metric(
    accelerator,
    metrics: Dict,
    train_time: float,
    prefix: str = "eval",
):
    """Helper function to log all evaluation metrics with the correct prefixes and styling."""
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    accelerator.log(log_metrics)


def log_pred(
    accelerator,
    pred_str: List[str],
    label_str: List[str],
    norm_pred_str: List[str],
    norm_label_str: List[str],
    prefix: str = "eval",
    num_lines: int = 200000,
):
    """Helper function to log target/predicted transcriptions to weights and biases (wandb)."""
    if accelerator.is_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        # pretty name for split
        prefix = prefix.replace("/", "-")

        # convert str data to a wandb compatible format
        str_data = [[label_str[i], pred_str[i], norm_label_str[i], norm_pred_str[i]] for i in range(len(pred_str))]
        # log as a table with the appropriate headers
        wandb_tracker.log_table(
            table_name=f"{prefix}/all_predictions",
            columns=["Target", "Pred", "Norm Target", "Norm Pred"],
            data=str_data[:num_lines],
        )

        # log incorrect normalised predictions
        str_data = np.asarray(str_data)
        str_data_incorrect = str_data[str_data[:, -2] != str_data[:, -1]]
        # log as a table with the appropriate headers
        wandb_tracker.log_table(
            table_name=f"{prefix}/incorrect_predictions",
            columns=["Target", "Pred", "Norm Target", "Norm Pred"],
            data=str_data_incorrect[:num_lines],
        )


def main():
    # 1. Parse input arguments
    # We keep distinct sets of args, for cleaner separation of model/data/training related args
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Initialize the accelerator
    # We will let the accelerator handle device placement for us in this example
    # We simply have to specify the training precision and any trackers being used
    # We'll use the same dtype arguments as our JAX/Flax training script and convert
    # it to accelerate format
    if model_args.dtype == "float16":
        mixed_precision = "fp16"
        torch_dtype = torch.float16
    elif model_args.dtype == "bfloat16":
        mixed_precision = "bf16"
        torch_dtype = torch.bfloat16
    else:
        mixed_precision = "no"
        torch_dtype = torch.float32

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
        kwargs_handlers=[kwargs],
    )

    accelerator.init_trackers(project_name=data_args.wandb_project)

    # 3. Set-up basic logging
    # Create one log on every process with the configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Log a small summary on each proces
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Load dataset
    raw_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()
    token = model_args.token if model_args.token is not None else HfFolder().get_token()

    data_splits = data_args.dataset_split_name.split("+")
    for split in data_splits:
        with accelerator.main_process_first():
            raw_datasets[split] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=split,
                cache_dir=data_args.dataset_cache_dir,
                token=token,
                streaming=data_args.streaming,
                num_proc=data_args.preprocessing_num_workers if not data_args.streaming else None,
            )

    if data_args.audio_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--audio_column_name '{data_args.audio_column_name}' not found in dataset"
            f" '{data_args.dataset_name}'. Make sure to set `--audio_column_name` to"
            " the correct audio column - one of"
            f" {', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    if data_args.text_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--text_column_name {data_args.text_column_name} not found in dataset"
            f" '{data_args.dataset_name}'. Make sure to set `--text_column_name` to the"
            " correct text column - one of"
            f" {', '.join(next(iter(raw_datasets.values())).column_names)}."
        )
    
    # 7. Load pretrained model, tokenizer, and feature extractor
    config = WhisperConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=token,
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        (model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=token,
    )
    tokenizer = WhisperTokenizerFast.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=token,
    )
    processor = WhisperProcessor.from_pretrained(
        (model_args.processor_name if model_args.processor_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=token,
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        subfolder=model_args.subfolder,
        token=token,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )
    model.eval()

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    return_timestamps = data_args.return_timestamps
    if hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual:
        is_multilingual = True
        # We need to set the language and task ids for multilingual checkpoints
        tokenizer.set_prefix_tokens(
            language=data_args.language, task=data_args.task, predict_timestamps=return_timestamps
        )
    elif data_args.language is not None:
        raise ValueError(
            "Setting language token for an English-only checkpoint is not permitted. The language argument should "
            "only be set for multilingual checkpoints."
        )
    else:
        is_multilingual = False

    # 6. Resample speech dataset: `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name,
        datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
    )

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = int(data_args.max_duration_in_seconds * feature_extractor.sampling_rate)
    max_label_length = (
        data_args.max_label_length if data_args.max_label_length is not None else model.config.max_length
    )
    audio_column_name = data_args.audio_column_name
    sampling_rate = feature_extractor.sampling_rate

    preprocessing_batch_size = data_args.preprocessing_batch_size
    num_workers = data_args.preprocessing_num_workers
    dataloader_num_workers = training_args.dataloader_num_workers

    text_column_name = data_args.text_column_name
    model_input_name = feature_extractor.model_input_names[0]
    id_column_name = data_args.id_column_name
    speaker_id_column_name = data_args.speaker_id_column_name
    normalizer = (
        BasicTextNormalizer()
        if data_args.language is not None
        else EnglishTextNormalizer(tokenizer.english_spelling_normalizer)
    )

    timestamp_position = 3 if is_multilingual else 1
    decoder_prev_token_id = tokenizer.convert_tokens_to_ids("<|startofprev|>")
    decoder_eot_token_id = tokenizer.eos_token_id

    if data_args.max_samples_per_split is not None:
        for split in data_splits:
            raw_datasets[split] = (
                raw_datasets[split].take(data_args.max_samples_per_split)
                if data_args.streaming
                else raw_datasets[split].select(range(data_args.max_samples_per_split))
            )

    if speaker_id_column_name is not None:
        raw_datasets = raw_datasets.sort(speaker_id_column_name)

    def concatenate_dataset(batch):
        audio, text, speaker_id = [], [], []

        for sample_audio, sample_text, sample_speaker_id, sample_id in zip(
            batch[audio_column_name], 
            batch[text_column_name], 
            batch[speaker_id_column_name] if speaker_id_column_name else len(text) * [None],
            batch[id_column_name]
        ):
            try:
                audio_array = sample_audio["array"]
            except LibsndfileError:
                logger.warning(f"{sample_id} corrupted! Skipping sample.")
                continue

            if len(sample_text) == 0 or sample_text is None:
                logger.warning(f"{sample_id} has empty transcription! Skipping sample.")
                continue

            audio.append(audio_array)
            text.append(sample_text)
            sample_speaker_id.append(sample_id)

        # initialize concatenations
        concat_audio = [audio[0]]
        concat_text = [text[0]]
        concat_speaker_id = [speaker_id[0]]
        condition_on_prev = [0]

        for audio, text, speaker_id in zip(audio[1:], text[1:], speaker_id[1:]):
            is_same_speaker = speaker_id == concat_speaker_id[-1]
            is_concatenable = len(audio) + len(concat_audio[-1]) <= max_input_length 

            if is_same_speaker and is_concatenable:
                # inplace concatenation
                concat_audio[-1] = np.append(concat_audio[-1], audio)
                concat_text[-1] = concat_text[-1] + " " + text
            else:
                concat_audio.append(audio)
                concat_text.append(text)
                concat_speaker_id.append(speaker_id)
                condition_on_prev.append(1 if is_same_speaker else 0)   

        batch[audio_column_name] = [{"array": array, "sampling_rate": sampling_rate} for array in concat_audio]
        batch[text_column_name] = concat_text
        batch[id_column_name] = concat_speaker_id
        batch["condition_on_prev"] = condition_on_prev

        return batch

    raw_datasets_features = list(next(iter(raw_datasets.values())).features.keys())
    if data_args.concatenate_audio and not data_args.streaming:
        with accelerator.main_process_first():
            raw_datasets = raw_datasets.map(
                concatenate_dataset,
                batched=True,
                batch_size=preprocessing_batch_size,
                num_proc=num_workers,
                remove_columns=set(raw_datasets_features)
                - {audio_column_name, text_column_name, id_column_name, "condition_on_prev"},
                desc="Concatenating dataset...",
            )

        raw_datasets = raw_datasets.cast_column(
            audio_column_name, datasets.features.Audio(sampling_rate=sampling_rate)
        )
        pretty_name = data_args.dataset_name.split("/")[-1]

        def postprocess_ids(speaker_ids, indices):
            speaker_ids_formatted = []
            for speaker, idx in zip(speaker_ids, indices):
                formatted_idx = f"{pretty_name}-{speaker}-{idx}" if speaker is not None else f"{pretty_name}-{idx}"
                speaker_ids_formatted.append(formatted_idx)
            return {id_column_name: speaker_ids_formatted}
        
        with accelerator.main_process_first():
            raw_datasets = raw_datasets.map(
                postprocess_ids,
                input_columns=[id_column_name],
                with_indices=True,
                desc="Setting sample idxs...",
                batched=True,
                batch_size=preprocessing_batch_size,
                num_proc=num_workers,
            )
    elif data_args.concatenate_audio and data_args.streaming:
        raise ValueError(
            "Streaming mode is not yet compatible with concatenating audios to `max_duration_in_seconds`."
            "Either set `--streaming=False` and download the audios locally, or open an issue on the Distil-Whisper repo to request this feature."
        )

    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]

        # process targets
        input_str = batch[text_column_name]
        batch["labels"] = tokenizer(input_str, max_length=max_label_length, truncation=True).input_ids
        return batch

    raw_datasets_features = list(next(iter(raw_datasets.values())).features.keys())
    file_ids_dataset = IterableDatasetDict() if data_args.streaming else DatasetDict()
    for split in raw_datasets:
        file_ids_dataset[split] = raw_datasets[split][id_column_name]
    if data_args.streaming:
        with accelerator.main_process_first():
            vectorized_datasets = raw_datasets.map(prepare_dataset, remove_columns=raw_datasets_features)
    else:
        with accelerator.main_process_first():
            vectorized_datasets = raw_datasets.map(
                prepare_dataset,
                remove_columns=raw_datasets_features,
                num_proc=num_workers,
                desc="preprocess dataset",
            )

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        return

    if data_args.streaming and dataloader_num_workers > 0:
        logger.warning(
            "Using multiple dataloader num workers with streaming mode will result in different shards of "
            "data being transcribed in parallel. This is not advised if you want to preserve the order of the "
            "audio-text data."
        )

    # Handle the repository creation
    output_dir = training_args.output_dir
    if accelerator.is_main_process:
        if training_args.push_to_hub:
            if training_args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(output_dir).absolute().name,
                    token=training_args.hub_token,
                )
            else:
                repo_name = training_args.hub_model_id
            create_repo(repo_name, repo_type="dataset", exist_ok=True, token=training_args.hub_token)
            snapshot_download(repo_id=repo_name, local_dir=output_dir)

            # Ensure large txt files can be pushed to the Hub with git-lfs
            with open(os.path.join(output_dir, ".gitattributes"), "r+") as f:
                git_lfs_extensions = f.read()
                if "*.csv" not in git_lfs_extensions:
                    f.write("*.csv filter=lfs diff=lfs merge=lfs -text")

        elif output_dir is not None:
            # this is where we'll save our transcriptions
            os.makedirs(output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # 8. Load Metric
    metric = evaluate.load("wer")

    def compute_metrics(preds, labels, file_ids):
        # replace padded labels by the padding token
        for idx in range(len(labels)):
            labels[idx][labels[idx] == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=False, decode_with_timestamps=return_timestamps)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # normalize everything and re-compute the WER
        norm_pred_str = [normalizer(pred) for pred in pred_str]
        norm_label_str = [normalizer(label) for label in label_str]
        # for logging, we need the pred/labels to match the norm_pred/norm_labels, so discard any filtered samples here
        pred_str = [pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
        label_str = [label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]
        file_ids = [file_ids[i] for i in range(len(file_ids)) if len(norm_label_str[i]) > 0]
        # filtering step to only evaluate the samples that correspond to non-zero normalized references:
        norm_pred_str = [norm_pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
        norm_label_str = [norm_label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]

        wer = 100 * metric.compute(predictions=norm_pred_str, references=norm_label_str)

        return {"wer": wer}, pred_str, label_str, norm_pred_str, norm_label_str, file_ids

    def filter_eot_tokens(preds):
        for idx in range(len(preds)):
            # remove the EOT tokens to get the 'true' token length
            token_ids = [token for token in preds[idx] if token != decoder_eot_token_id]
            token_ids = token_ids + [decoder_eot_token_id]
            preds[idx] = token_ids
        return preds

    # 12. Define Training Schedule
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,  # <|startoftranscript|>
        input_padding="longest",
        target_padding="max_length",
        max_target_length=max_label_length,
    )

    # 14. Define generation arguments - we need to do this before we wrap the models in DDP
    # so that we can still access the configs
    num_beams = (
        training_args.generation_num_beams
        if training_args.generation_num_beams is not None
        else getattr(model.generation_config, "num_beams", 1)
    )

    gen_kwargs = {
        "max_length": max_label_length,
        "num_beams": num_beams,
        "return_timestamps": return_timestamps,
    }
    if hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual:
        # forcing the language and task tokens helps multilingual models in their generations
        gen_kwargs.update(
            {
                "language": data_args.language,
                "task": data_args.task,
            }
        )
    # remove any preset forced decoder ids since these are deprecated
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    # 15. Prepare everything with accelerate
    model = accelerator.prepare(model)

    def eval_step_with_save(split="eval"):
        # ======================== Evaluating ==============================
        eval_preds = []
        eval_labels = []
        eval_ids = []
        pred_str = []
        eval_start = time.time()

        eval_loader = DataLoader(
            vectorized_datasets[split],
            batch_size=per_device_eval_batch_size,
            collate_fn=data_collator,
            num_workers=dataloader_num_workers,
            pin_memory=True,
        )
        file_loader = DataLoader(
            file_ids_dataset[split],
            batch_size=per_device_eval_batch_size * accelerator.num_processes,
            num_workers=dataloader_num_workers,
        )

        eval_loader = accelerator.prepare(eval_loader)
        batches = tqdm(eval_loader, desc=f"Evaluating {split}...", disable=not accelerator.is_local_main_process)

        # make the split name pretty for librispeech etc
        split = split.replace(".", "-").split("/")[-1]
        output_csv = os.path.join(output_dir, f"{split}-transcription.csv")

        for step, (batch, file_ids) in enumerate(zip(batches, file_loader)):
            # Generate predictions and pad to max generated length
            generate_fn = model.module.generate if accelerator.num_processes > 1 else model.generate
            generated_ids = generate_fn(batch["input_features"].to(dtype=torch_dtype), **gen_kwargs)
            generated_ids = accelerator.pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id)
            # Gather all predictions and targets
            generated_ids, labels = accelerator.gather_for_metrics((generated_ids, batch["labels"]))
            eval_preds.extend(generated_ids.cpu().numpy())
            eval_labels.extend(labels.cpu().numpy())
            eval_ids.extend(file_ids)

            if step % training_args.logging_steps == 0 and step > 0:
                batches.write(f"Saving transcriptions for split {split} step {step}")
                accelerator.wait_for_everyone()
                pred_ids = eval_preds[-(len(eval_preds) - len(pred_str)) :]
                pred_ids = filter_eot_tokens(pred_ids)
                pred_str.extend(
                    tokenizer.batch_decode(
                        pred_ids, skip_special_tokens=False, decode_with_timestamps=return_timestamps
                    )
                )
                csv_data = [[eval_ids[i], pred_str[i]] for i in range(len(eval_preds))]

                with open(output_csv, "w", encoding="UTF8", newline="") as f:
                    writer = csv.writer(f)
                    # write multiple rows
                    writer.writerow(["file_id", "whisper_transcript"])
                    writer.writerows(csv_data)

                if training_args.push_to_hub and accelerator.is_main_process:
                    upload_folder(
                        folder_path=output_dir,
                        repo_id=repo_name,
                        repo_type="dataset",
                        commit_message=f"Saving transcriptions for split {split} step {step}.",
                    )

        accelerator.wait_for_everyone()
        eval_time = time.time() - eval_start

        # compute WER metric for eval sets
        wer_desc = ""
        if "validation" in split or "test" in split:
            eval_preds = filter_eot_tokens(eval_preds)
            wer_metric, pred_str, label_str, norm_pred_str, norm_label_str, eval_ids = compute_metrics(
                eval_preds, eval_labels, eval_ids
            )
            wer_desc = " ".join([f"Eval {key}: {value} |" for key, value in wer_metric.items()])
            # Save metrics + predictions
            log_metric(
                accelerator,
                metrics=wer_metric,
                train_time=eval_time,
                prefix=split,
            )
            log_pred(
                accelerator,
                pred_str,
                label_str,
                norm_pred_str,
                norm_label_str,
                prefix=split,
            )
        else:
            pred_ids = eval_preds[-(len(eval_preds) - len(pred_str)) :]
            pred_ids = filter_eot_tokens(pred_ids)
            pred_str.extend(
                tokenizer.batch_decode(pred_ids, skip_special_tokens=False, decode_with_timestamps=return_timestamps)
            )

        batches.write(f"Saving final transcriptions for split {split}.")
        csv_data = [[eval_ids[i], eval_preds[i]] for i in range(len(eval_preds))]
        with open(output_csv, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            # write multiple rows
            writer.writerow(["file_id", "whisper_transcript"])
            writer.writerows(csv_data)

        # Print metrics
        logger.info(wer_desc)

        if not data_args.streaming:
            raw_datasets[split] = raw_datasets[split].add_column("whisper_transcript", pred_str)
            raw_datasets[split] = raw_datasets[split].add_column("eval_preds", eval_preds)

            def add_concatenated_text(eval_preds, condition_on_prev):
                concatenated_prev = [None]
                for token_ids, condition in zip(eval_preds[:-1], condition_on_prev[1:]):
                    if condition is False:
                        concatenated_prev.append(None)
                    else:
                        prompt_ids = [token for token in token_ids if token != decoder_eot_token_id]
                        prompt_ids = [decoder_prev_token_id] + prompt_ids[timestamp_position:]
                        concatenated_prev.append(prompt_ids)
                return {"condition_on_prev": concatenated_prev}

            if data_args.concatenate_audio:
                with accelerator.main_process_first():
                    raw_datasets[split] = raw_datasets[split].map(
                        add_concatenated_text,
                        input_columns=["eval_preds", "condition_on_prev"],
                        remove_columns=["eval_preds"],
                        desc="Setting condition on prev...",
                        batched=True,
                        batch_size=preprocessing_batch_size,
                        num_proc=num_workers,
                    )

    logger.info("***** Running Labelling *****")
    logger.info("  Instantaneous batch size per device =" f" {training_args.per_device_eval_batch_size}")
    logger.info(
        f"  Total eval batch size (w. parallel & distributed) = {training_args.per_device_eval_batch_size * accelerator.num_processes}"
    )
    logger.info(f"  Predict labels with timestamps = {return_timestamps}")
    for split in data_splits:
        eval_step_with_save(split=split)
        accelerator.wait_for_everyone()
        if training_args.push_to_hub and accelerator.is_main_process:
            upload_folder(
                folder_path=output_dir,
                repo_id=repo_name,
                repo_type="dataset",
                commit_message=f"Saving final transcriptions for split {split.replace('.', '-').split('/')[-1]}",
            )
    if not data_args.streaming and accelerator.is_main_process:
        raw_datasets.save_to_disk(output_dir, num_proc=num_workers)
        if training_args.push_to_hub:
            raw_datasets.push_to_hub(repo_name, config_name=data_args.dataset_config_name)
    accelerator.end_training()


if __name__ == "__main__":
    main()
