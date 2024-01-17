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
Evaluating a Whisper model on one or more short-form evaluation datasets.
"""
import csv

# You can also adapt this script for your own evaluation tasks. Pointers for this are left as comments.
import logging
import os
import string
import sys
import time
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
from huggingface_hub import HfFolder, Repository, create_repo, get_full_repo_name
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer, BasicTextNormalizer
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
    attn_type: Optional[str] = field(
        default=None,
        metadata={"help": "Which attn type to use: ['eager', 'sdpa', 'flash_attention_2']"},
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
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'."},
    )
    max_label_length: int = field(
        default=256,
        metadata={"help": "Truncate transcriptions that are longer `max_label_length` tokens."},
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
                " evaluate both the 'train' and 'validation' splits sequentially."
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

def convert_dataset_str_to_list(
    dataset_names,
    dataset_config_names,
    splits=None,
    text_column_names=None,
    dataset_samples=None,
    default_split="train",
) -> List[Dict]:
    """
    Given three lists of dataset names, configs and splits, this function groups the corresponding
    names/configs/splits. Each dataset is assigned a unique dictionary with these metadata values, and the
    function returns a list of dictionaries, one for each dataset.
    """
    if isinstance(dataset_names, str):
        dataset_names = dataset_names.split("+")
        dataset_config_names = dataset_config_names.split("+")
        splits = splits.split("+") if splits is not None else None
        text_column_names = text_column_names.split("+") if text_column_names is not None else None
        dataset_samples = dataset_samples.split("+") if dataset_samples is not None else None

    # basic checks to ensure we've got the right number of datasets/configs/splits/columns/probs
    if len(dataset_names) != len(dataset_config_names):
        raise ValueError(
            f"Ensure one config is passed for each dataset, got {len(dataset_names)} datasets and"
            f" {len(dataset_config_names)} configs."
        )

    if splits is not None and len(splits) != len(dataset_names):
        raise ValueError(
            f"Ensure one split is passed for each dataset, got {len(dataset_names)} datasets and {len(splits)} splits."
        )

    if text_column_names is not None and len(text_column_names) != len(dataset_names):
        raise ValueError(
            f"Ensure one text column name is passed for each dataset, got {len(dataset_names)} datasets and"
            f" {len(text_column_names)} text column names."
        )

    if dataset_samples is not None:
        if len(dataset_samples) != len(dataset_names):
            raise ValueError(
                f"Ensure one sample is passed for each dataset, got {len(dataset_names)} datasets and "
                f"{len(dataset_samples)} samples."
            )
        dataset_samples = [float(ds_sample) for ds_sample in dataset_samples]
    else:
        dataset_samples = [None] * len(dataset_names)

    text_column_names = (
        text_column_names if text_column_names is not None else ["text" for _ in range(len(dataset_names))]
    )
    splits = splits if splits is not None else [default_split for _ in range(len(dataset_names))]

    dataset_names_dict = []
    for i, ds_name in enumerate(dataset_names):
        dataset_names_dict.append(
            {
                "name": ds_name,
                "config": dataset_config_names[i],
                "split": splits[i],
                "text_column_name": text_column_names[i],
                "samples": dataset_samples[i],
            }
        )
    return dataset_names_dict

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

    dataset_names_dict = convert_dataset_str_to_list(
        data_args.dataset_name,
        data_args.dataset_config_name,
        splits=data_args.dataset_split_name,
        text_column_names=data_args.text_column_name,
    )
    all_eval_splits = []
    if len(dataset_names_dict) == 1:
        # load a single eval set
        dataset_dict = dataset_names_dict[0]
        all_eval_splits.append("eval")
        raw_datasets["eval"] = load_dataset(
            dataset_dict["name"],
            dataset_dict["config"],
            split=dataset_dict["split"],
            cache_dir=data_args.dataset_cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
        if data_args.text_column_name != "text":
            raw_datasets["eval"] = raw_datasets["eval"].rename_column(data_args.text_column_name, "text")
    else:
        # load multiple eval sets
        for dataset_dict in dataset_names_dict:
            if dataset_dict["name"] == "esb/diagnostic-dataset":
                # for the ESB diagnostic dataset, the dataset name is effectively the config
                pretty_name = f"{dataset_dict['config']}-diagnostic/{dataset_dict['split']}"
            else:
                pretty_name = f"{dataset_dict['name'].split('/')[-1]}/{dataset_dict['split'].replace('.', '-')}"
            all_eval_splits.append(pretty_name)
            raw_datasets[pretty_name] = load_dataset(
                dataset_dict["name"],
                dataset_dict["config"],
                split=dataset_dict["split"],
                cache_dir=data_args.dataset_cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
            # make column names consistent (text, audio)
            if dataset_dict["text_column_name"] != "text":
                raw_datasets[pretty_name] = raw_datasets[pretty_name].rename_column(
                    dataset_dict["text_column_name"], "text"
                )
            raw_datasets[pretty_name] = raw_datasets[pretty_name].remove_columns(
                set(raw_datasets[pretty_name].features.keys()) - {"audio", "text"}
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

    # set attn_type in a backwards compatible way
    if model_args.attn_type == "flash_attn":
        attn_type = "sdpa"
    elif model_args.attn_type == "flash_attn_2":
        attn_type = "flash_attention_2"
    elif model_args.attn_type in [None, "eager", "sdpa", "flash_attention_2"]:
        attn_type = model_args.attn_type
    else:
        raise ValueError(
            f"`attn_type` should be one of ['eager', 'sdpa', 'flash_attention_2'], got {model_args.attn_type}."
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
        attn_implementation=attn_type,
    )

    model.eval()

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    return_timestamps = data_args.return_timestamps
    if hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual:
        # We need to set the language and task ids for multilingual checkpoints
        tokenizer.set_prefix_tokens(
            language=data_args.language, task=data_args.task, predict_timestamps=return_timestamps
        )
    elif data_args.language is not None:
        raise ValueError(
            "Setting language token for an English-only checkpoint is not permitted. The language argument should "
            "only be set for multilingual checkpoints."
        )

    # 6. Resample speech dataset: `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name,
        datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
    )

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_label_length = (
        data_args.max_label_length if data_args.max_label_length is not None else model.config.max_length
    )
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    dataloader_num_workers = training_args.dataloader_num_workers
    model_input_name = feature_extractor.model_input_names[0]
    normalizer = (
        BasicTextNormalizer() if data_args.language is not None else EnglishTextNormalizer(tokenizer.english_spelling_normalizer)
    )

    if data_args.max_samples_per_split is not None:
        for split in all_eval_splits:
            raw_datasets[split] = (
                raw_datasets[split].take(data_args.max_samples_per_split)
                if data_args.streaming
                else raw_datasets[split].select(range(data_args.max_samples_per_split))
            )

    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]

        # process targets
        batch["labels"] = tokenizer(batch["text"], max_length=max_label_length, truncation=True).input_ids
        return batch

    raw_datasets_features = list(next(iter(raw_datasets.values())).features.keys())
    if data_args.streaming:
        vectorized_datasets = raw_datasets.map(prepare_dataset, remove_columns=raw_datasets_features)
    else:
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

    # 8. Load Metric
    metric = evaluate.load("wer")

    def compute_metrics(preds, labels):
        # replace padded labels by the padding token
        for idx in range(len(labels)):
            labels[idx][labels[idx] == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True, decode_with_timestamps=return_timestamps)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
        wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

        # normalize everything and re-compute the WER
        norm_pred_str = [normalizer(pred) for pred in pred_str]
        norm_label_str = [normalizer(label) for label in label_str]
        # for logging, we need the pred/labels to match the norm_pred/norm_labels, so discard any filtered samples here
        pred_str = [pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
        label_str = [label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]
        # filtering step to only evaluate the samples that correspond to non-zero normalized references:
        norm_pred_str = [norm_pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
        norm_label_str = [norm_label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]

        wer = 100 * metric.compute(predictions=norm_pred_str, references=norm_label_str)

        return {"wer": wer, "wer_ortho": wer_ortho}, pred_str, label_str, norm_pred_str, norm_label_str

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
    max_new_tokens = training_args.generation_max_length if training_args.generation_max_length is not None else max_label_length

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
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

    # 15. Prepare everything with accelerate
    model = accelerator.prepare(model)

    def eval_step(split="eval"):
        # ======================== Evaluating ==============================
        eval_preds = []
        eval_labels = []
        eval_start = time.time()

        eval_loader = DataLoader(
            vectorized_datasets[split],
            batch_size=per_device_eval_batch_size,
            collate_fn=data_collator,
            num_workers=dataloader_num_workers,
            pin_memory=True,
        )

        eval_loader = accelerator.prepare(eval_loader)
        batches = tqdm(eval_loader, desc=f"Evaluating {split}...", disable=not accelerator.is_local_main_process)

        # make the split name pretty for librispeech etc
        split = split.replace(".", "-").split("/")[-1]

        for step, batch in enumerate(batches):
            # Generate predictions and pad to max generated length
            generate_fn = model.module.generate if accelerator.num_processes > 1 else model.generate
            generated_ids = generate_fn(batch["input_features"].to(dtype=torch_dtype), **gen_kwargs)
            generated_ids = accelerator.pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id)
            # Gather all predictions and targets
            generated_ids, labels = accelerator.gather_for_metrics((generated_ids, batch["labels"]))
            eval_preds.extend(generated_ids.cpu().numpy())
            eval_labels.extend(labels.cpu().numpy())

        accelerator.wait_for_everyone()
        eval_time = time.time() - eval_start

        # compute WER metric for eval sets
        wer_metric, pred_str, label_str, norm_pred_str, norm_label_str = compute_metrics(eval_preds, eval_labels)
        wer_desc = " ".join([f"Eval {key}: {value} |" for key, value in wer_metric.items()])
        logger.info(wer_desc)
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

    logger.info("***** Running Evaluation *****")
    logger.info("  Instantaneous batch size per device =" f" {training_args.per_device_eval_batch_size}")
    logger.info(
        f"  Total eval batch size (w. parallel & distributed) = {training_args.per_device_eval_batch_size * accelerator.num_processes}"
    )
    logger.info(f"  Predict labels with timestamps = {return_timestamps}")
    for split in all_eval_splits:
        eval_step(split=split)
        accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
