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
Training the Whisper model for sequence to sequence speech recognition via teacher-student distillation.
"""
# You can also adapt this script for your own distillation tasks. Pointers for this are left as comments.

import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
import evaluate
import numpy as np
import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import (
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
)
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AddedToken,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
    get_scheduler,
    set_seed,
)
from transformers.modeling_outputs import BaseModelOutput
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
    teacher_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained teacher model or model identifier from huggingface.co/models"}
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset ids by a '+' symbol. For example, to load LibriSpeech "
            "and Common Voice, set `train_dataset_name='librispeech_asr+common_voice'`."
        },
    )
    train_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset configs by a '+' symbol. Note that the order of the configs should "
            "match the order of the datasets."
        },
    )
    train_dataset_samples: str = field(
        default=None,
        metadata={
            "help": "Number of samples in each dataset when loading multiple datasets with streaming mode. "
            "Not required when using one dataset or non-streaming mode. The sample values provide the sampling "
            "probability for each dataset. Setting them equal to the number of sample values ensures that every "
            "sample from every dataset is used once per epoch."
        },
    )
    eval_dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the evaluation dataset to use (via the datasets library). Defaults to the training "
            "dataset name if unspecified. Load multiple evaluation datasets by separating dataset "
            "ids by a '+' symbol."
        },
    )
    eval_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the evaluation dataset to use (via the datasets library). Defaults to the "
            "training dataset config name if unspecified."
        },
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
        metadata={"help": "The number of processes to use for the preprocessing if using non-streaming mode."},
    )
    preprocessing_batch_size: Optional[int] = field(
        default=256,
        metadata={"help": "Number of examples per batch provided to the `prepare_dataset` function."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset column containing the text data in the training set."},
    )
    eval_text_column_name: str = field(
        default="text",
        metadata={"help": ("The name of the dataset column containing the text data in the evaluation set.")},
    )
    max_duration_in_seconds: float = field(
        default=30.0,
        metadata={"help": "Filter audio files that are longer than `max_duration_in_seconds` seconds"},
    )
    min_duration_in_seconds: float = field(
        default=0.0,
        metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"},
    )
    max_label_length: int = field(
        default=128,
        metadata={"help": "Truncate transcriptions that are longer `max_label_length` tokens."},
    )
    pad_target_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If set will pad the target sequence to a multiple of the provided"
                " value. This is important to avoid triggering recompilations on TPU."
                " If unspecified, will default to padding the targets to max length."
            )
        },
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
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": (
                "The name of the evaluation data set split to use (via the datasets library). Defaults to 'validation'"
            )
        },
    )
    streaming: bool = field(
        default=True,
        metadata={"help": "Whether to use Datasets' streaming mode to load and pre-process the data."},
    )
    wer_threshold: float = field(
        default=None,
        metadata={
            "help": "Filter training data with Whisper transcriptions that have greater than `wer_threshold` "
            "WER with the normalised transcriptions. This only takes effect if training on pseudo-labels targets."
            "If `--use_pseudo_labels=False`, then no WER filtering is performed, since we train directly on the text"
            "transcriptions."
        },
    )
    use_pseudo_labels: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use pseudo-label transcriptions as the targets. If True, the pseudo-labels "
            "must be in the dataset column `whisper_transcript` from the previous pseudo-labelling step. This is "
            "not currently yet configurable."
        },
    )
    timestamp_probability: float = field(
        default=0.2, metadata={"help": "Probability for training on timestamped tokens if the data contains it."}
    )
    condition_on_prev_probability: float = field(
        default=0.2, metadata={"help": "Probability for conditioning on the previous text example."}
    )
    return_timestamps: bool = field(
        default=False, metadata={"help": "Whether or not to predict timestamps in the generation step."}
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
    wandb_project: str = field(
        default="distil-whisper",
        metadata={"help": "The name of the wandb project."},
    )


@dataclass
class DistillationTrainingArguments(Seq2SeqTrainingArguments):
    freeze_encoder: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to freeze the entire encoder model. Only recommended when the entire encoder has been "
                "copied from the teacher model."
            )
        },
    )
    temperature: Optional[float] = field(
        default=2.0, metadata={"help": "Temperature to anneal the logits when computing the softmax."}
    )
    kl_weight: Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                "Weighting assigned to the MSE loss in the KD formulation. MSE loss is "
                "computed between the teacher-student hidden states and attentions."
            )
        },
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "The data type (dtype) in which to run training. One of `float32` (full-precision), "
                "`float16` or `bfloat16` (both half-precision)."
            )
        },
    )


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`Wav2Vec2Processor`])
            The processor used for proccessing the data.
        decoder_start_token_id (:obj: `int`)
            The start-of-sequence token id of the decoder.
        decoder_prev_token_id (:obj: `int`)
            The start-of-prompt token id of the decoder
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
    decoder_prev_token_id: int
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

        # shift labels to the right to get decoder input ids
        labels = labels_batch["input_ids"]
        decoder_input_ids = labels[:, :-1]
        labels = labels[:, 1:]
        labels_mask = labels_batch.attention_mask[:, 1:]

        # replace padding with -100 to ignore correctly when computing the loss
        labels = labels.masked_fill(labels_mask.ne(1), -100)

        # replace initial prompt tokens with -100 to ignore correctly when computing the loss
        bos_index = torch.argmax((labels == self.decoder_start_token_id).long(), dim=1)
        bos_index = torch.where(bos_index > 0, bos_index + 1, bos_index)
        prompt_mask = torch.arange(labels.shape[1]) < bos_index[:, None]
        labels = torch.where(prompt_mask, -100, labels)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids

        return batch


def log_metric(
    accelerator,
    metrics: Dict,
    train_time: float,
    step: int,
    epoch: int,
    learning_rate: float = None,
    prefix: str = "train",
):
    """Helper function to log all training/evaluation metrics with the correct prefixes and styling."""
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    log_metrics[f"{prefix}/epoch"] = epoch
    if learning_rate is not None:
        log_metrics[f"{prefix}/learning_rate"] = learning_rate
    accelerator.log(log_metrics, step=step)


def log_pred(
    accelerator,
    pred_str: List[str],
    label_str: List[str],
    norm_pred_str: List[str],
    norm_label_str: List[str],
    step: int,
    prefix: str = "eval",
    num_lines: int = 200000,
):
    """Helper function to log target/predicted transcriptions to weights and biases (wandb)."""
    if accelerator.is_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        # pretty name for current step: step 50000 -> step 50k
        cur_step_pretty = f"{int(step // 1000)}k" if step > 1000 else step
        prefix_pretty = prefix.replace("/", "-")

        # convert str data to a wandb compatible format
        str_data = [[label_str[i], pred_str[i], norm_label_str[i], norm_pred_str[i]] for i in range(len(pred_str))]
        # log as a table with the appropriate headers
        wandb_tracker.log_table(
            table_name=f"predictions/{prefix_pretty}-step-{cur_step_pretty}",
            columns=["Target", "Pred", "Norm Target", "Norm Pred"],
            data=str_data[:num_lines],
            step=step,
        )

        # log incorrect normalised predictions
        str_data = np.asarray(str_data)
        str_data_incorrect = str_data[str_data[:, -2] != str_data[:, -1]]
        # log as a table with the appropriate headers
        wandb_tracker.log_table(
            table_name=f"incorrect_predictions/{prefix_pretty}-step-{cur_step_pretty}",
            columns=["Target", "Pred", "Norm Target", "Norm Pred"],
            data=str_data_incorrect[:num_lines],
            step=step,
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


def load_multiple_datasets(
    dataset_names: Union[List, str],
    dataset_config_names: Union[List, str],
    splits: Optional[Union[List, str]] = None,
    text_column_names: Optional[List] = None,
    sampling_rate: Optional[int] = 16000,
    stopping_strategy: Optional[str] = "first_exhausted",
    dataset_samples: Optional[Union[List, np.array]] = None,
    streaming: Optional[bool] = True,
    seed: Optional[int] = None,
    accelerator: Optional[Accelerator] = None,
    use_pseudo_labels: float = None,
    **kwargs,
) -> IterableDataset:
    dataset_names_dict = convert_dataset_str_to_list(
        dataset_names, dataset_config_names, splits, text_column_names, dataset_samples
    )

    if dataset_samples is not None:
        dataset_samples = [ds_dict["samples"] for ds_dict in dataset_names_dict]
        probabilities = np.array(dataset_samples) / np.sum(dataset_samples)
    else:
        probabilities = None

    all_datasets = []
    # iterate over the datasets we want to interleave
    for dataset_dict in tqdm(
        dataset_names_dict,
        desc="Combining datasets...",
        disable=not accelerator.is_local_main_process if accelerator is not None else False,
    ):
        dataset = load_dataset(
            dataset_dict["name"],
            dataset_dict["config"],
            split=dataset_dict["split"],
            streaming=streaming,
            **kwargs,
        )
        # resample to specified sampling rate
        dataset = dataset.cast_column("audio", datasets.features.Audio(sampling_rate))
        dataset_features = dataset.features.keys()
        columns_to_keep = {"audio", "text"}

        if dataset_dict["text_column_name"] not in dataset_features:
            raise ValueError(
                f"Text column name {dataset_dict['text_column_name']} not found in dataset"
                f" '{dataset_dict['name']}'. Make sure to set `--text_column_name` to the"
                f" correct text column - one of {', '.join(dataset_features)}."
            )

        # blanket renaming of all transcription columns to text
        if dataset_dict["text_column_name"] != "text":
            dataset = dataset.rename_column(dataset_dict["text_column_name"], "text")

        if use_pseudo_labels:
            if "whisper_transcript" not in dataset_features:
                raise ValueError(
                    f"Pseudo-label column `whisper_transcript` not found in dataset {dataset_dict['name']}. Ensure"
                    "pseudo-labels are present in the dataset under this column name, or train directly on the text "
                    "labels by setting `--use_pseudo_labels=False` and defining the appropriate `--text_column_name`."
                )
            columns_to_keep.add("whisper_transcript")
        dataset_features = dataset.features.keys()
        dataset = dataset.remove_columns(set(dataset_features - columns_to_keep))
        all_datasets.append(dataset)

    if len(all_datasets) == 1:
        # we have a single dataset so just return it as is
        return all_datasets[0]

    if streaming:
        interleaved_dataset = interleave_datasets(
            all_datasets,
            stopping_strategy=stopping_strategy,
            probabilities=probabilities,
            seed=seed,
        )
    else:
        interleaved_dataset = concatenate_datasets(all_datasets)

    return interleaved_dataset


def get_layers_to_supervise(student_layers: int, teacher_layers: int) -> Dict:
    """Helper function to map the student layer i to the teacher layer j whose output we'd like them to emulate. Used
    for MSE loss terms in distillation (hidden-states and activations). Student layers are paired with teacher layers
    in equal increments, e.g. for a 12-layer model distilled to a 3-layer model, student layer 0 emulates teacher layer
    3 (such that it behaves like the first 4 teacher layers), student layer 1 emulates teacher layer 7, and student layer
    2 emulates teacher layer 11. This mapping is summarised by the dictionary: {0: 3, 1: 7, 2: 11}, which is precisely
    the output of this function for the arguments (student_layers=3, teacher_layers=12)."""
    layer_intervals = np.linspace(teacher_layers // student_layers - 1, teacher_layers - 1, student_layers, dtype=int)
    layer_intervals[-1] = teacher_layers - 1
    layer_map = {}

    for student_layer, teacher_layer in enumerate(layer_intervals):
        layer_map[student_layer] = teacher_layer

    return layer_map


def sorted_checkpoints(output_dir=None, checkpoint_prefix="checkpoint") -> List[str]:
    """Helper function to sort saved checkpoints from oldest to newest."""
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

    for path in glob_checkpoints:
        regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
        if regex_match is not None and regex_match.groups() is not None:
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def rotate_checkpoints(save_total_limit=None, output_dir=None, checkpoint_prefix="checkpoint") -> None:
    """Helper function to delete old checkpoints."""
    if save_total_limit is None or save_total_limit <= 0:
        return
    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(output_dir=output_dir, checkpoint_prefix=checkpoint_prefix)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        shutil.rmtree(checkpoint, ignore_errors=True)


_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+)-epoch-(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _RE_CHECKPOINT.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_RE_CHECKPOINT.search(x).groups()[0])))


def get_parameter_names(model, forbidden_layer_types, forbidden_module=None):
    """
    Returns the names of the model parameters that are not inside a forbidden layer or forbidden module.
    Can be used to get a subset of parameter names for decay masks, or to exclude parameters from an optimiser
    (e.g. if the module is frozen).
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types, forbidden_module)
            if not (
                isinstance(child, tuple(forbidden_layer_types))
                or (child in tuple(forbidden_module) if forbidden_module is not None else False)
            )
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def main():
    # 1. Parse input arguments
    # We keep distinct sets of args, for cleaner separation of model/data/training related args
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DistillationTrainingArguments))

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
    # The teacher model can safely be cast to the dtype of training since we don't
    # update the params
    if training_args.dtype == "float16":
        mixed_precision = "fp16"
        teacher_dtype = torch.float16
    elif training_args.dtype == "bfloat16":
        mixed_precision = "bf16"
        teacher_dtype = torch.bfloat16
    else:
        mixed_precision = "no"
        teacher_dtype = torch.float32

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
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

    # 4. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 5. Handle the repository creation
    if accelerator.is_main_process:
        if training_args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = training_args.hub_model_id
            if repo_name is None:
                repo_name = Path(training_args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            repo_id = create_repo(repo_name, exist_ok=True, token=training_args.hub_token).repo_id
            # Clone repo locally
            repo = Repository(training_args.output_dir, clone_from=repo_id, token=training_args.hub_token)

            with open(os.path.join(training_args.output_dir, ".gitignore"), "w+") as gitignore:
                if "wandb" not in gitignore:
                    gitignore.write("wandb\n")
        elif training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # 6. Load dataset - either streaming or non-streaming (offline)
    raw_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()

    # set seed for determinism
    set_seed(training_args.seed)

    if training_args.do_train:
        raw_datasets["train"] = load_multiple_datasets(
            data_args.train_dataset_name,
            data_args.train_dataset_config_name,
            splits=data_args.train_split_name,
            text_column_names=data_args.text_column_name,
            use_pseudo_labels=data_args.use_pseudo_labels,
            streaming=data_args.streaming,
            dataset_samples=data_args.train_dataset_samples,
            seed=training_args.seed,
            accelerator=accelerator,
            cache_dir=data_args.dataset_cache_dir,
            token=model_args.token,
        )
        raw_datasets_train_features = list(raw_datasets["train"].features.keys())

    if training_args.do_eval:
        dataset_names_dict = convert_dataset_str_to_list(
            data_args.eval_dataset_name if data_args.eval_dataset_name else data_args.train_dataset_name,
            data_args.eval_dataset_config_name
            if data_args.eval_dataset_config_name
            else data_args.train_dataset_config_name,
            splits=data_args.eval_split_name,
            text_column_names=data_args.eval_text_column_name,
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
            if data_args.eval_text_column_name != "text":
                raw_datasets["eval"] = raw_datasets["eval"].rename_column(data_args.eval_text_column_name, "text")
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

    if not training_args.do_train and not training_args.do_eval:
        raise ValueError(
            "Cannot not train and not do evaluation. At least one of training or evaluation has to be performed."
        )

    # 7. Load pretrained model, tokenizer, and feature extractor
    config = WhisperConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        (model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )
    tokenizer = WhisperTokenizerFast.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
    )

    # override timestamp tokens until tokenizer issues are fixed in transformers
    timestamps = [AddedToken("<|%.2f|>" % (i * 0.02), lstrip=False, rstrip=False) for i in range(1500 + 1)]
    tokenizer.add_tokens(timestamps)

    teacher_model = WhisperForConditionalGeneration.from_pretrained(
        model_args.teacher_model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        low_cpu_mem_usage=True,
        torch_dtype=teacher_dtype,
    )

    student_model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        subfolder=model_args.subfolder,
        token=model_args.token,
        low_cpu_mem_usage=True,
    )

    if student_model.config.decoder_start_token_id is None or teacher_model.config.decoder_start_token_id is None:
        raise ValueError(
            f"Make sure that `config.decoder_start_token_id` is correctly defined for both the "
            f"student and teacher model. Got {student_model.config.decoder_start_token_id} for the "
            f"student and {teacher_model.config.decoder_start_token_id} for the teacher."
        )

    share_hidden_states = training_args.freeze_encoder and student_model.config.d_model == teacher_model.config.d_model

    # enable gradient checkpointing if necessary
    if training_args.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()

    # freeze student encoder if necessary
    if training_args.freeze_encoder:
        student_model.freeze_encoder()
        student_model.model.encoder.gradient_checkpointing = False

    # if share_hidden_states:
    # tie the weights for the student encoder if we're freezing it and it's the same as the teacher
    #    student_model.model.encoder = teacher_model.model.encoder

    if hasattr(teacher_model.generation_config, "is_multilingual") and teacher_model.generation_config.is_multilingual:
        # We need to set the language and task ids for previously multilingual checkpoints
        is_multilingual = True
        tokenizer.set_prefix_tokens(language=data_args.language, task=data_args.task, predict_timestamps=False)
        student_model.generation_config.update(
            **{
                "language": data_args.language,
                "task": data_args.task,
            }
        )
    elif data_args.language is not None:
        raise ValueError(
            "Setting language token for an English-only checkpoint is not permitted. The language argument should "
            "only be set for multilingual checkpoints."
        )
    else:
        is_multilingual = False

    # 8. Create a single speech processor - make sure all processes wait until data is saved
    if accelerator.is_main_process:
        feature_extractor.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        # save the config and generation config as well
        config.save_pretrained(training_args.output_dir)
        student_model.generation_config.save_pretrained(training_args.output_dir)

    accelerator.wait_for_everyone()
    processor = WhisperProcessor.from_pretrained(training_args.output_dir)

    # 9. Resample speech dataset: `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    sampling_rate = feature_extractor.sampling_rate
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name,
        datasets.features.Audio(sampling_rate=sampling_rate),
    )

    # 10. Preprocessing the datasets: we need to read the audio files as arrays and tokenize the targets.
    # 10.1: Define the pre-processing constants
    max_input_length = int(data_args.max_duration_in_seconds * sampling_rate)
    min_input_length = int(data_args.min_duration_in_seconds * sampling_rate)
    max_label_length = (
        data_args.max_label_length if data_args.max_label_length is not None else student_model.config.max_length
    )

    timestamp_probability = data_args.timestamp_probability
    condition_on_prev_probability = data_args.condition_on_prev_probability
    return_timestamps = data_args.return_timestamps if timestamp_probability > 0 else False

    timestamp_ids = tokenizer.timestamp_ids()
    timestamp_begin = tokenizer.all_special_ids[-1]
    timestamp_position = 3 if is_multilingual else 1

    decoder_start_token_id = student_model.config.decoder_start_token_id  # <|startoftranscript|>
    decoder_prev_token_id = tokenizer.all_special_ids[-3]  # <|startofprev|>
    decoder_eot_token_id = tokenizer.eos_token_id

    language = data_args.language
    task = data_args.task

    num_workers = data_args.preprocessing_num_workers
    dataloader_num_workers = training_args.dataloader_num_workers

    metric = evaluate.load("wer")
    normalizer = (
        BasicTextNormalizer() if language is not None else EnglishTextNormalizer(tokenizer.english_spelling_normalizer)
    )
    wer_threshold = data_args.wer_threshold
    use_pseudo_labels = data_args.use_pseudo_labels
    train_text_column_name = "whisper_transcript" if use_pseudo_labels else "text"

    # 10.2: filter based on maximum number of training/evaluation samples
    if training_args.do_train and data_args.max_train_samples is not None:
        raw_datasets["train"] = (
            raw_datasets["train"].take(data_args.max_train_samples)
            if data_args.streaming
            else raw_datasets["train"].select(range(data_args.max_train_samples))
        )

    if training_args.do_eval and data_args.max_eval_samples is not None:
        for eval_split in all_eval_splits:
            raw_datasets[eval_split] = (
                raw_datasets[eval_split].take(data_args.max_eval_samples)
                if data_args.streaming
                else raw_datasets[eval_split].select(range(data_args.max_eval_samples))
            )

    # 10.3: filter training data based on WER threshold -> this is KEY to good distillation performance
    def is_wer_in_range(ground_truth, whisper_transcript):
        norm_ground_truth = normalizer(ground_truth)
        if (
            isinstance(whisper_transcript, str)
            and whisper_transcript.startswith("[")
            and whisper_transcript.endswith("]")
        ):
            whisper_transcript = re.findall(r"\d+", whisper_transcript)
            whisper_transcript = [int(token) for token in whisper_transcript]
        if isinstance(whisper_transcript, list):
            whisper_transcript = tokenizer.decode(whisper_transcript, skip_special_tokens=True)
        if len(norm_ground_truth) > 0 and whisper_transcript is not None:
            norm_whisper_transcript = normalizer(whisper_transcript)
            wer = 100 * metric.compute(predictions=[norm_whisper_transcript], references=[norm_ground_truth])
            return wer < wer_threshold
        else:
            # filter automatically since we can't know the WER
            return False

    filter_by_wer_threshold = partial(
        raw_datasets["train"].filter,
        function=is_wer_in_range,
        input_columns=["text", "whisper_transcript"],
    )

    if wer_threshold is not None and use_pseudo_labels:
        raw_datasets["train"] = (
            filter_by_wer_threshold(num_proc=num_workers, desc="filtering train dataset by wer")
            if not data_args.streaming
            else filter_by_wer_threshold()
        )

    # 10.4: pre-process training/evaluation datasets
    def has_timestamp_tokens(input_str):
        """
        Identify whether the input string contains timestamp tokens, of the form <|0.00|>, by searching for
        pairs of left and right-angle brackets.
        """
        return bool(re.search("\<[^\>]*\>", input_str))

    def prepare_train_dataset(batch):
        """
        Pre-process the raw dataset in a three stage process:
            1. Convert the audio arrays to log-mel spectrogram inputs
            2. Possibly filter the timestamp tokens from the token ids (depending on the timestamp probability)
            3. Possibly add prompt tokens if conditioning on previous text (depending on the conditioning probability)
        TODO(SG): see whether we can 'pack' the audio inputs closer to 30 second chunks
        """
        # process audio input
        audio = [sample["array"] for sample in batch["audio"]]
        inputs = feature_extractor(audio, sampling_rate=sampling_rate)
        batch["input_features"] = inputs.input_features
        batch["input_length"] = [len(sample) for sample in audio]

        # process text targets - for training these are the Whisper-generated pseudo-labels
        input_str_batched = batch[train_text_column_name]

        all_token_ids = []
        all_token_ids_unprompted = []
        for input_str in input_str_batched:
            if isinstance(input_str, list):
                # pseudo-labelled transcriptions have been retained as token ids (`decode_token_ids=False`)
                token_ids = input_str
            elif input_str[0].startswith("[") and input_str[0].endswith("]"):
                token_ids = re.findall(r"\d+", input_str)
                token_ids = [int(token) for token in token_ids]
            else:
                token_ids = None

            if token_ids is not None:
                # remove the EOT tokens to get the 'true' token length
                token_ids = [token for token in token_ids if token != decoder_eot_token_id]
                token_ids = token_ids + [decoder_eot_token_id]
                # check whether we have timestamps in the PLs and filter if required
                has_timestamps = len(set(token_ids) & set(timestamp_ids)) > 0
                if has_timestamps:
                    # sample from binomial distribution to get probability of training on timestamps
                    predict_timestamps = bool(np.random.binomial(1, timestamp_probability))
                    if not predict_timestamps:
                        # filter timestamps and insert the <|notimestamps|> task token
                        token_ids = [token for token in token_ids if token < timestamp_begin]
                        token_ids.insert(timestamp_position, timestamp_begin)
            else:
                # pseudo-labelled transcriptions have been decoded to text (`decode_token_ids=True`)
                has_timestamps = has_timestamp_tokens(input_str)

                if has_timestamps:
                    predict_timestamps = bool(np.random.binomial(1, timestamp_probability))
                    if not predict_timestamps:
                        # filter timestamp token ids if not part of the prediction task
                        input_str = tokenizer._filter_timestamp_ids(input_str)
                else:
                    predict_timestamps = False

                tokenizer.set_prefix_tokens(language=language, task=task, predict_timestamps=predict_timestamps)
                token_ids = tokenizer(input_str).input_ids

            all_token_ids_unprompted.append(token_ids)
            # check whether to condition on previous text - we do this with probability condition_on_prev_probability
            condition_on_prev = bool(np.random.binomial(1, condition_on_prev_probability))
            if condition_on_prev and len(all_token_ids_unprompted) > 1:
                # prompt ids are the penultimate token ids in the batch
                prompt_ids = all_token_ids_unprompted[-2]
                # strip timestamp tokens from prompt
                prompt_ids = [token for token in prompt_ids if token < timestamp_begin]
                if len(prompt_ids) > 0:
                    # remove the standard task tokens and add the special <|startofprev|> token
                    prompt_ids = [decoder_prev_token_id] + prompt_ids[timestamp_position:-1]
                if len(prompt_ids + token_ids) < max_label_length:
                    token_ids = prompt_ids + token_ids
            all_token_ids.append(token_ids)

        batch["labels"] = all_token_ids
        return batch

    def prepare_eval_dataset(batch):
        # process audio input
        sample = batch["audio"]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch["input_features"] = inputs.input_features[0]
        batch["input_length"] = len(sample["array"])

        # process targets - for evaluation these are the ground-truth transcriptions
        input_str = batch["text"]
        batch["labels"] = tokenizer(input_str).input_ids
        return batch

    vectorized_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()
    if training_args.do_train:
        # with streaming mode we can only have 1 worker, whereas with non-streaming
        # we can use `num_workers` (which is much faster)
        # We gate the pre-processing function accordingly
        map_fn_train = partial(
            raw_datasets["train"].map,
            function=prepare_train_dataset,
            remove_columns=raw_datasets_train_features,
            batched=True,
            batch_size=data_args.preprocessing_batch_size,
        )
        vectorized_datasets["train"] = (
            map_fn_train(num_proc=num_workers, desc="preprocess train dataset")
            if not data_args.streaming
            else map_fn_train()
        )
    if training_args.do_eval:
        for eval_split in all_eval_splits:
            raw_datasets_eval_features = list(raw_datasets[eval_split].features.keys())
            map_fn_eval = partial(
                raw_datasets[eval_split].map, function=prepare_eval_dataset, remove_columns=raw_datasets_eval_features
            )
            if accelerator.is_main_process:
                vectorized_datasets[eval_split] = (
                    map_fn_eval(num_proc=num_workers, desc="preprocess eval dataset")
                    if not data_args.streaming
                    else map_fn_eval()
                )

    # 10.5: Filter training data with inputs longer than `max_input_length`
    def is_audio_in_length_range(length):
        return min_input_length < length < max_input_length

    filter_by_audio_fn = partial(
        vectorized_datasets.filter, function=is_audio_in_length_range, input_columns=["input_length"]
    )
    vectorized_datasets = (
        filter_by_audio_fn(num_proc=num_workers, desc="filtering train dataset by audio length")
        if not data_args.streaming
        else filter_by_audio_fn()
    )

    # 10.6: Filter training data with labels longer than `max_label_length`
    def is_labels_in_length_range(labels):
        return 0 < len(labels) <= max_label_length

    filter_by_labels_fn = partial(
        vectorized_datasets.filter, function=is_labels_in_length_range, input_columns=["labels"]
    )
    vectorized_datasets = (
        filter_by_labels_fn(num_proc=num_workers, desc="filtering train dataset")
        if not data_args.streaming
        else filter_by_labels_fn()
    )

    # Pre-processing complete!
    # For large datasets it is advised to run the preprocessing on a
    # single machine first with `--preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step, `--preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        if data_args.streaming:
            raise ValueError(
                "When using streaming mode, dataset pre-processing is performed on the fly, hence there is no notion"
                "of a cached pre-processed dataset. Remove the argument `--preprocessing_only` to run pre-processing "
                "on the fly with streaming mode."
            )
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        return

    # 11. Define Evaluation Metrics
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
    # Store some constants
    per_device_train_batch_size = int(training_args.per_device_train_batch_size)
    train_batch_size = per_device_train_batch_size * accelerator.num_processes
    gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)

    if not data_args.streaming and training_args.max_steps < 0:
        num_epochs = int(training_args.num_train_epochs)
        steps_per_epoch = len(vectorized_datasets["train"]) // (train_batch_size * gradient_accumulation_steps)
        total_train_steps = steps_per_epoch * num_epochs
    elif training_args.max_steps > 0:
        logger.info("max_steps is given, it will override any value given in num_train_epochs")
        total_train_steps = int(training_args.max_steps)
        # Setting a very large number of epochs so we go as many times as necessary over the iterator.
        num_epochs = sys.maxsize
        steps_per_epoch = total_train_steps
    else:
        raise ValueError("max_steps must be specified when training with a streaming (iterable) dataset")

    if training_args.eval_steps is None:
        logger.info(
            f"eval_steps is not set, evaluating at the end of {'each epoch' if not data_args.streaming else 'training'}"
        )
        eval_steps = steps_per_epoch
    else:
        eval_steps = training_args.eval_steps

    # 13. Define optimizer, LR scheduler, collator
    decay_parameters = get_parameter_names(
        student_model,
        [nn.LayerNorm],
        forbidden_module=[student_model.model.encoder] if training_args.freeze_encoder else None,
    )
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [param for name, param in student_model.named_parameters() if name in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [param for name, param in student_model.named_parameters() if name not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )

    # LR scheduler gets stepped by `num_processes` each time -> account for this in warmup / total steps
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps * accelerator.num_processes,
        num_training_steps=total_train_steps * accelerator.num_processes,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=decoder_start_token_id,
        decoder_prev_token_id=decoder_prev_token_id,
        input_padding="longest",
        target_padding="max_length",
        max_target_length=max_label_length,
    )

    # 14. Define generation arguments - we need to do this before we wrap the models in DDP
    # so that we can still access the configs
    num_beams = (
        training_args.generation_num_beams
        if training_args.generation_num_beams is not None
        else getattr(student_model.generation_config, "num_beams", 1)
    )

    gen_kwargs = {
        "max_length": max_label_length,
        "num_beams": num_beams,
        "return_timestamps": return_timestamps,
    }
    if hasattr(teacher_model.generation_config, "is_multilingual") and teacher_model.generation_config.is_multilingual:
        # forcing the language and task tokens helps multilingual models in their generations
        gen_kwargs.update(
            {
                "language": data_args.language,
                "task": data_args.task,
            }
        )

    # 15. Prepare everything with accelerate
    student_model, teacher_model, optimizer, lr_scheduler = accelerator.prepare(
        student_model, teacher_model, optimizer, lr_scheduler
    )

    def kl_divergence(target_distribution, log_predicted_distribution, labels):
        kl_loss = nn.KLDivLoss(reduction="none")
        divergence = kl_loss(log_predicted_distribution, target_distribution)
        # ignore padded tokens from divergence, i.e. where labels are not set to -100
        padding_mask = labels >= 0
        padding_mask = padding_mask.unsqueeze(-1)
        divergence = divergence * padding_mask
        # take the average over the mini-batch
        divergence = divergence.sum() / padding_mask.sum()
        return divergence

    # Define gradient update step fn
    def train_step(
        batch,
        temperature=2.0,
    ):
        student_model.train()
        teacher_model.eval()

        student_outputs = student_model(**batch)
        with torch.no_grad():
            if share_hidden_states:
                # if the student and teacher share the same frozen encoder then we don't have to recompute the
                # encoder hidden-states for the teacher model, we can just re-use from the student
                encoder_outputs = BaseModelOutput(student_outputs.encoder_last_hidden_state)
                teacher_outputs = teacher_model(encoder_outputs=encoder_outputs, labels=batch["labels"])
            else:
                # do the full forward pass for the teacher model (encoder + decoder)
                teacher_outputs = teacher_model(**batch)

        # CE (data) loss
        ce_loss = student_outputs.loss
        # rescale distribution by temperature to ensure gradients scale correctly
        teacher_distribution = nn.functional.softmax(teacher_outputs.logits / temperature, dim=-1)
        # log softmax of student predictions for numerical stability
        student_distribution = nn.functional.log_softmax(student_outputs.logits / temperature, dim=-1)
        # KL-divergence loss (scaled by temperature)
        kl_loss = kl_divergence(teacher_distribution, student_distribution, batch["labels"]) * temperature**2

        # use Distil-Whisper formulation (fix weight of CE loss and tune KL weight)
        loss = 0.8 * ce_loss + training_args.kl_weight * kl_loss
        metrics = {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss}
        return loss, metrics

    # Define eval fn
    def eval_step(batch):
        student_model.eval()
        teacher_model.eval()

        with torch.no_grad():
            student_outputs = student_model(**batch)
            if share_hidden_states:
                encoder_outputs = BaseModelOutput(student_outputs.encoder_last_hidden_state)
                teacher_outputs = teacher_model(encoder_outputs=encoder_outputs, labels=batch["labels"])
            else:
                teacher_outputs = teacher_model(**batch)

        # CE (data) loss
        ce_loss = student_outputs.loss

        # log softmax / softmax for numerical stability
        student_distribution = nn.functional.log_softmax(student_outputs.logits, dim=-1)
        teacher_distribution = nn.functional.softmax(teacher_outputs.logits, dim=-1)
        # temperature is always 1 for eval
        kl_loss = kl_divergence(teacher_distribution, student_distribution, batch["labels"])

        # use Distil-Whisper formulation (fix weight of CE loss and tune KL weight)
        loss = 0.8 * ce_loss + training_args.kl_weight * kl_loss
        metrics = {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss}
        return metrics

    def generate_step(batch):
        student_model.eval()
        output_ids = accelerator.unwrap_model(student_model).generate(batch["input_features"], **gen_kwargs)
        output_ids = accelerator.pad_across_processes(output_ids, dim=1, pad_index=tokenizer.pad_token_id)
        return output_ids

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {total_train_steps * train_batch_size * gradient_accumulation_steps}")
    logger.info("  Instantaneous batch size per device =" f" {training_args.per_device_train_batch_size}")
    logger.info("  Gradient accumulation steps =" f" {gradient_accumulation_steps}")
    logger.info(
        f"  Total train batch size (w. parallel & distributed) = {train_batch_size * gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {total_train_steps}")

    # ======================== Training ================================
    train_time = 0
    train_start = time.time()
    steps_trained_progress_bar = tqdm(
        range(total_train_steps), desc="Train steps ... ", position=0, disable=not accelerator.is_local_main_process
    )
    continue_training = True
    epochs_trained = 0
    cur_step = 0

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if checkpoint is not None:
        accelerator.load_state(checkpoint)
        # Find num steps and epoch from saved state string pattern
        pattern = r"checkpoint-(\d+)-epoch-(\d+)"
        match = re.search(pattern, checkpoint)
        cur_step = int(match.group(1))
        epochs_trained = int(match.group(2))

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {cur_step}")

        steps_trained_progress_bar.update(cur_step)

        for epoch in range(0, epochs_trained):
            vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(training_args.seed)

        if not data_args.streaming and training_args.max_steps < 0:
            # we know exactly the number of steps per epoch, so can skip through the required number of batches
            resume_step = (cur_step - epochs_trained * steps_per_epoch) * gradient_accumulation_steps
        else:
            # Currently we don't know how many steps we've taken in the current epoch
            # So we just shuffle the dataset one extra time and start from a fresh epoch
            # This is "good enough" for our purposes but not fully correct
            resume_step = None
            vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(training_args.seed)
    else:
        resume_step = None

    for epoch in range(epochs_trained, num_epochs):
        vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(training_args.seed)
        train_dataloader = DataLoader(
            vectorized_datasets["train"],
            collate_fn=data_collator,
            batch_size=per_device_train_batch_size,
            num_workers=dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )
        train_dataloader = accelerator.prepare(train_dataloader)
        if hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(epoch)

        if resume_step is not None:
            # Skip the first N batches in the dataloader when resuming from a checkpoint
            train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            resume_step = None

        for batch in train_dataloader:
            with accelerator.accumulate(student_model):
                loss, train_metric = train_step(batch, temperature=training_args.temperature)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student_model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Check if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                steps_trained_progress_bar.update(1)
                cur_step += 1

                if cur_step % training_args.logging_steps == 0:
                    steps_trained_progress_bar.write(
                        f"Step... ({cur_step} / {total_train_steps} | Loss:"
                        f" {train_metric['loss']}, Learning Rate:"
                        f" {lr_scheduler.get_last_lr()[0]})"
                    )
                    log_metric(
                        accelerator,
                        metrics=train_metric,
                        learning_rate=lr_scheduler.get_last_lr()[0],
                        train_time=train_time + time.time() - train_start,
                        step=cur_step,
                        epoch=epoch,
                        prefix="train",
                    )

                # save checkpoint and weights after each save_steps and at the end of training
                if (cur_step % training_args.save_steps == 0) or cur_step == total_train_steps:
                    intermediate_dir = os.path.join(training_args.output_dir, f"checkpoint-{cur_step}-epoch-{epoch}")
                    accelerator.save_state(output_dir=intermediate_dir)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        rotate_checkpoints(training_args.save_total_limit, output_dir=training_args.output_dir)

                        if cur_step == total_train_steps:
                            # un-wrap student model for save
                            student_model = accelerator.unwrap_model(student_model)
                            student_model.save_pretrained(training_args.output_dir)
                            # re-wrap student model for final eval
                            student_model = accelerator.prepare(student_model)

                        if training_args.push_to_hub:
                            repo.push_to_hub(
                                commit_message=f"Saving train state of step {cur_step}",
                                blocking=False,
                            )

                if training_args.do_eval and (cur_step % eval_steps == 0 or cur_step == total_train_steps):
                    train_time += time.time() - train_start
                    student_model.eval()
                    # ======================== Evaluating ==============================
                    for eval_split in all_eval_splits:
                        eval_metrics = []
                        eval_preds = []
                        eval_labels = []
                        eval_start = time.time()

                        validation_dataloader = DataLoader(
                            vectorized_datasets[eval_split],
                            collate_fn=data_collator,
                            batch_size=per_device_eval_batch_size,
                            drop_last=False,
                            num_workers=dataloader_num_workers,
                            pin_memory=training_args.dataloader_pin_memory,
                        )
                        validation_dataloader = accelerator.prepare(validation_dataloader)

                        for batch in tqdm(
                            validation_dataloader,
                            desc=f"Evaluating {eval_split}...",
                            position=2,
                            disable=not accelerator.is_local_main_process,
                        ):
                            # Model forward
                            eval_metric = eval_step(batch)
                            eval_metric = accelerator.gather_for_metrics(eval_metric)
                            eval_metrics.append(eval_metric)

                            # generation
                            if training_args.predict_with_generate:
                                generated_ids = generate_step(batch)
                                # Gather all predictions and targets
                                generated_ids, labels = accelerator.gather_for_metrics(
                                    (generated_ids, batch["labels"])
                                )
                                eval_preds.extend(generated_ids)
                                eval_labels.extend(labels)

                        eval_time = time.time() - eval_start
                        # normalize eval metrics
                        eval_metrics = {
                            key: torch.mean(torch.stack([d[key] for d in eval_metrics])) for key in eval_metrics[0]
                        }

                        # compute WER metric
                        wer_desc = ""
                        if training_args.predict_with_generate:
                            wer_metric, pred_str, label_str, norm_pred_str, norm_label_str = compute_metrics(
                                eval_preds, eval_labels
                            )
                            eval_metrics.update(wer_metric)
                            wer_desc = " ".join([f"Eval {key}: {value} |" for key, value in wer_metric.items()])
                            log_pred(
                                accelerator,
                                pred_str,
                                label_str,
                                norm_pred_str,
                                norm_label_str,
                                step=cur_step,
                                prefix=eval_split,
                            )

                        # Print metrics and update progress bar
                        steps_trained_progress_bar.write(
                            f"Eval results for step ({cur_step} / {total_train_steps} | Eval Loss: {eval_metrics['loss']} |"
                            f" {wer_desc})"
                        )

                        log_metric(
                            accelerator,
                            metrics=eval_metrics,
                            train_time=eval_time,
                            step=cur_step,
                            epoch=epoch,
                            prefix=eval_split,
                        )

                    # flush the train metrics
                    train_start = time.time()

                # break condition
                if cur_step == total_train_steps:
                    continue_training = False
                    break

        if not continue_training:
            break

    accelerator.end_training()


if __name__ == "__main__":
    main()
