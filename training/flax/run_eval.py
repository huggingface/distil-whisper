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
Evaluating a Whisper model on one or more evaluation datasets.
"""
# You can also adapt this script for your own speech recognition validation. Pointers for this are left as comments.

import logging
import os
import string
import sys
import time
from dataclasses import field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
import evaluate
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import transformers
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset
from flax import jax_utils
from flax.jax_utils import pad_shard_unpad
from flax.training.common_utils import get_metrics, onehot
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperTokenizerFast,
    is_tensorboard_available,
    is_wandb_available,
)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from distil_whisper import FlaxWhisperForConditionalGeneration


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

require_version(
    "datasets>=1.18.0",
    "To fix: pip install -r examples/flax/speech-recogintion/requirements.txt",
)

logger = logging.getLogger(__name__)


@flax.struct.dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": ("Path to pretrained model or model identifier from huggingface.co/models")}
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
        metadata={"help": ("Where to store the pretrained models downloaded from huggingface.co")},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": ("Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.")},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": ("The specific model version to use (can be a branch name, tag name or commit id).")},
    )
    subfolder: str = field(
        default="",
        metadata={
            "help": "In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can"
            "specify the folder name here."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login`"
                " (necessary to use this script with private models)."
            )
        },
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized"
                " and trained. Choose one of `[float32, float16, bfloat16]`."
            )
        },
    )
    load_with_scan: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to load the model with scan enabled. Required when the model was saved with scan enabled"
            )
        },
    )
    return_timestamps: bool = field(
        default=False, metadata={"help": "Whether or not to predict timestamps in the generation step."}
    )


@flax.struct.dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset hours by a '+' symbol."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    dataset_split_name: Optional[str] = field(
        default=None,
        metadata={"help": "The split name of the dataset to use (via the datasets library)."},
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
        default=None,
        metadata={"help": "The name of the dataset column containing the text data. Defaults to `text`."},
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
    wandb_project: str = field(
        default="distil-whisper",
        metadata={"help": "The name of the wandb project."},
    )
    wandb_name: str = field(
        default=None,
        metadata={"help": "The name of the wandb run."},
    )
    wandb_job_type: str = field(
        default="distil-whisper",
        metadata={"help": "The name of the wandb job type."},
    )
    wandb_dir: str = field(
        default=None,
        metadata={"help": "The absolute path to save the wandb logs."},
    )
    save_code_to_wandb: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to save main script to wandb. This is valuable for improving"
                " experiment reproducibility and to diff code across experiments in"
                " the UI."
            )
        },
    )
    streaming: bool = field(
        default=True,
        metadata={"help": "Whether to use Datasets' streaming mode to load and the data."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of eval examples to this value if set."},
    )
    log_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "For debugging purposes, record the audio samples as well as the ground truths / preds."},
    )


def shift_tokens_right(label_ids: np.array, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift label ids one token to the right.
    """
    shifted_label_ids = np.zeros_like(label_ids)
    shifted_label_ids[:, 1:] = label_ids[:, :-1]
    shifted_label_ids[:, 0] = decoder_start_token_id

    return shifted_label_ids


@flax.struct.dataclass
class FlaxDataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`Wav2Vec2Processor`])
            The processor used for proccessing the data.
        decoder_start_token_id (:obj: `int`)
            The begin-of-sentence of the decoder.
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
        log_audio (:obj:`bool`):
            Whether we're logging audio samples as part of our eval. If so, will forward on the audio samples to the batch.
         audio_column_name (:obj:`str`):
            Name of the audio column in the dataset. Only relevant if logging audio samples.
    """

    processor: Any
    decoder_start_token_id: int
    input_padding: Union[bool, str] = "max_length"
    target_padding: Union[bool, str] = "max_length"
    max_target_length: Optional[int] = None
    log_audio: Optional[bool] = False
    audio_column_name: Optional[str] = "audio"

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
            return_tensors="np",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            max_length=self.max_target_length,
            padding=self.target_padding,
            return_tensors="np",
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        labels = labels_batch["input_ids"]
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
            labels_batch.attention_mask = labels_batch.attention_mask[:, 1:]

        decoder_input_ids = shift_tokens_right(labels, self.decoder_start_token_id)

        # replace padding with -100 to ignore correctly when computing the loss
        labels = np.ma.array(labels, mask=np.not_equal(labels_batch.attention_mask, 1))
        labels = labels.filled(fill_value=-100)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids

        if self.log_audio:
            audio_samples = [feature[self.audio_column_name] for feature in features]
            batch["audio"] = audio_samples

        return batch


def get_data_loader(
    dataset: Dataset,
    batch_size: int,
    data_collator: FlaxDataCollatorSpeechSeq2SeqWithPadding,
    dataloader_num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Returns batches of size `batch_size` from `dataset`. If `drop_last` is set to `False`, the final batch may be incomplete,
    and range in size from 1 to `batch_size`. Shuffle batches if `shuffle` is `True`.

    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int): how many samples per batch to load.
        data_collator (FlaxDataCollatorSpeechSeq2SeqWithPadding, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a map-style dataset.
        dataloader_num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
    """

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=pin_memory,
        collate_fn=data_collator,
        num_workers=dataloader_num_workers,
    )

    return data_loader


def write_metric(summary_writer, eval_metrics, step, prefix="eval"):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"{prefix}/{metric_name}", value, step)


def write_wandb_metric(wandb_logger, metrics, train_time, prefix):
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    wandb_logger.log(log_metrics)  # TODO(SG): bug with wandb means we can't log the step count


def convert_audio_to_wandb(wandb_logger, audio):
    return wandb_logger.Audio(audio["array"][:, np.newaxis], sample_rate=audio["sampling_rate"])


def write_wandb_pred(
    wandb_logger,
    eval_audios,
    pred_str,
    label_str,
    norm_pred_str,
    norm_label_str,
    prefix="eval",
    num_lines=200000,
):
    columns = ["Target", "Pred", "Norm Target", "Norm Pred"]
    # convert str data to a wandb compatible format
    str_data = [[label_str[i], pred_str[i], norm_label_str[i], norm_pred_str[i]] for i in range(len(pred_str))]

    if len(eval_audios) > 0:
        columns.insert(0, "Audio")
        str_data = [
            [
                convert_audio_to_wandb(wandb_logger, eval_audios[i]),
                *str_data[i],
            ]
            for i in range(len(pred_str))
        ]

    # log as a table with the appropriate headers
    wandb_logger.log(
        {f"{prefix}/all_predictions": wandb_logger.Table(columns=columns, data=str_data[:num_lines])},
    )
    # log incorrect normalised predictions
    str_data = np.asarray(str_data)
    str_data_incorrect = str_data[str_data[:, -2] != str_data[:, -1]]
    # log as a table with the appropriate headers
    wandb_logger.log(
        {f"{prefix}/incorrect_predictions": wandb_logger.Table(columns=columns, data=str_data_incorrect[:num_lines])},
    )


def convert_dataset_str_to_list(
    dataset_names, dataset_config_names, splits=None, text_column_names=None, dataset_hours=None, default_split="train"
):
    if isinstance(dataset_names, str):
        dataset_names = dataset_names.split("+")

        # we assume that all the datasets we're using derive from the distil-whisper org on the Hub - prepend the org name if necessary
        for i in range(len(dataset_names)):
            ds_name = dataset_names[i]
            dataset_names[i] = f"distil-whisper/{ds_name}" if "/" not in ds_name else ds_name

        dataset_config_names = dataset_config_names.split("+")
        splits = splits.split("+") if splits is not None else None
        text_column_names = text_column_names.split("+") if text_column_names is not None else None
        dataset_hours = dataset_hours.split("+") if dataset_hours is not None else None

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

    if dataset_hours is not None:
        if len(dataset_hours) != len(dataset_names):
            raise ValueError(
                f"Ensure one probability is passed for each dataset, got {len(dataset_names)} datasets and "
                f"{len(dataset_hours)} hours."
            )
        dataset_hours = [float(ds_hours) for ds_hours in dataset_hours]
    else:
        dataset_hours = [None] * len(dataset_names)

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
                "hours": dataset_hours[i],
            }
        )
    return dataset_names_dict


class FlaxWhisperFeatureExtractor(WhisperFeatureExtractor):
    def _np_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        """
        Compute the log-mel spectrogram of the provided audio using torch filters. Using the torch implementation
        computes stft filter banks approx 5x faster than its numpy counterpart, which is the native implementation
        in transformers, and matches to within 1e-5 abs tolerance.
        """
        waveform = torch.from_numpy(waveform).type(torch.float32)

        window = torch.hann_window(self.n_fft)
        stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = torch.from_numpy(self.mel_filters).type(torch.float32)
        mel_spec = mel_filters.T @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.numpy()


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your JAX/Flax versions.
    send_example_telemetry("run_flax_speech_recognition_seq2seq", model_args, data_args, framework="flax")

    # 2. Setup logging
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Set the verbosity to info of the Transformers logger.
    # We only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    logger.info("Evaluation parameters %s", training_args)

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if "tensorboard" in training_args.report_to:
        if has_tensorboard and jax.process_index() == 0:
            try:
                from flax.metrics.tensorboard import SummaryWriter

                summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
            except ImportError as ie:
                has_tensorboard = False
                logger.warning(
                    "Unable to display metrics through TensorBoard because some" f" package are not installed: {ie}"
                )
        else:
            logger.warning(
                "Unable to display metrics through TensorBoard because the package is"
                " not installed: Please run `pip install tensorboard` to enable."
            )

    # Enable wandb only on the master node
    has_wandb = is_wandb_available()
    if "wandb" in training_args.report_to:
        if has_wandb and jax.process_index() == 0:
            import wandb as wandb_logger

            # Set up wandb run
            wandb_logger.init(
                project=data_args.wandb_project,
                name=data_args.wandb_name,
                job_type=data_args.wandb_job_type,
                dir=data_args.wandb_dir,
                save_code=data_args.save_code_to_wandb,
            )
        else:
            logger.warning("Wandb logging requires wandb to be installed. Run `pip install wandb` to enable.")

    # 3. Load dataset
    raw_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()

    # Convert lists of dataset names/configs/splits to a dict
    # names: "librispeech_asr+gigaspeech", configs: "all+l", splits: "validation.clean+validation"
    # -> [{"name: "librispeech_asr": "config": "all", "split": "validation.clean"}, {"name: "gigaspeech": "config": "l", "split": "validation"}
    dataset_names_dict = convert_dataset_str_to_list(
        data_args.dataset_name,
        data_args.dataset_config_name,
        splits=data_args.dataset_split_name,
        text_column_names=data_args.text_column_name,
    )

    if len(dataset_names_dict) == 1:
        # load a single eval set
        dataset_dict = dataset_names_dict[0]
        raw_datasets["eval"] = load_dataset(
            dataset_dict["name"],
            dataset_dict["config"],
            split=dataset_dict["split"],
            cache_dir=data_args.dataset_cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
        if dataset_dict["text_column_name"] not in list(raw_datasets["eval"].features.keys()):
            raise ValueError(
                f"--text column name {dataset_dict['text_column_name']} not found in the evaluation "
                f"dataset {dataset_dict['name']}. Ensure `text_column_name` is set to the correct column "
                f"for the target text. Should be one of {' '.join(list(raw_datasets['eval'].features.keys()))}"
            )
        if dataset_dict["text_column_name"] != "text":
            raw_datasets["eval"] = raw_datasets["eval"].rename_column(dataset_dict["text_column_name"], "text")
    else:
        # load multiple eval sets
        for dataset_dict in tqdm(dataset_names_dict, desc="Loading datasets..."):
            # Clean-up the dataset name for pretty logging
            # ("distil-whisper/librispeech_asr", "validation.clean") -> "librispeech_asr/validation-clean"
            pretty_name = f"{dataset_dict['name'].split('/')[-1]}/{dataset_dict['split'].replace('.', '-')}"
            raw_datasets[pretty_name] = load_dataset(
                dataset_dict["name"],
                dataset_dict["config"],
                split=dataset_dict["split"],
                cache_dir=data_args.dataset_cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
            if dataset_dict["text_column_name"] not in list(raw_datasets[pretty_name].features.keys()):
                raise ValueError(
                    f"`--text_column_name` {dataset_dict['text_column_name']} not found in the evaluation "
                    f"dataset {dataset_dict['name']}. Ensure `text_column_name` is set to the correct column "
                    f"for the target text. Should be one of {' '.join(list(raw_datasets[pretty_name].features.keys()))}"
                )
            if dataset_dict["text_column_name"] != "text":
                raw_datasets[pretty_name] = raw_datasets[pretty_name].rename_column(
                    dataset_dict["text_column_name"], "text"
                )

    # 5. Load pretrained model, tokenizer, and feature extractor
    config = WhisperConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    feature_extractor = FlaxWhisperFeatureExtractor.from_pretrained(
        (model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = WhisperTokenizerFast.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    processor = WhisperProcessor.from_pretrained(
        (model_args.processor_name if model_args.processor_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        dtype=getattr(jnp, model_args.dtype),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        _do_init=False,
        subfolder=model_args.subfolder,
        # use_scan=model_args.load_with_scan,  # Model might have (erroneously) been saved with scan still enabled
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # disable scan if necessary (makes the inference step faster)
    if model_args.load_with_scan:
        model.disable_scan()  # to disable scan in the nn.Module
        params = model.convert_scan_to_unroll(params)  # to convert the scan params to unrolled

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
    normalizer = EnglishTextNormalizer(tokenizer.english_spelling_normalizer)

    if data_args.max_eval_samples is not None:
        for split in raw_datasets:
            raw_datasets[split] = (
                raw_datasets[split].take(data_args.max_eval_samples)
                if data_args.streaming
                else raw_datasets[split].select(range(data_args.max_eval_samples))
            )

    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]

        # process targets
        input_str = batch["text"]
        batch["labels"] = tokenizer(input_str, max_length=max_label_length, truncation=True).input_ids
        return batch

    vectorized_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()

    for split in raw_datasets:
        raw_datasets_features = list(raw_datasets[split].features.keys())
        if data_args.log_audio:
            # if logging audio samples preserve the audio column when mapping the dataset
            raw_datasets_features.remove(audio_column_name)

        map_fn = partial(
            raw_datasets[split].map,
            function=prepare_dataset,
            remove_columns=raw_datasets_features,
        )

        vectorized_datasets[split] = (
            map_fn(num_proc=num_workers, desc="preprocess eval dataset")
            if not data_args.streaming
            else map_fn()  # In streaming, we can't run multiproc - errors out if we try to
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
    # convention is that we space all punctuation *except* apostrophes
    all_punctuation = list(string.punctuation.replace("'", ""))
    return_timestamps = model_args.return_timestamps

    def compute_metrics(preds, labels):
        # replace padded labels by the padding token
        for idx in range(len(labels)):
            labels[idx][labels[idx] == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True, decode_with_timestamps=return_timestamps)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # space punctuation for orthographic WER (c.f. ESB paper https://arxiv.org/abs/2210.13352)
        spaced_pred_str = [
            pred_str[i].replace(punctuation, f" {punctuation} ")
            for punctuation in all_punctuation
            for i in range(len(pred_str))
        ]
        spaced_label_str = [
            label_str[i].replace(punctuation, f" {punctuation} ")
            for punctuation in all_punctuation
            for i in range(len(label_str))
        ]
        wer_ortho = 100 * metric.compute(predictions=spaced_pred_str, references=spaced_label_str)

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

    data_collator = FlaxDataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        input_padding="longest",
        target_padding="max_length",
        max_target_length=max_label_length,
        log_audio=data_args.log_audio,
    )

    # Store some constants
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    eval_batch_size = per_device_eval_batch_size * jax.device_count()

    # label smoothed cross entropy
    def loss_fn(logits, labels, label_smoothing_factor=0.0):
        """
        The label smoothing implementation is adapted from Flax's official example:
        https://github.com/google/flax/blob/87a211135c6a377c8f29048a1cac3840e38b9da4/examples/wmt/train.py#L104
        """
        vocab_size = logits.shape[-1]
        confidence = 1.0 - label_smoothing_factor
        low_confidence = (1.0 - confidence) / (vocab_size - 1)
        normalizing_constant = -(
            confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
        )
        soft_labels = onehot(labels, vocab_size, on_value=confidence, off_value=low_confidence)

        loss = optax.softmax_cross_entropy(logits, soft_labels)
        loss = loss - normalizing_constant

        # ignore padded tokens from loss, i.e. where labels are not set to -100
        padding_mask = labels >= 0
        loss = loss * padding_mask
        loss = loss.sum()
        num_labels = padding_mask.sum()
        return loss, num_labels

    # Define eval fn
    def eval_step(params, batch, label_smoothing_factor=0.0):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, freeze_encoder=True, train=False)[0]

        loss, num_labels = loss_fn(logits, labels, label_smoothing_factor)
        num_labels = jax.lax.psum(num_labels, "batch")

        # true loss = total loss / total samples
        loss = jax.lax.psum(loss, "batch")
        loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)

        metrics = {"loss": loss}
        return metrics

    # Define generation function
    num_beams = (
        training_args.generation_num_beams
        if training_args.generation_num_beams is not None
        else model.config.num_beams
    )

    # forcing the language and task tokens helps the flax teacher model in its generations
    gen_kwargs = {
        "max_length": max_label_length,
        "num_beams": num_beams,
        "language": "<|en|>",
        "task": "transcribe",
        "return_timestamps": return_timestamps,
    }

    def generate_step(params, batch):
        output_ids = model.generate(
            batch[model_input_name],
            attention_mask=batch.get("attention_mask"),
            params=params,
            freeze_encoder=True,
            **gen_kwargs,
        )
        return output_ids.sequences

    # Create parallel version of the eval and generate step
    p_eval_step = jax.pmap(
        partial(eval_step, label_smoothing_factor=training_args.label_smoothing_factor),
        "batch",
    )
    p_generate_step = jax.pmap(generate_step, "batch")

    # Replicate params on each device
    params = jax_utils.replicate(params)

    def eval_step(split="eval"):
        # ======================== Evaluating ==============================
        eval_metrics = []
        eval_preds = []
        eval_labels = []
        eval_audios = []
        eval_start = time.time()

        eval_loader = get_data_loader(
            vectorized_datasets[split],
            batch_size=eval_batch_size,
            data_collator=data_collator,
            dataloader_num_workers=dataloader_num_workers,
        )
        for batch in tqdm(eval_loader, desc=f"Evaluating {split}..."):
            # Model forward
            labels = batch["labels"]
            if data_args.log_audio:
                eval_audios.extend(batch.pop("audio"))

            metrics = pad_shard_unpad(p_eval_step, static_return=True)(
                params, batch.data, min_device_batch=per_device_eval_batch_size
            )
            eval_metrics.append(metrics)

            # generation
            if training_args.predict_with_generate:
                generated_ids = pad_shard_unpad(p_generate_step)(
                    params, batch.data, min_device_batch=per_device_eval_batch_size
                )
                eval_preds.extend(jax.device_get(generated_ids.reshape(-1, gen_kwargs["max_length"])))
                eval_labels.extend(labels)

        eval_time = time.time() - eval_start

        # normalize eval metrics
        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_util.tree_map(jnp.mean, eval_metrics)

        # compute WER metric
        wer_desc = ""
        if training_args.predict_with_generate:
            wer_metric, pred_str, label_str, norm_pred_str, norm_label_str = compute_metrics(eval_preds, eval_labels)
            eval_metrics.update(wer_metric)
            wer_desc = " ".join([f"Eval {key}: {value} |" for key, value in wer_metric.items()])

        # Print metrics
        logger.info(f"Eval Loss: {eval_metrics['loss']} | {wer_desc})")

        # Save metrics
        if has_tensorboard and jax.process_index() == 0 and "tensorboard" in training_args.report_to:
            write_metric(summary_writer, eval_metrics, model_args.step, prefix=split)

        if has_wandb and jax.process_index() == 0 and "wandb" in training_args.report_to:
            write_wandb_metric(wandb_logger, eval_metrics, eval_time, prefix=split)
            if training_args.predict_with_generate:
                write_wandb_pred(
                    wandb_logger, eval_audios, pred_str, label_str, norm_pred_str, norm_label_str, prefix=split
                )

    logger.info("***** Running Eval *****")
    logger.info("  Instantaneous batch size per device =" f" {training_args.per_device_eval_batch_size}")
    logger.info(f"  Total eval batch size (w. parallel & distributed) = {eval_batch_size}")
    for split in vectorized_datasets:
        eval_step(split=split)


if __name__ == "__main__":
    main()
