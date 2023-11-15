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
Fine-tuning the Whisper model for sequence to sequence speech recognition.
"""
# You can also adapt this script for your own speech recognition task. Pointers for this are left as comments.

import logging
import os
import string
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import datasets
import evaluate
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import transformers
from datasets import Dataset, DatasetDict, load_dataset
from flax import jax_utils, traverse_util
from flax.jax_utils import pad_shard_unpad, unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    is_tensorboard_available,
    is_wandb_available,
)
from transformers.file_utils import get_full_repo_name
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
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout ratio for activations inside the fully connected layer."},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout ratio for the attention probabilities."},
    )
    dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )


@flax.struct.dataclass
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
        metadata={"help": ("The configuration name of the dataset to use (via the datasets library).")},
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
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of"
                " training examples to this value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of"
                " evaluation examples to this value if set."
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": ("The name of the dataset column containing the audio data. Defaults to 'audio'")},
    )
    text_column_name: str = field(
        default="whisper_transcript",
        metadata={
            "help": (
                "The name of the dataset column containing the text data. Defaults to"
                " 'whisper_transcript'which is the pseudo-labelled Whisper"
                " transcription data."
            )
        },
    )
    max_duration_in_seconds: float = field(
        default=30.0,
        metadata={"help": ("Filter audio files that are longer than `max_duration_in_seconds` seconds")},
    )
    min_duration_in_seconds: float = field(
        default=0.0,
        metadata={"help": ("Filter audio files that are shorter than `min_duration_in_seconds` seconds")},
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
            "help": ("The name of the training data set split to use (via the datasets library). Defaults to 'train'")
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": (
                "The name of the evaluation data set split to use (via the datasets"
                " library). Defaults to 'validation'"
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
                " experimentreproducibility and to diff code across experiments in"
                " the UI."
            )
        },
    )


@dataclass
class FlaxSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    use_scan: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Whether or not to use `scan_with_axes` over the encoder and decoder"
                " blocks. Using scan results in faster compile times and more efficient"
                " memory use during training, since all of the layers in the"
                " encoder/decoder are stacked, and we perform a lax.scan over the"
                " stacked block to index each layer. However, it results in slower"
                " inference time due to the overhead of stacking the layers this way."
                " Thus, we always default to disabling scan for the inference step."
            )
        },
    )
    freeze_encoder: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to freeze the entire encoder model. Only recommended when the"
                " entire encoder has been copiedfrom the teacher model."
            )
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

        return batch


def get_data_loader(
    rng: jax.random.PRNGKey,
    dataset: Dataset,
    batch_size: int,
    data_collator: FlaxDataCollatorSpeechSeq2SeqWithPadding,
    shuffle: bool = True,
    drop_last: bool = True,
    dataloader_num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Returns batches of size `batch_size` from `dataset`. If `drop_last` is set to `False`, the final batch may be incomplete,
    and range in size from 1 to `batch_size`. Shuffle batches if `shuffle` is `True`.

    Args:
        rng (List(int)): JAX rng for generating pseudo random numbers. Used if shuffling the dataset.
        dataset (Dataset): dataset from which to load the data.
        batch_size (int): how many samples per batch to load.
        data_collator (FlaxDataCollatorSpeechSeq2SeqWithPadding, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a map-style dataset.
        shuffle (bool, optional): set to `True` to have the batches reshuffled.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        dataloader_num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.

    """
    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
        batch_idx = np.asarray(batch_idx)
        dataset = dataset.select(batch_idx)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=data_collator,
        num_workers=dataloader_num_workers,
    )

    return data_loader


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))


def write_metric(summary_writer, train_metrics, eval_metrics, train_time, step, logging_steps):
    summary_writer.scalar("train/time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        steps_arr = np.arange(0, step, logging_steps)[-len(vals) :]
        tag = f"train/{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, steps_arr[i])

    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval/{metric_name}", value, step)


def write_wandb_metric(wandb_logger, metrics, train_time, step, prefix):
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    wandb_logger.log(log_metrics, step)


def write_wandb_pred(wandb_logger, pred_str, label_str, prefix="eval", num_lines=100):
    # convert str data to a wandb compatible format
    if num_lines < len(pred_str):
        str_data = [[label_str[i], pred_str[i]] for i in range(num_lines)]
    else:
        str_data = [[label_str[i], pred_str[i]] for i in range(len(pred_str))]
    # log as a table with the appropriate headers
    wandb_logger.log(
        {f"{prefix}/predictions": wandb_logger.Table(columns=["label_str", "pred_str"], data=str_data)},
    )


def create_learning_rate_fn(
    num_train_steps: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate,
        end_value=0,
        transition_steps=num_train_steps - num_warmup_steps,
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FlaxSeq2SeqTrainingArguments))

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

    logger.info("Training/evaluation parameters %s", training_args)

    # Check the output dir is valid
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not"
            " empty.Use `--overwrite_output_dir` to overcome."
        )

    # Handle the repository creation
    if training_args.push_to_hub:
        if training_args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(training_args.output_dir).absolute().name,
                token=training_args.hub_token,
            )
        else:
            repo_name = training_args.hub_model_id
        create_repo(repo_name, exist_ok=True, token=training_args.hub_token)
        repo = Repository(
            training_args.output_dir,
            clone_from=repo_name,
            token=training_args.hub_token,
        )

    # 3. Load dataset
    raw_datasets = DatasetDict()

    if training_args.do_train:
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.train_split_name,
            cache_dir=data_args.dataset_cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            num_proc=data_args.preprocessing_num_workers,
        )

    if training_args.do_eval:
        raw_datasets["eval"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.eval_split_name,
            cache_dir=data_args.dataset_cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            num_proc=data_args.preprocessing_num_workers,
        )

    if not training_args.do_train and not training_args.do_eval:
        raise ValueError(
            "Cannot not train and not do evaluation. At least one of training or evaluation has to be performed."
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

    # 5. Load pretrained model, tokenizer, and feature extractor
    config = AutoConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        (model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config.update(
        {
            "activation_dropout": model_args.activation_dropout,
            "attention_dropout": model_args.attention_dropout,
            "dropout": model_args.dropout,
        }
    )

    model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        dtype=getattr(jnp, model_args.dtype),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        _do_init=False,
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # enable scan / gradient checkpointing if necessary
    if training_args.use_scan:
        model.enable_scan()  # to enable scan in the nn.Module
        params = model.convert_unroll_to_scan(params)  # to convert the unrolled params to scan

    if training_args.gradient_checkpointing:
        model.enable_gradient_checkpointing()  # to enable checkpointing in the nn.Module, there is no change to the params structure

    if hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual:
        # We need to set the language and task ids for previously multilingual checkpoints
        tokenizer.set_prefix_tokens(language="English", task="transcribe", predict_timestamps=False)
        model.generation_config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(
            language="English", task="transcribe", no_timestamps=True
        )

    # 6. Resample speech dataset: `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name,
        datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
    )

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = int(data_args.max_duration_in_seconds * feature_extractor.sampling_rate)
    min_input_length = int(data_args.min_duration_in_seconds * feature_extractor.sampling_rate)
    max_label_length = (
        data_args.max_label_length if data_args.max_label_length is not None else model.config.max_length
    )
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    dataloader_num_workers = training_args.dataloader_num_workers
    text_column_name = data_args.text_column_name
    model_input_name = feature_extractor.model_input_names[0]
    normalizer = EnglishTextNormalizer(tokenizer.english_spelling_normalizer)

    if training_args.do_train and data_args.max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

    if training_args.do_eval and data_args.max_eval_samples is not None:
        raw_datasets["eval"] = raw_datasets["eval"].select(range(data_args.max_eval_samples))

    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = len(sample["array"])

        # process targets
        input_str = " " + batch[text_column_name].lower()
        batch["labels"] = tokenizer(input_str).input_ids
        return batch

    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        remove_columns=next(iter(raw_datasets.values())).column_names,
        num_proc=num_workers,
        desc="preprocess train dataset",
    )

    # filter training data with inputs longer than max_input_length
    def is_audio_in_length_range(length):
        return min_input_length < length < max_input_length

    vectorized_datasets = vectorized_datasets.filter(
        is_audio_in_length_range,
        num_proc=num_workers,
        input_columns=["input_length"],
    )

    # filter training data with labels longer than max_label_length
    def is_labels_in_length_range(labels):
        return 0 < len(labels) < max_label_length

    vectorized_datasets = vectorized_datasets.filter(
        is_labels_in_length_range,
        num_proc=num_workers,
        input_columns=["labels"],
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
    all_punctuation = list(string.punctuation.replace("'", ""))

    def compute_metrics(preds, labels):
        # replace padded labels by the padding token
        for idx in range(len(labels)):
            labels[idx][labels[idx] == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # space punctuation for orthographic WER (c.f. ESB paper https://arxiv.org/abs/2210.13352)
        spaced_pred_str = [
            pred_str[i].replace(punctuation, "") for punctuation in all_punctuation for i in range(len(pred_str))
        ]
        spaced_label_str = [
            label_str[i].replace(punctuation, "") for punctuation in all_punctuation for i in range(len(label_str))
        ]
        wer_ortho = 100 * metric.compute(predictions=spaced_pred_str, references=spaced_label_str)

        # normalize everything and re-compute the WER
        norm_pred_str = [normalizer(pred) for pred in pred_str]
        norm_label_str = [normalizer(label) for label in label_str]
        # filtering step to only evaluate the samples that correspond to non-zero normalized references:
        norm_pred_str = [norm_pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
        norm_label_str = [norm_label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]

        wer = 100 * metric.compute(predictions=norm_pred_str, references=norm_label_str)

        return {"wer": wer, "wer_ortho": wer_ortho}, pred_str, label_str

    # 9. Save feature extractor, tokenizer, config and generation config
    feature_extractor.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    config.save_pretrained(training_args.output_dir)
    model.generation_config.save_pretrained(
        training_args.output_dir
    )  # generation config stays bound to model to make it easy to jit

    processor = AutoProcessor.from_pretrained(training_args.output_dir)

    data_collator = FlaxDataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        input_padding="longest",
        target_padding="max_length",
        max_target_length=max_label_length,
    )

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                "Unable to display metrics through TensorBoard because some package" f" are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not"
            " installed: Please run `pip install tensorboard` to enable."
        )

    # Enable wandb only on the master node
    has_wandb = is_wandb_available()
    if has_wandb:
        import wandb as wandb_logger

        # Set up wandb run
        if jax.process_index() == 0:
            wandb_logger.init(
                project=data_args.wandb_project,
                name=data_args.wandb_name,
                job_type=data_args.wandb_job_type,
                dir=data_args.wandb_dir,
                save_code=data_args.save_code_to_wandb,
            )
    else:
        logger.warning("Wandb logging requires wandb to be installed. Run `pip install wandb` to enable.")

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    eval_batch_size = per_device_eval_batch_size * jax.device_count()
    steps_per_epoch = len(vectorized_datasets["train"]) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    # Create learning rate schedule
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        total_train_steps,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        # find out all LayerNorm parameters
        layer_norm_candidates = [
            "layer_norm",
            "self_attn_layer_norm",
            "final_layer_norm",
            "encoder_attn_layer_norm",
        ]
        layer_norm_named_params = {
            layer[-2:]
            for layer_norm_name in layer_norm_candidates
            for layer in flat_params.keys()
            if layer_norm_name in "".join(layer).lower()
        }
        flat_mask = {path: path[-1] != "bias" and path[-2:] not in layer_norm_named_params for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    adamw = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        mask=decay_mask_fn,
    )

    # Setup train state
    state = TrainState.create(apply_fn=model.__call__, params=params, tx=adamw, dropout_rng=dropout_rng)

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

    # Define gradient update step fn
    def train_step(state, batch, freeze_encoder, label_smoothing_factor=0.0):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def compute_loss(params):
            labels = batch.pop("labels")
            logits = state.apply_fn(
                **batch,
                params=params,
                dropout_rng=dropout_rng,
                freeze_encoder=freeze_encoder,
                train=True,
            )[0]
            loss, num_labels = loss_fn(logits, labels, label_smoothing_factor)
            return loss, num_labels

        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
        (loss, num_labels), grad = grad_fn(state.params)
        num_labels = jax.lax.psum(num_labels, "batch")

        # true loss = total loss / total samples
        loss = jax.lax.psum(loss, "batch")
        loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)

        # true grad = total grad / total samples
        grad = jax.lax.psum(grad, "batch")
        grad = jax.tree_util.tree_map(lambda x: x / num_labels, grad)
        new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

        metrics = {
            "loss": loss,
            "learning_rate": linear_decay_lr_schedule_fn(state.step),
        }
        return new_state, metrics

    # Define eval fn
    def eval_step(params, batch, label_smoothing_factor=0.0):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]

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
    gen_kwargs = {"max_length": max_label_length, "num_beams": num_beams}

    def generate_step(params, batch):
        output_ids = model.generate(
            batch[model_input_name],
            attention_mask=batch.get("attention_mask"),
            params=params,
            **gen_kwargs,
        )
        return output_ids.sequences

    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(
        partial(train_step, label_smoothing_factor=training_args.label_smoothing_factor),
        "batch",
        donate_argnums=(0,),
        static_broadcasted_argnums=(2,),
    )
    p_eval_step = jax.pmap(
        partial(eval_step, label_smoothing_factor=training_args.label_smoothing_factor),
        "batch",
    )
    p_generate_step = jax.pmap(generate_step, "batch")

    # Replicate the train state on each device
    state = state.replicate()

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(vectorized_datasets['train'])}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info("  Instantaneous batch size per device =" f" {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    train_time = 0
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)
        train_metrics = []

        # Generate an epoch by shuffling sampling indices from the train dataset
        train_loader = get_data_loader(
            input_rng,
            vectorized_datasets["train"],
            batch_size=train_batch_size,
            data_collator=data_collator,
            dataloader_num_workers=dataloader_num_workers,
        )
        # train
        for step, batch in enumerate(tqdm(train_loader, desc="Training...", position=1), 1):
            batch = shard(batch.data)
            state, train_metric = p_train_step(state, batch, training_args.freeze_encoder)

            cur_step = epoch * steps_per_epoch + step
            if cur_step % training_args.logging_steps == 0:
                train_metrics.append(train_metric)
                train_metric_to_write = unreplicate(train_metric)
                epochs.write(
                    f"Step... ({cur_step} / {total_train_steps} | Loss:"
                    f" {train_metric_to_write['loss']}, Learning Rate:"
                    f" {train_metric_to_write['learning_rate']})"
                )
                if has_wandb and jax.process_index() == 0:
                    write_wandb_metric(
                        wandb_logger,
                        train_metric_to_write,
                        train_time + time.time() - train_start,
                        cur_step,
                        "train",
                    )

        train_time += time.time() - train_start

        train_metric = unreplicate(train_metric)

        epochs.write(
            f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {train_metric['loss']},"
            f" Learning Rate: {train_metric['learning_rate']})"
        )

        # ======================== Evaluating ==============================
        eval_metrics = []
        eval_preds = []
        eval_labels = []
        eval_start = time.time()

        eval_loader = get_data_loader(
            input_rng,
            vectorized_datasets["eval"],
            batch_size=eval_batch_size,
            data_collator=data_collator,
            shuffle=False,
            drop_last=False,
            dataloader_num_workers=dataloader_num_workers,
        )
        for batch in tqdm(eval_loader, desc="Evaluating...", position=2):
            # Model forward
            labels = batch["labels"]

            metrics = pad_shard_unpad(p_eval_step, static_return=True)(
                state.params, batch.data, min_device_batch=per_device_eval_batch_size
            )
            eval_metrics.append(metrics)

            # generation
            if training_args.predict_with_generate:
                generated_ids = pad_shard_unpad(p_generate_step)(
                    state.params, batch.data, min_device_batch=per_device_eval_batch_size
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
            wer_metric, pred_str, label_str = compute_metrics(eval_preds, eval_labels)
            eval_metrics.update(wer_metric)
            wer_desc = " ".join([f"Eval {key}: {value} |" for key, value in wer_metric.items()])

        # Print metrics and update progress bar
        desc = f"Epoch... ({epoch + 1}/{num_epochs} | Eval Loss: {eval_metrics['loss']} |" f" {wer_desc})"
        epochs.write(desc)
        epochs.desc = desc

        # Save metrics
        if has_tensorboard and jax.process_index() == 0:
            write_metric(
                summary_writer,
                train_metrics,
                eval_metrics,
                train_time,
                cur_step,
                training_args.logging_steps,
            )

        if has_wandb and jax.process_index() == 0:
            write_wandb_metric(wandb_logger, eval_metrics, eval_time, cur_step, "eval")
            if training_args.predict_with_generate:
                write_wandb_pred(wandb_logger, pred_str, label_str)

        # save checkpoint after each epoch and push checkpoint to the hub
        if jax.process_index() == 0:
            params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
            model.save_pretrained(training_args.output_dir, params=params)
            tokenizer.save_pretrained(training_args.output_dir)
            if training_args.push_to_hub:
                repo.push_to_hub(
                    commit_message=f"Saving weights and logs of epoch {epoch + 1}",
                    blocking=False,
                )


if __name__ == "__main__":
    main()
