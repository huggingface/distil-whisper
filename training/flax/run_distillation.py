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
import torch
import transformers
from datasets import (
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
)
from flax import jax_utils, traverse_util
from flax.jax_utils import pad_shard_unpad, unreplicate
from flax.serialization import from_bytes, to_bytes
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from huggingface_hub import Repository, create_repo
from jax.experimental.compilation_cache import compilation_cache as cc
from optax._src import linear_algebra
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper
from tqdm import tqdm
from transformers import (
    AddedToken,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperTokenizerFast,
    is_tensorboard_available,
    is_wandb_available,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.modeling_flax_outputs import FlaxBaseModelOutput
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
        metadata={"help": ("Path to pretrained student model or model identifier from huggingface.co/models")}
    )
    teacher_model_name_or_path: str = field(
        metadata={"help": ("Path to pretrained teacher model or model identifier from huggingface.co/models")}
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
    load_with_scan_weights: bool = field(
        default=False,
        metadata={
            "help": "Whether the pre-trained checkpoint has its weights stored in scan format. Set to True for scanned "
            "weights, defaults to False for non-scan (unrolled) weights."
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

    train_dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset ids by a '+' symbol. For example, to load and combine "
            " librispeech and common voice, set `train_dataset_name='librispeech_asr+common_voice'`."
        },
    )
    train_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset configs by a '+' symbol."
        },
    )
    train_dataset_samples: str = field(
        default=None,
        metadata={
            "help": "Number of samples in the training data. Load and combine "
            "multiple datasets by separating dataset samples by a '+' symbol."
        },
    )
    eval_dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the evaluation dataset to use (via the datasets library). Defaults to the training dataset name if unspecified."
        },
    )
    eval_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the evaluation dataset to use (via the datasets library). Defaults to the training dataset config name if unspecified"
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
    train_text_column_name: str = field(
        default="whisper_transcript",
        metadata={
            "help": (
                "The name of the dataset column containing the text data. Defaults to"
                " 'whisper_transcript'which is the pseudo-labelled Whisper"
                " transcription data."
            )
        },
    )
    eval_text_column_name: str = field(
        default="text",
        metadata={
            "help": (
                "The name of the dataset column containing the text data. Defaults to"
                " 'text', which is the original text data"
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
                " experiment reproducibility and to diff code across experiments in"
                " the UI."
            )
        },
    )
    streaming: bool = field(
        default=True,
        metadata={"help": "Whether to use Datasets' streaming mode to load and the data."},
    )
    wer_threshold: float = field(
        default=None,
        metadata={
            "help": "Filter training data with Whisper transcriptions that have greater than `wer_threshold` "
            "WER with the normalised transcriptions."
        },
    )
    prefetch_size: int = field(
        default=0,
        metadata={"help": "Number of samples to pre-fetch if using an iterable dataset."},
    )
    timestamp_probability: float = field(
        default=0.5, metadata={"help": "Probability for training on timestamped tokens if the data contains it."}
    )
    return_timestamps: bool = field(
        default=False, metadata={"help": "Whether or not to predict timestamps in the generation step."}
    )
    round_timestamps: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to round the timestamp tokens to the nearest tenth of a second."
            "By default, Whisper predicts timestamps to the nearest hundredth of a second."
            "Reducing the timestamp precision to one tenth of a second simplifies the timestamp"
            "prediction task, at the expense of timestamp granularity."
        },
    )


@dataclass
class FlaxSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    use_scan: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Whether or not to use `scan_with_axes` over the encoder and decoder blocks. Using scan results "
                "in faster compile times and more efficient memory use during training, since all of the layers "
                "in the encoder/decoder are stacked, and we perform a lax.scan over the stacked block to index "
                "each layer. However, it results in slower inference time due to the overhead of stacking the "
                "layers this way. Thus, we **always** default to disabling scan for the inference step."
            )
        },
    )
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
    mse_weight: Optional[float] = field(
        default=0.0,
        metadata={
            "help": (
                "Weighting assigned to the MSE loss in the KD formulation. MSE loss is "
                "computed between the teacher-student hidden states and attentions."
            )
        },
    )
    precision: Optional[str] = field(
        default="half_mixed",
        metadata={
            "help": (
                "Precision with which run training, Can be one of `full`, `half_mixed` or `full_mixed`, the latter two"
                "of which enable *mixed-precision* training. **Note that this only specifies the dtype of the computation "
                "and optimizer state. It does not influence the dtype of model parameters.** An explanation of the three "
                "settings is provided below:"
                "   1. Full precision: forward pass, backward pass and optimiser states all in float32."
                "   2. Half mixed precision: forward pass in bfloat16, backward pass and optimiser states in float32. This "
                "   corresponds to setting the dtype argument to bfloat16 when instantiating the model."
                "   3. Full mixed precision: forward pass, backward pass and optimiser states all in bfloat16. The dtype "
                "   argument is set to bfloat16 for the forward pass, and the gradients computed with respect to the bfloat16 "
                "   parameters in the backward pass (giving bfloat16 gradients). The new optimiser states and parameter "
                "   updates are computed in float32 by upcasting the bfloat16 gradients and optimiser states to float32 "
                "   prior to the optimiser update step. The optimiser states are returned in float32 (but not saved to "
                "   memory) and then downcasted to bfloat16 (saved to memory) for the subsequent train step."
                "For further details, refer to https://github.com/deepmind/optax/discussions/336"
            )
        },
    )
    compilation_cache: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable the JAX (experimental) compilation cache. The compilation step is *cached* the "
                "first time it is run. Successive compilation steps for the same function utilise the cache to reduce"
                "the compilation time."
            )
        },
    )
    save_train_state: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether or not to save the Flax Train State on each `save_steps` steps. Required if you intend"
            "to resume training from partial training runs. If False, only the model weights will be saved."
            "If True, both the model weights and Flax Train state will be saved."
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
        if set(np.unique(labels[:, 0])).issubset({self.decoder_start_token_id, self.decoder_prev_token_id}):
            decoder_input_ids = labels[:, :-1]
            labels = labels[:, 1:]
            labels_batch.attention_mask = labels_batch.attention_mask[:, 1:]
        else:
            decoder_input_ids = shift_tokens_right(labels, self.decoder_start_token_id)

        # replace padding with -100 to ignore correctly when computing the loss
        labels = np.ma.array(labels, mask=np.not_equal(labels_batch.attention_mask, 1))
        labels = labels.filled(fill_value=-100)

        # replace initial prompt tokens with -100 to ignore correctly when computing the loss
        bos_index = np.argmax(labels == self.decoder_start_token_id, axis=1)
        prompt_mask = np.arange(labels.shape[1]) < bos_index[:, None]
        labels = np.where(prompt_mask, -100, labels)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids

        return batch


def get_data_loader(
    seed: int,
    dataset: IterableDataset,
    batch_size: int,
    data_collator: FlaxDataCollatorSpeechSeq2SeqWithPadding,
    shuffle: bool = True,
    drop_last: bool = True,
    dataloader_num_workers: int = 0,
    skip_batches: int = 0,
    pin_memory: bool = True,
    prefetch_size: int = 0,
) -> DataLoader:
    """
    Returns batches of size `batch_size` from `dataset`. If `drop_last` is set to `False`, the final batch may be incomplete,
    and range in size from 1 to `batch_size`. Shuffle batches if `shuffle` is `True`.

    Args:
        seed (int): Numpy seed for generating pseudo random numbers. Used if shuffling the dataset.
        dataset (IterableDataset): streaming dataset from which to load the data.
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
        skip_batches (int, optional): Efficiently skip the first `skip_batches`.
        pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.

    """
    if shuffle:
        dataset = dataset.shuffle(seed)

    if skip_batches > 0:
        dataset = dataset.skip(skip_batches * batch_size)

    if prefetch_size > 0:
        dataset = IterableWrapper(dataset)
        dataset = dataset.prefetch(prefetch_size)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=data_collator,
        num_workers=dataloader_num_workers,
    )

    return data_loader


def sorted_checkpoints(output_dir=None, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
            if regex_match is not None and regex_match.groups() is not None:
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def rotate_checkpoints(
    save_total_limit=None, use_mtime=False, output_dir=None, checkpoint_prefix="checkpoint"
) -> None:
    if save_total_limit is None or save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(
        use_mtime=use_mtime, output_dir=output_dir, checkpoint_prefix=checkpoint_prefix
    )
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        shutil.rmtree(checkpoint, ignore_errors=True)


def to_fp32(t):
    return jax.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, t)


def to_bf16(t):
    return jax.tree_map(lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, t)


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray
    max_grad_norm: float

    def apply_gradients(self, *, grads, to_dtype: to_fp32, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value, clipping the
        gradients by the maximum grad norm.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
          grads: Gradients that have the same pytree structure as `.params`.
          **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
          An updated instance of `self` with `step` incremented by one, `params`
          and `opt_state` updated by applying `grads`, and additional attributes
          replaced as specified by `kwargs`.
        """
        # clip gradients by global l2 norm
        casted_max_grad_norm = to_dtype(self.max_grad_norm)
        g_norm = linear_algebra.global_norm(grads)
        g_norm = jnp.maximum(casted_max_grad_norm, g_norm)
        grads = jax.tree_map(lambda t: (t / g_norm) * casted_max_grad_norm, grads)

        # perform update step in fp32 and subsequently downcast optimizer states if mixed precision training
        # grads and opt_state in bf16 (need to upcast), params in fp32 (leave as is)
        updates, new_opt_state = self.tx.update(to_fp32(grads), to_fp32(self.opt_state), self.params)

        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=to_dtype(new_opt_state),
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, to_dtype: to_fp32, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        # downcast optimizer state to bf16 if mixed-precision training
        opt_state = tx.init(to_dtype(params))
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))

    def unreplicate(self):
        return jax_utils.unreplicate(self)

    def save_state(self, output_dir, save_total_limit=None, checkpoint_prefix="checkpoint"):
        step = int(jax.device_get(unreplicate(self.step)))
        serialized_state = to_bytes(self.unreplicate())

        output_file = Path(os.path.join(output_dir, f"{checkpoint_prefix}-{step}", "train_state.msgpack"))
        output_file.parent.mkdir(exist_ok=True, parents=True)

        with output_file.open("wb") as f:
            f.write(serialized_state)

        logger.info(f"Flax train state saved in {output_file}")
        rotate_checkpoints(
            save_total_limit=save_total_limit, output_dir=output_dir, checkpoint_prefix=checkpoint_prefix
        )


def save_hf_weights(
    student_state: TrainState,
    student_model: FlaxWhisperForConditionalGeneration,
    processor: WhisperProcessor,
    output_dir: str,
    cur_step: int,
    total_train_steps: int,
    use_scan: bool = True,
    checkpoint_prefix: str = "checkpoint",
) -> None:
    # always disable scan in the params / model so that we can load from PyTorch directly - this is a no-op if we're not using scan for training
    student_state_params = unreplicate(student_state.params)
    student_state_params = student_model.convert_scan_to_unroll(student_state_params)
    student_params = jax.device_get(student_state_params)
    student_model.disable_scan()

    if cur_step != total_train_steps:
        output_dir = os.path.join(output_dir, f"{checkpoint_prefix}-{cur_step}")
        os.makedirs(output_dir, exist_ok=True)

    student_model.save_pretrained(output_dir, params=student_params)
    processor.save_pretrained(output_dir)

    # re-enable scan only if required for training
    if use_scan:
        student_model.enable_scan()


def write_train_metric(summary_writer, train_metrics, train_time, step, logging_steps):
    summary_writer.scalar("train/time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        steps_arr = np.arange(0, step, logging_steps)[-len(vals) :]
        tag = f"train/{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, steps_arr[i])


def write_eval_metric(summary_writer, eval_metrics, step, prefix="eval"):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"{prefix}/{metric_name}", value, step)


def write_wandb_metric(wandb_logger, metrics, train_time, step, epoch, prefix="train"):
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    log_metrics[f"{prefix}/epoch"] = epoch
    wandb_logger.log(log_metrics, step)


def write_wandb_pred(
    wandb_logger, pred_str, label_str, norm_pred_str, norm_label_str, cur_step, prefix="eval", num_lines=200000
):
    # pretty name for current step: step 50000 -> step 50k
    cur_step_pretty = f"{int(cur_step // 1000)}k" if cur_step > 1000 else cur_step
    # convert str data to a wandb compatible format
    str_data = [[label_str[i], pred_str[i], norm_label_str[i], norm_pred_str[i]] for i in range(len(pred_str))]
    # log as a table with the appropriate headers
    wandb_logger.log(
        {
            f"predictions/{prefix.replace('/', '-')}-step-{cur_step_pretty}": wandb_logger.Table(
                columns=["Target", "Pred", "Norm Target", "Norm Pred"], data=str_data[:num_lines]
            )
        },
        cur_step,
    )
    # log incorrect normalised predictions
    str_data = np.asarray(str_data)
    str_data_incorrect = str_data[str_data[:, -2] != str_data[:, -1]]
    # log as a table with the appropriate headers
    wandb_logger.log(
        {
            f"incorrect_predictions/{prefix.replace('/', '-')}-step-{cur_step_pretty}": wandb_logger.Table(
                columns=["Target", "Pred", "Norm Target", "Norm Pred"], data=str_data_incorrect[:num_lines]
            )
        },
        cur_step,
    )


def create_learning_rate_fn(
    num_train_steps: int, lr_scheduler_type: str, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    lr_scheduler_types = ("linear", "constant_with_warmup")

    if lr_scheduler_type not in lr_scheduler_types:
        raise ValueError(
            f"lr_scheduler_type of type {lr_scheduler_type} not supported, choose from {lr_scheduler_types}."
        )

    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate,
        end_value=0 if lr_scheduler_type == "linear" else learning_rate,
        transition_steps=num_train_steps - num_warmup_steps,
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def convert_dataset_str_to_list(
    dataset_names,
    dataset_config_names,
    splits=None,
    text_column_names=None,
    dataset_samples=None,
    default_split="train",
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
    streaming: bool = True,
    seed: int = None,
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

    if len(dataset_names_dict) == 1:
        dataset_dict = dataset_names_dict[0]
        # we have a single dataset so just return it as is
        return load_dataset(
            dataset_dict["name"],
            dataset_dict["config"],
            split=dataset_dict["split"],
            streaming=streaming,
            **kwargs,
        )

    all_datasets = []
    # iterate over the datasets we want to interleave
    for dataset_dict in tqdm(dataset_names_dict, desc="Combining datasets..."):
        dataset = load_dataset(
            dataset_dict["name"],
            dataset_dict["config"],
            split=dataset_dict["split"],
            streaming=streaming,
            **kwargs,
        )
        # resample to specified sampling rate
        dataset = dataset.cast_column("audio", datasets.features.Audio(sampling_rate))
        dataset = dataset.remove_columns(
            set(dataset.features.keys()) - {"audio", dataset_dict["text_column_name"], "whisper_transcript"}
        )
        all_datasets.append(dataset)

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


def get_layers_to_supervise(student_layers: int, teacher_layers: int) -> dict:
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

    # 2. Define remote logging - do this early so that we get the full traceback on our remote logs
    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard:
        if jax.process_index() == 0:
            try:
                from flax.metrics.tensorboard import SummaryWriter

                summary_writer = SummaryWriter(log_dir=os.path.join(Path(training_args.output_dir), "runs"))
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

    # 3. Setup local logging
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
            " empty. Use `--overwrite_output_dir` to overcome."
        )

    # 4. Handle the repository creation
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

    if training_args.compilation_cache:
        cc.initialize_cache(os.path.join(model_args.cache_dir, "jax_cache"))

    # 5. Load dataset
    raw_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()

    # set seed for determinism
    set_seed(training_args.seed)

    if training_args.do_train:
        raw_datasets["train"] = load_multiple_datasets(
            data_args.train_dataset_name,
            data_args.train_dataset_config_name,
            splits=data_args.train_split_name,
            streaming=data_args.streaming,
            dataset_samples=data_args.train_dataset_samples,
            seed=training_args.seed,
            cache_dir=data_args.dataset_cache_dir,
            token=True if model_args.use_auth_token else None,
        )

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
                token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
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
                    token=True if model_args.use_auth_token else None,
                    streaming=data_args.streaming,
                )
                features = raw_datasets[pretty_name].features.keys()
                if "text" not in features:
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

    raw_datasets_train_features = list(raw_datasets["train"].features.keys())

    if data_args.audio_column_name not in raw_datasets_train_features:
        raise ValueError(
            f"--audio_column_name '{data_args.audio_column_name}' not found in dataset"
            f" '{data_args.dataset_name}'. Make sure to set `--audio_column_name` to"
            " the correct audio column - one of"
            f" {', '.join(raw_datasets_train_features)}."
        )

    if data_args.train_text_column_name not in raw_datasets_train_features:
        raise ValueError(
            f"--train_text_column_name {data_args.train_text_column_name} not found in dataset"
            f" '{data_args.dataset_name}'. Make sure to set `--train_text_column_name` to the"
            " correct text column - one of"
            f" {', '.join(raw_datasets_train_features)}."
        )

    # 6. Load pretrained model, tokenizer, and feature extractor
    config = WhisperConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=True if model_args.use_auth_token else None,
    )
    feature_extractor = FlaxWhisperFeatureExtractor.from_pretrained(
        (model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=True if model_args.use_auth_token else None,
    )
    tokenizer = WhisperTokenizerFast.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=True if model_args.use_auth_token else None,
    )

    # override timestamp tokens until tokenizer issues are fixed in transformers
    timestamps = [AddedToken("<|%.2f|>" % (i * 0.02), lstrip=False, rstrip=False) for i in range(1500 + 1)]
    tokenizer.add_tokens(timestamps)

    config.update(
        {
            "activation_dropout": model_args.activation_dropout,
            "attention_dropout": model_args.attention_dropout,
            "dropout": model_args.dropout,
        }
    )

    if training_args.precision == "full_mixed":
        # forward pass, backward pass and optimiser states in bf16
        dtype = jnp.bfloat16
        to_dtype = to_bf16
    elif training_args.precision == "half_mixed" or model_args.dtype == "bfloat16":
        # forward pass in bf16, backward pass and optimiser states in fp32
        dtype = jnp.bfloat16
        to_dtype = to_fp32
    else:
        if training_args.precision != "full":
            raise ValueError(
                f"`precision` should be one of: `full`, `half_mixed` or `full_mixed`, got {training_args.precision}"
            )
        # forward pass, backward pass and optimiser states in fp32
        dtype = jnp.float32
        to_dtype = to_fp32

    student_model, student_params = FlaxWhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        dtype=dtype,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        subfolder=model_args.subfolder,
        token=True if model_args.use_auth_token else None,
        _do_init=False,
        use_scan=model_args.load_with_scan_weights,
    )

    teacher_model, teacher_params = FlaxWhisperForConditionalGeneration.from_pretrained(
        model_args.teacher_model_name_or_path,
        # config=config,
        dtype=dtype,
        cache_dir=model_args.cache_dir,
        # revision=model_args.model_revision,
        token=True if model_args.use_auth_token else None,
        _do_init=False,
    )

    if student_model.config.decoder_start_token_id is None or teacher_model.config.decoder_start_token_id is None:
        raise ValueError(
            f"Make sure that `config.decoder_start_token_id` is correctly defined for both the "
            f"student and teacher model. Got {student_model.config.decoder_start_token_id} for the "
            f"student and {teacher_model.config.decoder_start_token_id} for the teacher."
        )

    # enable scan / gradient checkpointing if necessary
    if training_args.use_scan:
        student_model.enable_scan()  # to enable scan in the nn.Module
        student_params = student_model.convert_unroll_to_scan(student_params)  # to convert the unrolled params to scan

        teacher_model.enable_scan()  # faster compile time (even though we don't train the teacher)
        teacher_params = teacher_model.convert_unroll_to_scan(teacher_params)

    if training_args.gradient_checkpointing:
        student_model.enable_gradient_checkpointing()  # to enable checkpointing in the nn.Module, there is no change to the params structure
        teacher_model.enable_gradient_checkpointing()

    if hasattr(teacher_model.generation_config, "is_multilingual") and teacher_model.generation_config.is_multilingual:
        # We need to set the language and task ids for previously multilingual checkpoints - for now we hardcode this to English
        tokenizer.set_prefix_tokens(language="English", task="transcribe", predict_timestamps=False)
        student_model.generation_config.update(
            **{
                "language": "<|en|>",
                "task": "transcribe",
            }
        )

    # 7. Resample speech dataset: `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name,
        datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
    )

    # 8. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = int(data_args.max_duration_in_seconds * feature_extractor.sampling_rate)
    min_input_length = int(data_args.min_duration_in_seconds * feature_extractor.sampling_rate)
    max_label_length = (
        data_args.max_label_length if data_args.max_label_length is not None else student_model.config.max_length
    )
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    dataloader_num_workers = training_args.dataloader_num_workers
    dataloader_prefetch_size = data_args.prefetch_size
    train_text_column_name = data_args.train_text_column_name
    eval_text_column_name = "text"
    model_input_name = feature_extractor.model_input_names[0]
    normalizer = EnglishTextNormalizer(tokenizer.english_spelling_normalizer)
    wer_threshold = data_args.wer_threshold
    round_timestamps = data_args.round_timestamps

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

    def is_wer_in_range(ground_truth, whisper_transcript):
        norm_ground_truth = normalizer(ground_truth)
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
        input_columns=[eval_text_column_name, train_text_column_name],
    )

    if wer_threshold is not None:
        raw_datasets["train"] = (
            filter_by_wer_threshold(num_proc=num_workers, desc="filtering train dataset by wer")
            if not data_args.streaming
            else filter_by_wer_threshold()
        )

    def has_timestamp_tokens(input_str):
        """
        Identify whether the input string contains timestamp tokens, of the form <|0.00|>, by searching for
        pairs of left and right-angle brackets.
        """
        return bool(re.search("\<[^\>]*\>", input_str))

    def round_timestamp_tokens(input_str: str, ndigits: int = 1):
        timestamps = re.findall("\<[^\>]*\>", input_str, re.DOTALL)
        for token in timestamps:
            # extract time digits from timestamp token, e.g. <|6.24|> to 6.24
            time_digit = token[2:-2]
            # round to specified number of digits, e.g. 6.24 to 6.2
            time_digit = round(float(time_digit), ndigits=ndigits)
            # replace in original string with the same precision, e.g. <|6.24|> to <|6.20|>
            input_str = input_str.replace(token, "<|{:.2f}|>".format(time_digit))
        return input_str

    def prepare_train_dataset(batch):
        # process audio input
        sample = batch[audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = len(sample["array"])

        # process text targets
        input_str = batch[train_text_column_name]

        # prompt & timestamp processing: for now, we only do one or the other
        if input_str.startswith("<|startoftranscript|>") or input_str.startswith("<|startofprev|>"):
            # prompted target text already has special ids added, so don't add them here
            batch["labels"] = tokenizer(input_str, add_special_tokens=False).input_ids
            return batch

        has_timestamps = has_timestamp_tokens(input_str)

        if has_timestamps:
            predict_timestamps = bool(np.random.binomial(1, data_args.timestamp_probability))
            if not predict_timestamps:
                # filter timestamp token ids if not part of the prediction task
                input_str = tokenizer._filter_timestamp_ids(input_str)
            elif round_timestamps:
                input_str = round_timestamp_tokens(input_str)
        else:
            predict_timestamps = False

        tokenizer.set_prefix_tokens(language="English", task="transcribe", predict_timestamps=predict_timestamps)
        input_ids = tokenizer(input_str).input_ids
        batch["labels"] = input_ids
        return batch

    def prepare_eval_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = len(sample["array"])

        # process targets
        input_str = batch[eval_text_column_name]
        batch["labels"] = tokenizer(input_str).input_ids
        return batch

    vectorized_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()
    if training_args.do_train:
        map_fn_train = partial(
            raw_datasets["train"].map, function=prepare_train_dataset, remove_columns=raw_datasets_train_features
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
            vectorized_datasets[eval_split] = (
                map_fn_eval(num_proc=num_workers, desc="preprocess eval dataset")
                if not data_args.streaming
                else map_fn_eval()
            )

    # filter training data with inputs longer than max_input_length
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

    # filter training data with labels longer than max_label_length
    def is_labels_in_length_range(labels):
        return 0 < len(labels) < max_label_length

    filter_by_labels_fn = partial(
        vectorized_datasets.filter, function=is_labels_in_length_range, input_columns=["labels"]
    )
    vectorized_datasets = (
        filter_by_labels_fn(num_proc=num_workers, desc="filtering train dataset")
        if not data_args.streaming
        else filter_by_labels_fn()
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
    return_timestamps = data_args.return_timestamps if data_args.timestamp_probability > 0 else False

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

    # 9. Save feature extractor, tokenizer, config and generation config
    feature_extractor.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    config.save_pretrained(training_args.output_dir)
    student_model.generation_config.save_pretrained(
        training_args.output_dir
    )  # generation config stays bound to model to make it easy to jit

    processor = WhisperProcessor.from_pretrained(training_args.output_dir)

    data_collator = FlaxDataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=student_model.config.decoder_start_token_id,  # <|startoftranscript|>
        decoder_prev_token_id=tokenizer.all_special_ids[-3],  # <|startofprev|>
        input_padding="longest",
        target_padding="max_length",
        max_target_length=max_label_length,
    )

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Store some constants
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    eval_batch_size = per_device_eval_batch_size * jax.device_count()

    if not data_args.streaming and training_args.max_steps < 0:
        num_epochs = int(training_args.num_train_epochs)
        steps_per_epoch = len(vectorized_datasets["train"]) // train_batch_size
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

    # Create learning rate schedule
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        total_train_steps * gradient_accumulation_steps,
        training_args.lr_scheduler_type,
        training_args.warmup_steps * gradient_accumulation_steps,
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

    if gradient_accumulation_steps > 1:
        # accumulate gradients and apply once every k steps
        adamw = optax.MultiSteps(adamw, every_k_schedule=gradient_accumulation_steps)

    share_hidden_states = training_args.freeze_encoder and student_model.config.d_model == teacher_model.config.d_model
    encoder_layer_mapping = get_layers_to_supervise(
        student_model.config.encoder_layers, teacher_model.config.encoder_layers
    )
    decoder_layer_mapping = get_layers_to_supervise(
        student_model.config.decoder_layers, teacher_model.config.decoder_layers
    )

    # Setup train state
    student_state = TrainState.create(
        apply_fn=student_model.decode if share_hidden_states else student_model.__call__,
        params=student_params,
        tx=adamw,
        to_dtype=to_dtype,
        dropout_rng=dropout_rng,
        max_grad_norm=training_args.max_grad_norm,
    )

    if training_args.resume_from_checkpoint is not None:
        if os.path.isfile(os.path.join(training_args.resume_from_checkpoint, "train_state.msgpack")):
            logger.info(
                f"Checkpoint detected, resuming training at {training_args.resume_from_checkpoint}. To avoid "
                "this behavior, omit the resume_from_checkpoint argument."
            )
            with Path(os.path.join(training_args.resume_from_checkpoint, "train_state.msgpack")).open("rb") as f:
                student_state = from_bytes(student_state, f.read())
        else:
            logger.warning(
                f"Checkpoint {training_args.resume_from_checkpoint} not detected, training from scratch. Ensure "
                f"you pass the path to a folder with a valid checkpoint for your model."
            )

    def cross_entropy_loss(logits, labels):
        vocab_size = logits.shape[-1]
        # optax onehot always returns a float32 device array, need to downcast if performing mixed precision training
        onehot_targets = to_dtype(onehot(labels, vocab_size))
        loss = optax.softmax_cross_entropy(logits, onehot_targets)
        # ignore padded tokens from loss, i.e. where labels are not set to -100
        padding = labels >= 0
        loss = loss * padding
        loss = loss.sum()
        num_labels = padding.sum()
        return loss, num_labels

    # temperature smoothed kl-divergence
    def kl_divergence(target_distribution, log_predicted_distribution, labels, eps=1e-20):
        divergence = -target_distribution * (log_predicted_distribution - jnp.log(target_distribution + eps))
        # ignore padded tokens from divergence, i.e. where labels are not set to -100
        padding_mask = labels >= 0
        padding_mask = jnp.expand_dims(padding_mask, axis=-1)
        divergence = (divergence * padding_mask).sum()
        return to_dtype(divergence)  # respect the dtype of the backprop

    def mean_square_error_loss(student_outputs, teacher_outputs):
        mse = dtype(0.0)

        # tie encoder embeddings
        mse += jnp.mean(
            jnp.square(teacher_outputs.encoder_hidden_states[0] - student_outputs.encoder_hidden_states[0])
        )

        for student_layer_id, teacher_layer_id in encoder_layer_mapping.items():
            # offset the hidden-state layer ids by 1 to account for the extra embedding hidden-state
            student_hidden_state = student_outputs.encoder_hidden_states[student_layer_id + 1]
            teacher_hidden_state = teacher_outputs.encoder_hidden_states[teacher_layer_id + 1]
            mse += jnp.mean(jnp.square(teacher_hidden_state - student_hidden_state))

            # student_attention = student_outputs.encoder_attentions[student_layer_id]
            # teacher_attention = teacher_outputs.encoder_attentions[teacher_layer_id]
            # mse += jnp.mean(jnp.square(student_attention - teacher_attention))

        # tie decoder embeddings
        mse += jnp.mean(
            jnp.square(teacher_outputs.decoder_hidden_states[0] - student_outputs.decoder_hidden_states[0])
        )

        for student_layer_id, teacher_layer_id in decoder_layer_mapping.items():
            # offset the hidden-state layer ids by 1 to account for the extra embedding hidden-state
            student_hidden_state = student_outputs.decoder_hidden_states[student_layer_id + 1]
            teacher_hidden_state = teacher_outputs.decoder_hidden_states[teacher_layer_id + 1]
            mse += jnp.mean(jnp.square(teacher_hidden_state - student_hidden_state))

            # student_attention = student_outputs.decoder_attentions[student_layer_id]
            # teacher_attention = teacher_outputs.decoder_attentions[teacher_layer_id]
            # mse += jnp.mean(jnp.square(student_attention - teacher_attention))

            # student_cross_attention = student_outputs.cross_attentions[student_layer_id]
            # teacher_cross_attention = teacher_outputs.cross_attentions[teacher_layer_id]
            # mse += jnp.mean(jnp.square(student_cross_attention - teacher_cross_attention))

        return to_dtype(mse)  # respect the dtype of the backprop

    # Define gradient update step fn
    def train_step(
        student_state,
        teacher_params,
        batch,
        freeze_encoder,
        share_hidden_states,
        temperature=2.0,
    ):
        dropout_rng, new_dropout_rng = jax.random.split(student_state.dropout_rng)

        def compute_loss(student_params):
            labels = batch.pop("labels")
            output_hidden_states = not share_hidden_states and training_args.mse_weight > 0.0

            teacher_outputs = teacher_model(
                **batch,
                params=teacher_params,
                freeze_encoder=True,
                output_hidden_states=output_hidden_states,
                train=False,
            )

            if share_hidden_states:
                # if the student and teacher share the same frozen encoder then we don't have to recompute the
                # encoder hidden-states for the student model, we can just re-use from the teacher
                encoder_hidden_states = jax.lax.stop_gradient(teacher_outputs.encoder_last_hidden_state)
                encoder_outputs = FlaxBaseModelOutput(last_hidden_state=encoder_hidden_states)

                student_outputs = student_state.apply_fn(
                    decoder_input_ids=batch["decoder_input_ids"],
                    encoder_outputs=encoder_outputs,
                    params=student_params,
                    dropout_rng=dropout_rng,
                    train=True,
                )
            else:
                # do the full forward pass for the student model (encoder + decoder)
                student_outputs = student_state.apply_fn(
                    **batch,
                    params=student_params,
                    dropout_rng=dropout_rng,
                    freeze_encoder=freeze_encoder,
                    output_hidden_states=output_hidden_states,
                    train=True,
                )

            # CE (data) loss
            ce_loss, num_labels = cross_entropy_loss(student_outputs.logits, labels)

            # rescale by temperature to ensure gradients scale correctly
            teacher_distribution = jax.nn.softmax(teacher_outputs.logits / temperature, axis=-1)
            # ensure no information flow backwards through teacher
            teacher_distribution = jax.lax.stop_gradient(teacher_distribution)
            # log softmax of student predictions for numerical stability
            student_distribution = jax.nn.log_softmax(student_outputs.logits / temperature, axis=-1)
            # KL-divergence loss (scaled by temperature)
            kl_loss = kl_divergence(teacher_distribution, student_distribution, labels) * temperature**2

            # MSE loss between enc-dec hidden-states and attentions
            mse_loss = (
                mean_square_error_loss(student_outputs, teacher_outputs)
                if output_hidden_states
                else jnp.zeros_like(kl_loss)
            )

            # use DistilBart formulation - only tune the MSE weight and take remaining HPs from DistilBERT
            ce_weight = 0.8 if training_args.kl_weight > 0 else 1.0
            loss = ce_weight * ce_loss + training_args.kl_weight * kl_loss + training_args.mse_weight * mse_loss

            return loss, (
                ce_loss,
                kl_loss,
                mse_loss,
                num_labels,
            )

        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
        (loss, (ce_loss, kl_loss, mse_loss, num_labels)), grad = grad_fn(to_dtype(student_state.params))

        # true loss = total loss / total samples
        loss = jax.lax.psum(loss, "batch")
        num_labels = jax.lax.psum(num_labels, "batch")
        loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)

        # true grad = total grad / total samples
        grad = jax.lax.psum(grad, "batch")
        grad = jax.tree_util.tree_map(lambda x: x / num_labels, grad)
        new_state = student_state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng, to_dtype=to_dtype)

        # CE/KL/MSE losses for logging
        ce_loss = jax.lax.psum(ce_loss, "batch")
        ce_loss = jax.tree_util.tree_map(lambda x: x / num_labels, ce_loss)

        kl_loss = jax.lax.psum(kl_loss, "batch")
        kl_loss = jax.tree_util.tree_map(lambda x: x / num_labels, kl_loss)

        mse_loss = jax.lax.psum(mse_loss, "batch")
        mse_loss = jax.tree_util.tree_map(lambda x: x / num_labels, mse_loss)

        metrics = {
            "loss": loss,
            "learning_rate": linear_decay_lr_schedule_fn(student_state.step),
            "ce_loss": ce_loss,
            "kl_loss": kl_loss,
            "mse_loss": mse_loss,
        }
        return new_state, metrics

    # Define eval fn
    def eval_step(student_params, teacher_params, batch):
        labels = batch.pop("labels")
        output_hidden_states = not share_hidden_states and training_args.mse_weight > 0

        student_outputs = student_model(
            **batch,
            params=student_params,
            output_hidden_states=output_hidden_states,
            train=False,
        )
        student_distribution = jax.nn.log_softmax(student_outputs.logits, axis=-1)
        ce_loss, num_labels = cross_entropy_loss(student_outputs.logits, labels)

        teacher_outputs = teacher_model(
            **batch,
            params=teacher_params,
            output_hidden_states=output_hidden_states,
            train=False,
        )
        teacher_distribution = jax.nn.softmax(teacher_outputs.logits, axis=-1)
        # temperature is always 1 for eval
        kl_loss = kl_divergence(teacher_distribution, student_distribution, labels)

        mse_loss = (
            mean_square_error_loss(student_outputs, teacher_outputs)
            if output_hidden_states
            else jnp.zeros_like(kl_loss)
        )

        ce_weight = 0.8 if training_args.kl_weight > 0 else 1.0
        loss = ce_weight * ce_loss + training_args.kl_weight * kl_loss + training_args.mse_weight * mse_loss
        # true loss = total loss / total samples
        loss = jax.lax.psum(loss, "batch")
        num_labels = jax.lax.psum(num_labels, "batch")
        loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)

        # CE/KL/MSE losses for logging
        ce_loss = jax.lax.psum(ce_loss, "batch")
        ce_loss = jax.tree_util.tree_map(lambda x: x / num_labels, ce_loss)

        kl_loss = jax.lax.psum(kl_loss, "batch")
        kl_loss = jax.tree_util.tree_map(lambda x: x / num_labels, kl_loss)

        mse_loss = jax.lax.psum(mse_loss, "batch")
        mse_loss = jax.tree_util.tree_map(lambda x: x / num_labels, mse_loss)

        metrics = {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss, "mse_loss": mse_loss}
        return metrics

    # Define generation function
    num_beams = (
        training_args.generation_num_beams
        if training_args.generation_num_beams is not None
        else student_model.config.num_beams
    )

    # forcing the language and task tokens helps the model in its generations
    gen_kwargs = {
        "max_length": max_label_length,
        "num_beams": num_beams,
        "language": "<|en|>",
        "task": "transcribe",
        "return_timestamps": return_timestamps,
    }

    def generate_step(student_params, batch):
        output_ids = student_model.generate(
            batch[model_input_name],
            attention_mask=batch.get("attention_mask"),
            params=student_params,
            **gen_kwargs,
        )
        return output_ids.sequences

    # Replicate the train state on each device
    student_state = student_state.replicate()

    # Replicate the teacher params on each device
    teacher_params = jax_utils.replicate(teacher_params)

    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(
        train_step,
        "batch",
        in_axes=(0, 0, 0, None, None, None),
        donate_argnums=(0,),
        static_broadcasted_argnums=(
            3,
            4,
        ),
    )
    p_eval_step = jax.pmap(eval_step, "batch")
    p_generate_step = jax.pmap(generate_step, "batch")

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
    train_metrics = []
    batches_to_skip = jax.device_get(unreplicate(student_state.step))
    cur_step = int(batches_to_skip)  # will be zero if starting from scratch
    epochs_trained = batches_to_skip // steps_per_epoch
    steps_trained_progress_bar = tqdm(range(total_train_steps), desc="Train steps ... ", position=0)
    steps_trained_progress_bar.update(batches_to_skip)
    continue_training = True
    minibatch_steps = 0

    if batches_to_skip > 0:
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {batches_to_skip}")

    # Generate a training data loader by shuffling sampling indices from the train dataset
    train_loader = get_data_loader(
        training_args.seed,
        vectorized_datasets["train"],
        batch_size=train_batch_size,
        data_collator=data_collator,
        dataloader_num_workers=dataloader_num_workers,
        skip_batches=batches_to_skip,
        prefetch_size=dataloader_prefetch_size,
    )

    for epoch in range(epochs_trained, num_epochs):
        if hasattr(train_loader, "dataset") and isinstance(train_loader.dataset, IterableDataset):
            train_loader.dataset.set_epoch(epoch)

        for batch in train_loader:
            minibatch_steps += 1
            update_step = minibatch_steps == gradient_accumulation_steps

            if update_step:
                steps_trained_progress_bar.update(1)
                cur_step += 1
                minibatch_steps = 0

            batch = shard(batch.data)
            student_state, train_metric = p_train_step(
                student_state,
                teacher_params,
                batch,
                training_args.freeze_encoder,
                share_hidden_states,
                training_args.temperature,
            )

            if cur_step % training_args.logging_steps == 0 and update_step:
                train_metrics.append(train_metric)
                train_metric_to_write = unreplicate(train_metric)
                steps_trained_progress_bar.write(
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
                        epoch,
                        prefix="train",
                    )

            # save checkpoint and weights after each save_steps and at the end of training
            if (cur_step % training_args.save_steps == 0 and update_step) or cur_step == total_train_steps:
                if jax.process_index() == 0:
                    save_hf_weights(
                        student_state,
                        student_model,
                        processor,
                        training_args.output_dir,
                        cur_step,
                        total_train_steps,
                        use_scan=training_args.use_scan,
                    )
                    if training_args.save_train_state:
                        student_state.save_state(
                            training_args.output_dir, save_total_limit=training_args.save_total_limit
                        )
                    if training_args.push_to_hub:
                        repo.push_to_hub(
                            commit_message=f"Saving train state of step {cur_step}",
                            blocking=False,
                        )

            if training_args.do_eval and (
                (cur_step % eval_steps == 0 and update_step) or cur_step == total_train_steps
            ):
                train_time += time.time() - train_start
                # ======================== Evaluating ==============================
                for eval_split in all_eval_splits:
                    eval_metrics = []
                    eval_preds = []
                    eval_labels = []
                    eval_start = time.time()

                    eval_loader = get_data_loader(
                        training_args.seed,
                        vectorized_datasets[eval_split],
                        batch_size=eval_batch_size,
                        data_collator=data_collator,
                        shuffle=False,
                        drop_last=False,
                        dataloader_num_workers=dataloader_num_workers,
                    )
                    for batch in tqdm(eval_loader, desc=f"Evaluating {eval_split}...", position=2):
                        # Model forward
                        labels = batch["labels"]

                        metrics = pad_shard_unpad(
                            p_eval_step,
                            static_argnums=(
                                0,
                                1,
                            ),
                            static_return=True,
                        )(
                            student_state.params,
                            teacher_params,
                            batch.data,
                            min_device_batch=per_device_eval_batch_size,
                        )
                        eval_metrics.append(metrics)

                        # generation
                        if training_args.predict_with_generate:
                            generated_ids = pad_shard_unpad(p_generate_step)(
                                student_state.params, batch.data, min_device_batch=per_device_eval_batch_size
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
                        wer_metric, pred_str, label_str, norm_pred_str, norm_label_str = compute_metrics(
                            eval_preds, eval_labels
                        )
                        eval_metrics.update(wer_metric)
                        wer_desc = " ".join([f"Eval {key}: {value} |" for key, value in wer_metric.items()])

                    # Print metrics and update progress bar
                    steps_trained_progress_bar.write(
                        f"Eval results for step ({cur_step} / {total_train_steps} | Eval Loss: {eval_metrics['loss']} |"
                        f" {wer_desc})"
                    )

                    if has_tensorboard and jax.process_index() == 0:
                        write_eval_metric(
                            summary_writer,
                            eval_metrics,
                            cur_step,
                            prefix=eval_split,
                        )

                    if has_wandb and jax.process_index() == 0:
                        write_wandb_metric(wandb_logger, eval_metrics, eval_time, cur_step, epoch, prefix=eval_split)
                        if training_args.predict_with_generate:
                            write_wandb_pred(
                                wandb_logger,
                                pred_str,
                                label_str,
                                norm_pred_str,
                                norm_label_str,
                                cur_step,
                                prefix=eval_split,
                            )

                if has_tensorboard and jax.process_index() == 0:
                    # we'll only log to tensorboard every eval steps
                    write_train_metric(
                        summary_writer,
                        train_metrics,
                        train_time,
                        cur_step,
                        training_args.logging_steps,
                    )

                # flush the train metrics
                train_start = time.time()
                train_metrics = []

            # break condition
            if cur_step == total_train_steps:
                continue_training = False
                break

        if not continue_training:
            break


if __name__ == "__main__":
    main()
