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
Evaluating a Whisper model on one or more long-form evaluation datasets.
"""
# You can also adapt this script for your own speech recognition validation. Pointers for this are left as comments.

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import torch
import transformers
from datasets import DatasetDict, IterableDatasetDict, load_dataset
from jiwer import process_words, wer_default
from nltk import ngrams
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    WhisperTokenizer,
    is_tensorboard_available,
    is_wandb_available,
    pipeline,
)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer, BasicTextNormalizer
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.34.0.dev0")

require_version("datasets>=2.14.6", "To fix: `pip install --upgrade datasets`")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
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
                "Floating-point format in which the model weights should be initialized"
                " and evaluated. Choose one of `[float32, float16, bfloat16]`."
            )
        },
    )
    return_timestamps: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to predict timestamps (alongside the text predictions). Timestamp predictions "
            "are discarded at the end of inference, but may assist in the model in reducing hallucinations."
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
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                "Exponential penalty to the length that is used with beam-based generation. It is applied as an "
                "exponent to the sequence length, which in turn is used to divide the score of the sequence. Since "
                "the score is the log likelihood of the sequence (i.e. negative), length_penalty > 1.0 promotes "
                "longer sequences, while length_penalty < 1.0 encourages shorter sequences."
            )
        },
    )
    do_sample: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use sampling ; use greedy decoding otherwise."},
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={"help": "The number of the highest probability vocabulary tokens to keep for top-k-filtering."},
    )
    temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "The value used to modulate the next token probabilities if sampling."},
    )
    chunk_length_s: Optional[float] = field(
        default=0,
        metadata={
            "help": "The input length for each chunk. By default, the chunk length is set to 0, which means no chunking."
        },
    )
    attn_type: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Which attention type to use in the encoder and decoder attention layers. Can be one of:"
                "1. `None`: default Transformers attention implementation."
                "2. `flash_attn`: Flash Attention through PyTorch SDPA. Requires `torch>=2.0` and `optimum` to be installed. Recommended for hardware where Flash Attention 2 is not supported, e.g. Turing GPUs, (T4, RTX 2080)"
                "3. `flash_attn_2`: Flash Attention 2 through the Flash Attention package https://github.com/Dao-AILab/flash-attention. **Always** recommended on supported hardware (Ampere, Ada, or Hopper GPUs, e.g., A100, RTX 3090, RTX 4090, H100)"
            )
        },
    )


@dataclass
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
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'."},
    )
    max_label_length: int = field(
        default=256,
        metadata={"help": "Truncate transcriptions that are longer `max_label_length` tokens."},
    )
    wandb_project: str = field(
        default="distil-whisper-long-form",
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
    log_predictions: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to log the ground truths / pred text to the wandb logger."},
    )
    ngram_degree: Optional[int] = field(
        default=5, metadata={"help": "Degree of n-grams used when computing duplicate n-grams in the predicted text."}
    )


def write_metric(summary_writer, eval_metrics, prefix="eval"):
    for metric_name, value in eval_metrics.items():
        summary_writer.add_scalar(f"{prefix}/{metric_name}", value, 0)


def write_wandb_metric(wandb_logger, metrics, train_time, prefix):
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    wandb_logger.log(log_metrics)


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
        {f"{prefix}/predictions": wandb_logger.Table(columns=columns, data=str_data)},
    )


def convert_dataset_str_to_list(
    dataset_names, dataset_config_names, splits=None, text_column_names=None, dataset_hours=None, default_split="train"
):
    if isinstance(dataset_names, str):
        dataset_names = dataset_names.split("+")
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


def data(dataset, text_column_name="text", log_audio=False):
    for item in dataset:
        yield {**item["audio"], "reference": item[text_column_name], "audio": item["audio"] if log_audio else None}


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

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if "tensorboard" in training_args.report_to:
        if has_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                summary_writer = SummaryWriter(log_dir=os.path.join(training_args.output_dir, "runs"))
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
        if has_wandb:
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

    # 2. Setup logging
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Set the verbosity to info of the Transformers logger.
    # We only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    logger.info("Evaluation parameters %s", training_args)

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

    # load multiple eval sets
    for dataset_dict in dataset_names_dict:
        # Clean-up the dataset name for pretty logging
        # ("distil-whisper/librispeech_asr", "validation.clean") -> "librispeech_asr/validation-clean"
        pretty_name = f"{dataset_dict['name'].split('/')[-1]}/{dataset_dict['split'].replace('.', '-')}"
        raw_datasets[pretty_name] = load_dataset(
            dataset_dict["name"],
            dataset_dict["config"],
            split=dataset_dict["split"],
            cache_dir=data_args.dataset_cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
        if dataset_dict["text_column_name"] not in list(raw_datasets[pretty_name].features.keys()):
            raise ValueError(
                f"--text column name {dataset_dict['text_column_name']} not found in the evaluation "
                f"dataset {dataset_dict['name']}. Ensure `text_column_name` is set to the correct column "
                f"for the target text. Should be one of {' '.join(list(raw_datasets[pretty_name].features.keys()))}"
            )
        if dataset_dict["text_column_name"] != "text":
            raw_datasets[pretty_name] = raw_datasets[pretty_name].rename_column(
                dataset_dict["text_column_name"], "text"
            )

    # Streaming mode robust way of obtaining the features
    raw_datasets_features = list(next(iter(raw_datasets.values())).features.keys())
    audio_column_name = data_args.audio_column_name

    if audio_column_name not in raw_datasets_features:
        raise ValueError(
            f"--audio_column_name '{audio_column_name}' not found in dataset"
            f" '{data_args.dataset_name}'. Make sure to set `--audio_column_name` to"
            " the correct audio column - one of"
            f" {', '.join(raw_datasets_features)}."
        )

    for split in raw_datasets:
        raw_datasets[split] = raw_datasets[split].remove_columns(
            set(raw_datasets[split].features.keys()) - {audio_column_name, "text"}
        )

    if data_args.max_eval_samples is not None:
        for split in raw_datasets:
            raw_datasets[split] = (
                raw_datasets[split].take(data_args.max_eval_samples)
                if data_args.streaming
                else raw_datasets[split].select(range(data_args.max_eval_samples))
            )

    # Store some constants
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    num_beams = training_args.generation_num_beams if training_args.generation_num_beams is not None else 1

    model_kwargs = {
        "cache_dir": model_args.cache_dir,
        "subfolder": model_args.subfolder,
    }

    # 5. Load pretrained model, tokenizer, and feature extractor
    pipe = pipeline(
        "automatic-speech-recognition",
        model_args.model_name_or_path,
        torch_dtype=getattr(torch, model_args.dtype),
        model_kwargs=model_kwargs,
        max_new_tokens=training_args.generation_max_length,
        batch_size=per_device_eval_batch_size,
        chunk_length_s=model_args.chunk_length_s,
        return_timestamps=model_args.return_timestamps,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    if pipe.model.can_generate():
        if pipe.model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
        generate_kwargs = {
            "num_beams": num_beams,
            "length_penalty": model_args.length_penalty,
            "do_sample": model_args.do_sample,
            "top_k": model_args.top_k,
            "temperature": model_args.temperature,
        }
        if hasattr(pipe.model.generation_config, "is_multilingual") and pipe.model.generation_config.is_multilingual:
            generate_kwargs = generate_kwargs.update({"langauge": model_args.language, "task": model_args.task})
        elif model_args.language is not None:
            raise ValueError(
                "Setting language token for an English-only checkpoint is not permitted. The language argument should "
                "only be set for multilingual checkpoints."
            )
    else:
        generate_kwargs = None

    # 8. Load Metric
    whisper_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")
    normalizer = (
        BasicTextNormalizer() if model_args.language is not None else EnglishTextNormalizer(whisper_tokenizer.english_spelling_normalizer)
    )

    def compute_metrics(pred_str, label_str, ngram_degree=5):
        # normalize everything and re-compute the WER
        norm_pred_str = [normalizer(pred) for pred in pred_str]
        norm_label_str = [normalizer(label) for label in label_str]
        # for logging, we need the pred/labels to match the norm_pred/norm_labels, so discard any filtered samples here
        pred_str = [pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
        label_str = [label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]
        # filtering step to only evaluate the samples that correspond to non-zero normalized references:
        norm_pred_str = [norm_pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
        norm_label_str = [norm_label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]

        wer_output = process_words(norm_label_str, norm_pred_str, wer_default, wer_default)
        wer_norm = 100 * wer_output.wer
        ier_norm = 100 * wer_output.insertions / sum([len(ref) for ref in wer_output.references])
        ser_norm = 100 * wer_output.substitutions / sum([len(ref) for ref in wer_output.references])
        der_norm = 100 * wer_output.deletions / sum([len(ref) for ref in wer_output.references])

        all_ngrams = list(ngrams(" ".join(norm_pred_str).split(), ngram_degree))
        repeated_ngrams = len(all_ngrams) - len(set(all_ngrams))

        return (
            {"wer": wer_norm, "ier": ier_norm, "ser": ser_norm, "der": der_norm, "repeated_ngrams": repeated_ngrams},
            pred_str,
            label_str,
            norm_pred_str,
            norm_label_str,
        )

    def eval_step(split="eval"):
        # ======================== Evaluating ==============================
        eval_preds = []
        eval_labels = []
        eval_audios = []
        eval_start = time.time()

        for sample in tqdm(
            pipe(
                data(raw_datasets[split], log_audio=data_args.log_audio),
                generate_kwargs=generate_kwargs,
            ),
            desc=f"Evaluating {split}...",
        ):
            eval_preds.append(sample["text"])
            eval_labels.append(sample["reference"][0])
            if data_args.log_audio:
                eval_audios.append(sample["audio"][0])

        eval_time = time.time() - eval_start

        wer_metric, pred_str, label_str, norm_pred_str, norm_label_str = compute_metrics(
            eval_preds, eval_labels, data_args.ngram_degree
        )
        wer_desc = " ".join([f"{split} {key}: {value} |" for key, value in wer_metric.items()])

        # Print metrics to stdout
        logger.info(wer_desc)

        # Save metrics to tensorboard
        if has_tensorboard and "tensorboard" in training_args.report_to:
            write_metric(summary_writer, wer_metric, prefix=split)

        # Save metrics to wandb
        if has_wandb and "wandb" in training_args.report_to:
            write_wandb_metric(wandb_logger, wer_metric, eval_time, prefix=split)
            if data_args.log_predictions:
                write_wandb_pred(
                    wandb_logger, eval_audios, pred_str, label_str, norm_pred_str, norm_label_str, prefix=split
                )

    logger.info("***** Running Eval *****")
    logger.info("  Instantaneous batch size per device =" f" {training_args.per_device_eval_batch_size}")
    logger.info(f"  Total eval batch size (w. parallel & distributed) = {training_args.per_device_eval_batch_size}")
    logger.info(f"  Return timestamps = {model_args.return_timestamps}")
    if pipe.model.can_generate():
        logger.info(f"  Beam size = {num_beams}")
        if num_beams > 1:
            logger.info(f"  Length penalty size = {model_args.length_penalty}")
        logger.info(f"  Do sample = {model_args.do_sample}")
        if model_args.do_sample:
            logger.info(f"  Top k = {model_args.top_k}")
            logger.info(f"  Temperature = {model_args.temperature}")

    for split in raw_datasets:
        eval_step(split=split)


if __name__ == "__main__":
    main()
