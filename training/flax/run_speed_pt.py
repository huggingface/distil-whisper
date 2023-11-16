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

import json
import logging
import os
import string
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import datasets
import evaluate
import numpy as np
import torch
import transformers
import whisper
from datasets import DatasetDict, IterableDatasetDict, load_dataset
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    is_wandb_available,
    pipeline,
)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from transformers.models.whisper.modeling_whisper import WhisperForCausalLM
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

require_version(
    "datasets>=1.18.0",
    "To fix: pip install -r examples/flax/speech-recogintion/requirements.txt",
)

logger = logging.getLogger(__name__)

PIPELINE_BATCH_SIZE = 16


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
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "The name of the model to use (via the transformers library). "},
    )
    assistant_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The name of the assistant model to use to do speculative decoding. If None, no speculative decoding will be done."
        },
    )
    use_fp16: bool = field(
        default=True,
        metadata={"help": "Whether to evaluate in fp16"},
    )
    use_torch_compile: bool = field(
        default=False,
        metadata={"help": "Whether to compile the model"},
    )
    use_orig_whisper: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate with orig whisper"},
    )
    use_bf16: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate in bf16"},
    )
    use_pipeline: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate with Transformers pipeline"},
    )
    chunk_length_s: float = field(
        default=30.0, metadata={"help": "Chunk length to use when `use_pipeline` is enabled."}
    )
    return_timestamps: bool = field(
        default=False,
        metadata={
            "help": "Whether to decode with timestamps. This can help for improved WER for long form evaluation."
        },
    )
    attn_type: Optional[str] = field(
        default=None,
        metadata={"help": "Which attn type to use: None, 'flash', 'compile', 'flash+compile'"},
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "The batch size used for evluation."},
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "The beam size used for evluation."},
    )
    samples_per_dataset: Optional[int] = field(
        default=None,
        metadata={"help": "Number of samples per dataset used to measure speed."},
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
    max_gen_length: int = field(default=128, metadata={"help": "Generate up until max_gen_length tokens."})
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
        default="distil-whisper-speed-benchmark",
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


def write_metric(summary_writer, eval_metrics, step, prefix="eval"):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"{prefix}/{metric_name}", value, step)


def write_wandb_metric(wandb_logger, metrics, train_time, prefix):
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    wandb_logger.log(log_metrics)  # TODO(SG): bug with wandb means we can't log the step count


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


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser([DataTrainingArguments])

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        data_args = parser.parse_args_into_dataclasses()[0]

    # 2. Setup logging
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if data_args.use_pipeline and data_args.batch_size > 1:
        raise ValueError("Make sure that `batch_size` is set to 1 when `use_pipeline=True`.")

    has_wandb = is_wandb_available()
    if has_wandb:
        import wandb
        import wandb as wandb_logger

        # Set up wandb run
        wandb_logger.init(
            project=data_args.wandb_project,
            name=data_args.wandb_name,
            job_type=data_args.wandb_job_type,
            dir=data_args.wandb_dir,
            save_code=data_args.save_code_to_wandb,
        )
        wandb_logger.log({"torch_version": str(torch.__version__)})
        wandb_logger.log({"transformers_version": str(transformers.__version__)})
        wandb_logger.log({"batch_size": data_args.batch_size})

        if data_args.use_pipeline:
            wandb_logger.log({"chunk_length_s": data_args.chunk_length_s})
    else:
        raise ValueError("Wandb logging requires wandb to be installed. Run `pip install wandb` to enable.")

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
            use_auth_token=True,
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
                use_auth_token=True,
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
    processor = WhisperProcessor.from_pretrained(data_args.model_name_or_path)

    dtype = torch.float16 if data_args.use_fp16 else torch.float32
    if data_args.use_bf16:
        dtype = torch.bfloat16

    use_flash_attention_2 = data_args.attn_type is not None and "flash2" in data_args.attn_type

    # make sure we're not using a T4
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    gpu_type = [x for x in result.stdout.split("=") if len(x) > 1][1].split("0")[1].split()

    use_sdpa = False
    if gpu_type[0] == "Tesla" and use_flash_attention_2:
        use_flash_attention_2 = False
        use_sdpa = True

    use_orig_whisper = False
    if data_args.use_orig_whisper:
        use_orig_whisper = True

        model_name = data_args.model_name_or_path.split("/")[-1].split("whisper-")[-1]
        model = whisper.load_model(model_name)
        model.cuda()
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            data_args.model_name_or_path, torch_dtype=dtype, use_flash_attention_2=use_flash_attention_2
        )
        model.cuda()

    if use_sdpa:
        logger.info("Use SDPA via BetterTransformers...")
        model.to_bettertransformer()

    if data_args.use_torch_compile:
        logger.info("Enabling torch compile for the encoder.")
        # let's compile the encoder forward path
        model.model.encoder.forward = torch.compile(
            model.model.encoder.forward, mode="reduce-overhead", fullgraph=True
        )

        # init torch compile once to create binaries
        input_values = np.random.randn(data_args.batch_size, 16_000)
        input_features = processor(input_values, return_tensors="pt", sampling_rate=16_000).input_features
        input_features = input_features.to(dtype=dtype, device=model.device)

        # run generation three times to that model is compiled
        for _ in range(3):
            _ = model.generate(input_features)

    model_pipeline = None
    if data_args.use_pipeline:
        model_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=model.device,
            chunk_length_s=data_args.chunk_length_s,
        )
        model_pipeline_forward = model_pipeline._forward

    assistant_model = None
    if data_args.assistant_model_name_or_path is not None:
        logger.info("Loading assistant model...")

        if data_args.assistant_model_name_or_path.startswith("openai"):
            assistant_model = WhisperForConditionalGeneration.from_pretrained(
                data_args.assistant_model_name_or_path, torch_dtype=dtype, use_flash_attention_2=use_flash_attention_2
            )
        else:
            assistant_model = WhisperForCausalLM.from_pretrained(
                data_args.assistant_model_name_or_path, torch_dtype=dtype, use_flash_attention_2=use_flash_attention_2
            )

        assistant_model.cuda()

    # 6. Resample speech dataset: `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name,
        datasets.features.Audio(sampling_rate=processor.feature_extractor.sampling_rate),
    )

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_label_length = (
        data_args.max_label_length if data_args.max_label_length is not None else model.config.max_length
    )
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    model_input_name = processor.feature_extractor.model_input_names[0]
    normalizer = EnglishTextNormalizer(processor.tokenizer.english_spelling_normalizer)

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

        if model_pipeline is None and not use_orig_whisper:
            inputs = processor.feature_extractor(
                sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
            )
            batch[model_input_name] = inputs.get(model_input_name)
        else:
            batch[model_input_name] = sample["array"]

        # process audio length
        batch["length_in_s"] = len(sample["array"]) / sample["sampling_rate"]

        # process targets
        input_str = batch["text"]
        batch["labels"] = processor.tokenizer(input_str, max_length=max_label_length, truncation=True).input_ids
        return batch

    vectorized_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()

    for split in raw_datasets:
        raw_datasets_features = list(raw_datasets[split].features.keys())

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
    list(string.punctuation.replace("'", ""))

    def compute_metrics(pred_str, label_str):
        # normalize everything and re-compute the WER
        norm_pred_str = [normalizer(pred) for pred in pred_str]
        norm_label_str = [normalizer(label) for label in label_str]
        # for logging, we need the pred/labels to match the norm_pred/norm_labels, so discard any filtered samples here
        pred_str = [pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
        label_str = [label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]
        # filtering step to only evaluate the samples that correspond to non-zero normalized references:
        norm_pred_str = [norm_pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
        norm_label_str = [norm_label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]

        # if any of the two lengths is 0, return 0 WER
        if len(norm_pred_str) == 0 or len(norm_label_str) == 0:
            return 0.0

        wer = 100 * metric.compute(predictions=norm_pred_str, references=norm_label_str)

        return wer

    result_datasets = DatasetDict()

    def benchmark(batch):
        if model_pipeline is None and not use_orig_whisper:
            inputs = torch.cat(batch[model_input_name], dim=0).cuda()
            if data_args.use_fp16:
                inputs = inputs.to(torch.float16)
            if data_args.use_bf16:
                inputs = inputs.to(torch.bfloat16)

            inner_batch_size = inputs.shape[0]
        else:
            inner_batch_size = 1

            inputs = batch[model_input_name]

        gen_kwargs = {
            "return_timestamps": data_args.return_timestamps,
            "max_length": data_args.max_gen_length,
        }

        # if not data_args.model_name_or_path.endswith(".en"):
        if not data_args.model_name_or_path.endswith(".en") and not data_args.model_name_or_path.endswith("24-2"):
            gen_kwargs["language"] = "<|en|>"
            gen_kwargs["task"] = "transcribe"
            gen_kwargs["num_beams"] = data_args.num_beams

        # Time forward
        if use_orig_whisper:
            raw_audio = inputs[0].astype(np.float32)
            out_dict = model.transcribe(raw_audio)

            batch["transcription"] = [out_dict["text"]]
            batch["time"] = [out_dict["all_time"]]
        elif model_pipeline is not None:
            # if model is pipeline let's make sure that only forward is timed and not pre- and post-process
            time_result = []

            def _forward_time(*args, **kwargs):
                start_time = time.time()
                result = model_pipeline_forward(*args, **kwargs)
                end_time = time.time() - start_time

                time_result.append(end_time)

                return result

            model_pipeline._forward = _forward_time

            result = model_pipeline(inputs, batch_size=PIPELINE_BATCH_SIZE, generate_kwargs=gen_kwargs)[0]["text"]
            batch["transcription"] = [result]
            batch["time"] = [sum(time_result)]
        elif assistant_model is not None:
            gen_kwargs["assistant_model"] = assistant_model

            start_time = time.time()
            with torch.no_grad():
                encoder_outputs = model.get_encoder()(inputs)

            gen_kwargs["encoder_outputs"] = encoder_outputs

            if data_args.assistant_model_name_or_path.startswith("openai"):
                with torch.no_grad():
                    assistant_encoder_outputs = assistant_model.get_encoder()(inputs)

                gen_kwargs["assistant_encoder_outputs"] = assistant_encoder_outputs
            else:
                gen_kwargs["assistant_encoder_outputs"] = encoder_outputs

            output_ids = model.generate(**gen_kwargs)
            batch["time"] = inner_batch_size * [(time.time() - start_time) / inner_batch_size]

            batch["transcription"] = processor.batch_decode(output_ids, skip_special_tokens=True)
        else:
            start_time = time.time()
            output_ids = model.generate(inputs, **gen_kwargs)
            batch["time"] = inner_batch_size * [(time.time() - start_time) / inner_batch_size]

            batch["transcription"] = processor.batch_decode(output_ids, skip_special_tokens=True)

        batch["length_in_s"] = batch["length_in_s"]
        batch["reference"] = processor.batch_decode(batch["labels"], skip_special_tokens=True)
        batch["num_words"] = [len(r.split()) for r in batch["reference"]]

        return batch

    for split in vectorized_datasets:
        vectorized_datasets_features = [model_input_name]

        map_fn = partial(
            vectorized_datasets[split].map,
            function=benchmark,
            remove_columns=vectorized_datasets_features,
            batch_size=data_args.batch_size,
            batched=True,
        )

        result_datasets[split] = (
            map_fn(num_proc=1, desc="benchmark eval dataset") if not data_args.streaming else map_fn()
        )

    stats_dataset = DatasetDict()

    all_stats = {
        "times_audio_total": 0,
        "times_transcription_total": 0,
        "num_words_total": 0,
        "num_samples": 0,
        "time_per_sample": 0,
        "rtf": 0,
        "words_per_s": 0,
        "wer": 0,
    }

    count = 0
    for split in result_datasets:
        transcriptions = []
        references = []
        stats = {k: 0 for k in all_stats.keys()}

        print(f"Start benchmarking {split}...")
        if data_args.streaming:
            result_iter = iter(result_datasets[split])

        for result in result_iter:
            stats["times_audio_total"] += result["length_in_s"]
            stats["times_transcription_total"] += result["time"]
            stats["num_words_total"] += result["num_words"]
            stats["num_samples"] += 1
            transcriptions.append(result["transcription"])
            references.append(result["reference"])

            count += 1
            print(f"Processed {count} samples...")

            if data_args.samples_per_dataset is not None and stats["num_samples"] == data_args.samples_per_dataset:
                break

        stats["time_per_sample"] = stats["times_transcription_total"] / stats["num_samples"]
        stats["avg_length_sample"] = stats["times_audio_total"] / stats["num_samples"]
        stats["wer"] = compute_metrics(transcriptions, references)
        stats["rtf"] = stats["times_audio_total"] / stats["times_transcription_total"]
        stats["words_per_s"] = stats["num_words_total"] / stats["times_transcription_total"]

        stats_dataset[split] = stats

        log_stats = {f"{split}_{k}": v for k, v in stats.items()}
        wandb_logger.log(log_stats)

        all_stats["times_audio_total"] += stats["times_audio_total"]
        all_stats["times_transcription_total"] += stats["times_transcription_total"]
        all_stats["wer"] += stats["wer"]
        all_stats["num_samples"] += stats["num_samples"]
        all_stats["num_words_total"] += stats["num_words_total"]

    all_stats["time_per_sample"] = all_stats["times_transcription_total"] / all_stats["num_samples"]
    all_stats["avg_length_sample"] = all_stats["times_audio_total"] / all_stats["num_samples"]
    all_stats["wer"] = all_stats["wer"] / len(result_datasets)
    all_stats["rtf"] = all_stats["times_audio_total"] / all_stats["times_transcription_total"]
    all_stats["words_per_s"] = all_stats["num_words_total"] / all_stats["times_transcription_total"]

    stats_dataset["all"] = all_stats

    log_all_stats = {f"all_{k}": v for k, v in all_stats.items()}
    wandb_logger.log(log_all_stats)

    benchmark_artifact = wandb.Artifact("Benchmark", type="datasets")
    with tempfile.TemporaryDirectory() as temp_dir:
        for split in stats_dataset:
            file_name = os.path.join(temp_dir, f"{'_'.join(split.split('/'))}.json")

            with open(file_name, "w") as json_file:
                json.dump(stats_dataset[split], json_file)

            benchmark_artifact.add_file(file_name, split)

        wandb_logger.log_artifact(benchmark_artifact)

    print("Done!")


if __name__ == "__main__":
    main()
