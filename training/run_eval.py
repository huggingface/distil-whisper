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
Evaluating a Whisper model on one or more speech recognition datasets.
"""
# You can also adapt this script for your own speech recognition validation. Pointers for this are left as comments.

import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
import torch
import transformers
from datasets import DatasetDict, IterableDatasetDict, load_dataset
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    is_wandb_available,
    pipeline,
    set_seed,
)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer, BasicTextNormalizer
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperForCausalLM
from transformers.utils import check_min_version, is_accelerate_available
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.34.0.dev0")

require_version("datasets>=2.14.6", "To fix: `pip install --upgrade datasets`")

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
    subfolder: str = field(
        default="",
        metadata={"help": "If specified load weights from a subfolder in the model repository"},
    )
    model_variant: str = field(
        default=None,
        metadata={"help": "If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. "},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    assistant_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The name of the assistant model to use to do speculative decoding. If None, no speculative decoding will be done."
        },
    )
    dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized"
                " and the computations run. Choose one of `[float32, float16, bfloat16]`."
            )
        },
    )
    use_pipeline: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate with Transformers pipeline"},
    )
    chunk_length_s: float = field(
        default=30.0, metadata={"help": "Chunk length to use when `use_pipeline` is enabled."}
    )
    return_timestamps: bool = field(
        default=True,
        metadata={
            "help": "Whether to decode with timestamps. This can help for improved WER for long form evaluation."
        },
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language for multilingual evaluation. This argument should be set for multilingual evaluation "
                "only. For English speech recognition, it should be left as `None`."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={
            "help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."
            "This argument should be set for multilingual evaluation only. For English speech recognition, it should be left as `None`."
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Which attn type to use: ['eager', 'sdpa', 'flash_attention_2']"},
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "The batch size to be used for generation."},
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "The beam size to be used for evaluation. Set to 1 for greedy, or >1 for beam search."},
    )
    temperature_fallback: bool = field(
        default=True,
        metadata={"help": "Whether to use temperature fallback for evaluation."},
    )
    logprob_threshold: float = field(
        default=-1.0,
        metadata={"help": "Whether to use temperature fallback for evaluation."},
    )
    no_speech_threshold: float = field(
        default=0.6,
        metadata={
            "help": "Only relevant for long-form transcription. If defined, the 'no-speech' token combined with the `logprob_threshold`"
            "is used to determine whether a segment contains only silence. In this case, the transcription for this segment"
            "is skipped."
        },
    )
    compression_ratio_threshold: float = field(
        default=1.35,
        metadata={
            "help": "Only relevant for long-form transcription. If defined, the zlib compression rate of each segment will be computed. If the compression rate of"
            "a segment is higher than `compression_ratio_threshold`, temperature fallback is activated: the generated segment is discarded and the generation is"
            "repeated using a higher temperature. The intuition behind this feature is that segments with very high compression rates"
            "suffer from a lot of repetition. The unwanted repetition can be reduced by injecting more randomness by increasing the temperature. "
            "If `compression_ratio_threshold` is defined make sure that `temperature` is a list of values. The default value for `compression_ratio_threshold` is 1.35."
        },
    )
    condition_on_prev_tokens: bool = field(
        default=False,
        metadata={"help": "Whether to condition on previous tokens or not"},
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
    generation_max_length: int = field(
        default=256, metadata={"help": "Generate up until `generation_max_length` tokens."}
    )
    log_predictions: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to log the ground truths / pred text to the wandb logger."},
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
    seed: int = field(default=42, metadata={"help": "RNG seed for reproducibility."})
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    prompt_text: str = field(
        default=None,
        metadata={
            "help": "Text prompt to condition the generation on. Useful for controlling the style of transcription and predicting named entities."
        },
    )
    precise_tok_per_s: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, compute tok/sec by forcing the number of generated token ids to num_tokens on dummy batches. "
                "If False, computes tok/sec over the entire dataset with variable number of generated tokens."  
            )
        }
    )
    num_tokens: int = field(
        default=20,
        metadata={
            "help": "Number of tokens to generate if computing tok/sec with precise_tok_per_s."
        }
    )
    num_batches: int = field(
        default=100,
        metadata={
            "help": "Number of batches for the tok/sec calculation with precise_tok_per_s"
        }
    )



def write_metric(summary_writer, eval_metrics, step, prefix="eval"):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"{prefix}/{metric_name}", value, step)


def write_wandb_metric(wandb_logger, metrics, prefix):
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    wandb_logger.log(log_metrics)


def write_wandb_pred(
    wandb_logger,
    pred_str,
    label_str,
    norm_pred_str,
    norm_label_str,
    wer_per_sample,
    prefix="eval",
):
    columns = ["WER", "Target", "Pred", "Norm Target", "Norm Pred"]
    # convert str data to a wandb compatible format
    str_data = [
        [wer_per_sample[i], label_str[i], pred_str[i], norm_label_str[i], norm_pred_str[i]]
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

        # we assume that all the datasets we're using derive from the distil-whisper org on the Hub - prepend the org name if necessary
        for i in range(len(dataset_names)):
            ds_name = dataset_names[i]
            dataset_names[i] = f"distil-whisper/{ds_name}" if "/" not in ds_name else ds_name

        dataset_config_names = dataset_config_names.split("+") if dataset_config_names is not None else None
        splits = splits.split("+") if splits is not None else None
        text_column_names = text_column_names.split("+") if text_column_names is not None else None
        dataset_hours = dataset_hours.split("+") if dataset_hours is not None else None

    # basic checks to ensure we've got the right number of datasets/configs/splits/columns/probs
    if dataset_config_names is not None and len(dataset_names) != len(dataset_config_names):
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

    dataset_config_names = (
        dataset_config_names if dataset_config_names is not None else ["default" for _ in range(len(dataset_names))]
    )
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
    logger.setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 3. Set seed for reproducibility
    set_seed(data_args.seed)

    if data_args.use_pipeline and data_args.batch_size > 1:
        raise ValueError("Make sure that `batch_size` is set to 1 when `use_pipeline=True`.")

    has_wandb = is_wandb_available()
    if has_wandb:
        import wandb
        import wandb as wandb_logger

        # store generation HPs for runs
        generation_arguments = {
            "torch_version": str(torch.__version__),
            "transformers_version": str(transformers.__version__),
            "attn_implementation": data_args.attn_implementation,
            "model_name_or_path": data_args.model_name_or_path,
            "subfolder": data_args.subfolder,
            "assistant_model_name_or_path": data_args.assistant_model_name_or_path,
            "seed": data_args.seed,
            "batch_size": data_args.batch_size,
            "num_beams": data_args.num_beams,
            "return_timestamps": data_args.return_timestamps,
            "condition_on_prev_tokens": data_args.condition_on_prev_tokens,
            "temperature_fallback": data_args.temperature_fallback,
            "logprob_threshold": data_args.logprob_threshold,
            "no_speech_threshold": data_args.no_speech_threshold,
            "use_pipeline": data_args.use_pipeline,
            "chunk_length_s": data_args.chunk_length_s,
        }

        # Set up wandb run
        wandb_logger.init(
            project=data_args.wandb_project,
            name=data_args.wandb_name,
            job_type=data_args.wandb_job_type,
            dir=data_args.wandb_dir,
            save_code=data_args.save_code_to_wandb,
            config=generation_arguments,
        )

    else:
        raise ValueError("Wandb logging requires wandb to be installed. Run `pip install wandb` to enable.")

    # 3. Load dataset
    raw_datasets = IterableDatasetDict()

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
    for dataset_dict in tqdm(dataset_names_dict, desc="Loading datasets..."):
        sub_dataset = load_dataset(
            dataset_dict["name"],
            dataset_dict["config"],
            split=dataset_dict["split"],
            cache_dir=data_args.dataset_cache_dir,
            streaming=data_args.streaming,
            num_proc=data_args.preprocessing_num_workers,
        )

        if dataset_dict["text_column_name"] not in list(sub_dataset.features.keys()):
            raise ValueError(
                f"`--text_column_name` {dataset_dict['text_column_name']} not found in the evaluation "
                f"dataset {dataset_dict['name']}. Ensure `text_column_name` is set to the correct column "
                f"for the target text. Should be one of {' '.join(list(sub_dataset.features.keys()))}"
            )
        if dataset_dict["text_column_name"] != "text":
            sub_dataset = sub_dataset.rename_column(dataset_dict["text_column_name"], "text")
        if not data_args.streaming:
            sub_dataset = sub_dataset.to_iterable_dataset()
        
        # Clean-up the dataset name for pretty logging
        # ("distil-whisper/librispeech_asr", "validation.clean") -> "librispeech_asr/validation-clean"
        pretty_name = f"{dataset_dict['name'].split('/')[-1]}/{dataset_dict['split'].replace('.', '-')}"
        raw_datasets[pretty_name] = sub_dataset

    # 5. Load pretrained model, tokenizer, and feature extractor
    processor = WhisperProcessor.from_pretrained(
        data_args.model_name_or_path,
        subfolder=data_args.subfolder,
        cache_dir=data_args.cache_dir,
        use_fast=data_args.use_fast_tokenizer,
    )
    dtype = getattr(torch, data_args.dtype)
    model = WhisperForConditionalGeneration.from_pretrained(
        data_args.model_name_or_path,
        subfolder=data_args.subfolder,
        torch_dtype=dtype,
        attn_implementation=data_args.attn_implementation,
        low_cpu_mem_usage=is_accelerate_available(),
        cache_dir=data_args.cache_dir,
        variant=data_args.model_variant,
    )
    model.to("cuda:0", dtype=dtype)

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
                data_args.assistant_model_name_or_path,
                torch_dtype=dtype,
                attn_implementation=data_args.attn_implementation,
                low_cpu_mem_usage=is_accelerate_available(),
                cache_dir=data_args.cache_dir,
            )
        else:
            assistant_model = WhisperForCausalLM.from_pretrained(
                data_args.assistant_model_name_or_path,
                torch_dtype=dtype,
                attn_implementation=data_args.attn_implementation,
                low_cpu_mem_usage=is_accelerate_available(),
                cache_dir=data_args.cache_dir,
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
    audio_column_name = data_args.audio_column_name
    normalizer = (
        BasicTextNormalizer() if data_args.language is not None
        else EnglishTextNormalizer(processor.tokenizer.english_spelling_normalizer)
    )
    sampling_rate = processor.feature_extractor.sampling_rate

    if data_args.samples_per_dataset is not None:
        for split in raw_datasets:
            raw_datasets[split] = raw_datasets[split].take(data_args.samples_per_dataset)

    def prepare_dataset(batch):
        # process audio
        audio = [sample["array"].astype(np.float32) for sample in batch[audio_column_name]]

        if model_pipeline is None:
            inputs = processor.feature_extractor(
                audio,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                truncation=False,
                padding="longest",
                return_attention_mask=True,
            )
            if inputs.input_features.shape[-1] < 3000:
                inputs = processor.feature_extractor(
                    audio,
                    sampling_rate=sampling_rate,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
            batch["input_features"] = inputs.input_features.to(dtype)
            batch["attention_mask"] = inputs.attention_mask
        else:
            batch["input_features"] = audio

        # process audio length
        batch["length_in_s"] = [len(sample) / sampling_rate for sample in audio]
        # process targets
        batch["reference"] = batch["text"]
        return batch

    vectorized_datasets = IterableDatasetDict()

    for split in raw_datasets:
        raw_datasets_features = list(raw_datasets[split].features.keys())

        vectorized_datasets[split] = raw_datasets[split].map(
            function=prepare_dataset,
            remove_columns=raw_datasets_features,
            batch_size=data_args.batch_size,
            batched=True,
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

    metric = evaluate.load("wer")

    def compute_metrics(pred_str, label_str):
        # normalize everything and re-compute the WER
        norm_pred_str = [normalizer(pred) for pred in pred_str]
        norm_label_str = [normalizer(label) for label in label_str]

        # filtering step to only evaluate the samples that correspond to non-zero normalized references:
        norm_pred_str = [norm_pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
        norm_label_str = [norm_label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]

        wer = 100 * metric.compute(predictions=norm_pred_str, references=norm_label_str)
        return wer

    gen_kwargs = {
        "max_length": data_args.generation_max_length,
        "return_timestamps": data_args.return_timestamps,
        "num_beams": data_args.num_beams,
        "top_k": 0,
    }

    if hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual:
        gen_kwargs["language"] = data_args.language
        gen_kwargs["task"] = data_args.task
    elif data_args.language is not None:
        raise ValueError(
            "Setting language token for an English-only checkpoint is not permitted. The language argument should "
            "only be set for multilingual checkpoints."
        )

    if assistant_model is not None:
        gen_kwargs["assistant_model"] = assistant_model

    if data_args.prompt_text is not None:
        gen_kwargs["prompt_ids"] = processor.get_prompt_ids(data_args.prompt_text, return_tensors="pt").to("cuda:0")

    long_form_gen_kwargs = {
        "condition_on_prev_tokens": data_args.condition_on_prev_tokens,
        "compression_ratio_threshold": data_args.compression_ratio_threshold,
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if data_args.temperature_fallback else 0,
        "logprob_threshold": data_args.logprob_threshold,
        "no_speech_threshold": data_args.no_speech_threshold,
    }

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        task=data_args.task, 
        language=data_args.language, 
        no_timestamps=data_args.return_timestamps
    )

    def benchmark(batch):
        if model_pipeline is None:
            inputs = torch.stack(batch["input_features"], dim=0).cuda()
            attention_mask = torch.stack(batch["attention_mask"], dim=0).cuda()
            # automatically use long-form args if required
            inner_batch_size, num_mels, seq_len = inputs.shape
            if seq_len == 3000:
                batch_gen_kwargs = gen_kwargs
            else:
                batch_gen_kwargs = {**gen_kwargs, **long_form_gen_kwargs}

            set_seed(data_args.seed)
            start_time = time.time()
            output_ids = model.generate(inputs, attention_mask=attention_mask, **batch_gen_kwargs)
            gen_time = time.time() - start_time

            batch["time"] = inner_batch_size * [(gen_time) / inner_batch_size]

            if not data_args.precise_tok_per_s:
                n_generated_tokens = output_ids.numel() - inner_batch_size * len(forced_decoder_ids)
                batch["tokens_per_sec"] = inner_batch_size * [(n_generated_tokens / gen_time) / inner_batch_size]

            batch["transcription"] = processor.batch_decode(
                output_ids, skip_special_tokens=True, decode_with_timestamps=data_args.return_timestamps
            )

        else:
            inputs = batch["input_features"]
            # Time forward: let's make sure that only forward is timed and not pre- and post-processing
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

        batch["num_words"] = [len(r.split()) for r in batch["reference"]]
        return batch

    result_datasets = DatasetDict()

    for split in vectorized_datasets:
        result_datasets[split] = vectorized_datasets[split].map(
            function=benchmark,
            remove_columns=["input_features"],
            batch_size=data_args.batch_size,
            batched=True,
        )

    stats_dataset = DatasetDict()

    all_stats = {"rtf": 0, "wer": 0, "tokens_per_sec": 0}
    rtf_stats = {
        "times_audio_total": 0,
        "times_transcription_total": 0,
    }

    def benchmark_gen_time():
        if model_pipeline is None:
            dummy_encoder_outputs = BaseModelOutput(
                torch.randn((data_args.batch_size, model.config.max_source_positions, model.config.d_model),
                             dtype=model.dtype,
                             device=model.device
                )            
            )

            # benchmark time to generate fixed number of tokens
            n_tokens = data_args.num_tokens
            start_time = time.time()
            _ = model.generate(
                encoder_outputs=dummy_encoder_outputs,
                min_new_tokens=n_tokens,
                max_new_tokens=n_tokens,
                **gen_kwargs
            )
            gen_time = time.time() - start_time

            n_generated_tokens = n_tokens * data_args.batch_size
            tokens_per_sec = n_generated_tokens / gen_time
        
        return tokens_per_sec

    logger.info("***** Running Evaluation *****")
    for key in generation_arguments:
        logger.info(f"  {key}: {generation_arguments[key]}")

    datasets_evaluated_progress_bar = tqdm(result_datasets, desc="Datasets", position=0)
    for split in datasets_evaluated_progress_bar:
        
        transcriptions = []
        references = []
        stats = {}
        times_audio_total = 0
        times_transcription_total = 0
        tokens_per_secs = []

        if data_args.precise_tok_per_s:
            # evaluate generation speed for few batch
            for _ in range(data_args.num_batches):
                tokens_per_secs.append(benchmark_gen_time())

        datasets_evaluated_progress_bar.write(f"Start benchmarking {split}...")
        result_iter = iter(result_datasets[split])
        for result in tqdm(result_iter, desc="Samples", position=1):
            times_audio_total += result["length_in_s"]
            times_transcription_total += result["time"]
            # ensure prompt is removed from the transcription (awaiting fix in Transformers)
            if data_args.prompt_text is not None:
                result["transcription"] = result["transcription"].replace(data_args.prompt_text, "")
            transcriptions.append(result["transcription"])
            references.append(result["reference"])
            if not data_args.precise_tok_per_s:
                tokens_per_secs.append(result["tokens_per_sec"])

        norm_transcriptions = [normalizer(pred) for pred in transcriptions]
        norm_references = [normalizer(label) for label in references]

        transcriptions = [transcriptions[i] for i in range(len(transcriptions)) if len(norm_references[i]) > 0]
        references = [references[i] for i in range(len(references)) if len(norm_references[i]) > 0]

        norm_transcriptions = [
            norm_transcriptions[i] for i in range(len(norm_transcriptions)) if len(norm_references[i]) > 0
        ]
        norm_references = [norm_references[i] for i in range(len(norm_references)) if len(norm_references[i]) > 0]

        stats["wer"] = compute_metrics(norm_transcriptions, norm_references)

        wer_per_sample = []
        for pred, ref in zip(norm_transcriptions, norm_references):
            wer_per_sample.append(compute_metrics([pred], [ref]))

        stats["rtf"] = times_audio_total / times_transcription_total
        stats["tokens_per_sec"] = sum(tokens_per_secs) / len(tokens_per_secs) 
        stats_dataset[split] = stats

        wer_desc = " ".join([f"Eval {key}: {value} |" for key, value in stats.items()])
        datasets_evaluated_progress_bar.write(wer_desc)

        write_wandb_metric(wandb_logger, stats, prefix=split)

        if data_args.log_predictions:
            write_wandb_pred(
                wandb_logger,
                transcriptions,
                references,
                norm_transcriptions,
                norm_references,
                wer_per_sample,
                prefix=split,
            )

        rtf_stats["times_audio_total"] += times_audio_total
        rtf_stats["times_transcription_total"] += times_transcription_total
        all_stats["wer"] += stats["wer"]
        all_stats["tokens_per_sec"] += stats["tokens_per_sec"]

    all_stats["wer"] = all_stats["wer"] / len(result_datasets)
    # technically this is the reciprocal of the RTF, but it makes the scale easier to read on wandb
    all_stats["rtf"] = rtf_stats["times_audio_total"] / rtf_stats["times_transcription_total"]
    all_stats["tokens_per_sec"] = all_stats["tokens_per_sec"] / len(result_datasets)

    stats_dataset["all"] = all_stats

    write_wandb_metric(wandb_logger, all_stats, prefix="all")

    benchmark_artifact = wandb.Artifact("Benchmark", type="datasets")
    with tempfile.TemporaryDirectory() as temp_dir:
        for split in stats_dataset:
            file_name = os.path.join(temp_dir, f"{'_'.join(split.split('/'))}.json")

            with open(file_name, "w") as json_file:
                json.dump(stats_dataset[split], json_file)

            benchmark_artifact.add_file(file_name, split)

        wandb_logger.log_artifact(benchmark_artifact)


if __name__ == "__main__":
    main()
