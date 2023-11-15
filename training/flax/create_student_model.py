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
Initialise a student Whisper model from a pre-trained teacher model for
teacher-student distillation.
"""

import argparse
import copy
import logging

import jax
import numpy as np
from flax.core import freeze, unfreeze
from transformers import GenerationConfig, WhisperFeatureExtractor, WhisperProcessor

from distil_whisper import FlaxWhisperForConditionalGeneration


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Initialise a student Whisper model from a teacher model, copying the relevant layer weights and adjusting the processor as necessary."
    )
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        required=True,
        help="The HF Hub ID of the teacher checkpoint.",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="",
        help="In case the relevant teacher weights are located inside a subfolder of the model repo on huggingface.co, you "
        "can specify the folder name here.",
    )
    parser.add_argument(
        "--encoder_layers",
        type=int,
        default=None,
        help="Number of encoder layers to use in the student model. Defaults to all layers from the teacher.",
    )
    parser.add_argument(
        "--decoder_layers",
        type=int,
        default=2,
        help="Number of decoder layers to use in the student model. Defaults to 2 layers.",
    )
    parser.add_argument(
        "--max_source_positions",
        type=int,
        default=None,
        help="The maximum sequence length of log-mel filter-bank features that this model might ever be used with. Can "
        "be used to create a student model with a shorter context length than the teacher model. Defaults to the number "
        "of source positions in the teacher model (1500).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Where to save the student weights and processor.",
    )
    parser.add_argument(
        "--push_to_hub",
        type=bool,
        required=False,
        default=False,
        help="Whether to push the student weights and processor to the Hub.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where to store the pretrained models downloaded from huggingface.co",
    )

    args = parser.parse_args()
    return args


def init_student_model_from_teacher(
    teacher_checkpoint,
    encoder_layers=None,
    decoder_layers=2,
    max_source_positions=None,
    save_dir=None,
    push_to_hub=None,
    cache_dir=None,
    subfolder="",
):
    teacher_model, teacher_params = FlaxWhisperForConditionalGeneration.from_pretrained(
        teacher_checkpoint,
        _do_init=False,
        cache_dir=cache_dir,
        subfolder=subfolder,
    )
    processor = WhisperProcessor.from_pretrained(teacher_checkpoint)
    generation_config = GenerationConfig.from_pretrained(teacher_checkpoint)

    teacher_config = teacher_model.config
    teacher_encoder_layers = teacher_config.encoder_layers
    teacher_decoder_layers = teacher_config.decoder_layers

    student_config = copy.deepcopy(teacher_config)
    student_config.update(
        {
            "encoder_layers": encoder_layers if encoder_layers is not None else teacher_encoder_layers,
            "decoder_layers": decoder_layers,
            "max_source_positions": max_source_positions
            if max_source_positions is not None
            else student_config.max_source_positions,
        }
    )

    encoder_mapping = np.linspace(0, teacher_encoder_layers - 1, student_config.encoder_layers, dtype=int)
    encoder_mapping[-1] = teacher_encoder_layers - 1

    encoder_map = {}
    for student_layer, teacher_layer in enumerate(encoder_mapping):
        encoder_map[str(teacher_layer)] = str(student_layer)

    decoder_mapping = np.linspace(0, teacher_decoder_layers - 1, student_config.decoder_layers, dtype=int)
    decoder_mapping[-1] = teacher_decoder_layers - 1

    decoder_map = {}
    for student_layer, teacher_layer in enumerate(decoder_mapping):
        decoder_map[str(teacher_layer)] = str(student_layer)

    # init the student params from the teacher model
    student_params = unfreeze(teacher_params)
    student_params["model"]["decoder"]["layers"] = {}

    for layer in teacher_params["model"]["decoder"]["layers"]:
        if layer in decoder_map:
            # re-introduce pre-defined layers from the teacher
            student_params["model"]["decoder"]["layers"][decoder_map[layer]] = teacher_params["model"]["decoder"][
                "layers"
            ][layer]

    if encoder_layers is not None:
        student_params["model"]["encoder"]["layers"] = {}
        for layer in teacher_params["model"]["encoder"]["layers"]:
            if layer in encoder_map:
                # re-introduce pre-defined layers from the teacher
                student_params["model"]["encoder"]["layers"][encoder_map[layer]] = teacher_params["model"]["encoder"][
                    "layers"
                ][layer]

    if max_source_positions is not None:
        # slice the first MAX_SOURCE_POSITIONS embedding weights
        student_params["model"]["encoder"]["embed_positions"]["embedding"] = teacher_params["model"]["encoder"][
            "embed_positions"
        ]["embedding"][: student_config.max_source_positions, :]
        # update the feature extractor to handle the new input length
        chunk_length = int(student_config.max_source_positions * 2 / 100)
        processor.feature_extractor = WhisperFeatureExtractor(chunk_length=chunk_length)

    # remove the teacher params and model
    del teacher_params, teacher_model

    # save the converted weights and model
    student_params = freeze(student_params)
    student_model = FlaxWhisperForConditionalGeneration(student_config, _do_init=False)

    if save_dir is not None:
        student_model.save_pretrained(save_dir, params=student_params)
        # we also need to correctly save the processor and generation config
        processor.save_pretrained(save_dir)
        generation_config.save_pretrained(save_dir)

    # check we can do a forward pass with the saved model - first load the weights and processor
    logger.info("Checking we can load the saved model...")
    student_model, student_params = FlaxWhisperForConditionalGeneration.from_pretrained(
        save_dir,
        _do_init=False,
    )
    processor = WhisperProcessor.from_pretrained(save_dir)

    # define some random inputs
    input_features = processor(np.ones(16000), sampling_rate=16000, return_tensors="np").input_features
    decoder_start_token_id = student_model.config.decoder_start_token_id
    decoder_input_ids = np.ones((input_features.shape[0], 1)) * decoder_start_token_id

    # do a forward pass - outputs will be gibberish for the initialised model so we can't check them
    logger.info("Checking we can run the converted model forward...")
    _ = student_model(input_features, decoder_input_ids=decoder_input_ids, params=student_params).logits
    logger.info("Conversion successful!")

    if push_to_hub:
        student_model.push_to_hub(save_dir, params=student_params)
        processor.push_to_hub(save_dir)
        generation_config.push_to_hub(save_dir)


if __name__ == "__main__":
    args = parse_args()

    # Set the verbosity to info of the logger - we only want one process per machine to log things on the screen
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)

    init_student_model_from_teacher(
        teacher_checkpoint=args.teacher_checkpoint,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        max_source_positions=args.max_source_positions,
        save_dir=args.save_dir,
        push_to_hub=args.push_to_hub,
        cache_dir=args.cache_dir,
        subfolder=args.subfolder,
    )
