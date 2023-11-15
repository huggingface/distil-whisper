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
Convert a Flax training state to HF Transformers Whisper weights.
"""

import logging
import os
import sys
from dataclasses import field
from pathlib import Path
from typing import Callable, Optional

import flax
import jax
import jax.numpy as jnp
import optax
from flax import jax_utils, traverse_util
from flax.serialization import from_bytes
from flax.training import train_state
from flax.training.common_utils import shard_prng_key
from huggingface_hub import Repository, create_repo
from optax._src import linear_algebra
from transformers import (
    AutoConfig,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from distil_whisper import FlaxWhisperForConditionalGeneration


# initialise JAX for multi-host set-up on TPU
jax.distributed.initialize()

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
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
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
    load_with_scan_weights: bool = field(
        default=False,
        metadata={
            "help": "Whether the pre-trained checkpoint has its weights stored in scan format. Set to True for scanned "
            "weights, defaults to False for non-scan (unrolled) weights."
        },
    )
    use_scan: bool = field(
        default=True,
        metadata={"help": ("Whether or not to use `scan_with_axes` over the encoder and decoder blocks.")},
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


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray
    max_grad_norm: float

    def apply_gradients(self, *, grads, **kwargs):
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
        g_norm = linear_algebra.global_norm(grads)
        g_norm = jnp.maximum(self.max_grad_norm, g_norm)
        grads = jax.tree_map(lambda t: (t / g_norm) * self.max_grad_norm, grads)

        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))

    def unreplicate(self):
        return jax_utils.unreplicate(self)


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (
            ModelArguments,
            Seq2SeqTrainingArguments,
        )
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

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

    # 5. Load pretrained config, model and processor
    config = AutoConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    student_model, student_params = FlaxWhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        dtype=getattr(jnp, model_args.dtype),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        _do_init=False,
        use_scan=model_args.load_with_scan_weights,
    )

    # enable scan / gradient checkpointing if necessary in the student model
    if model_args.use_scan:
        student_model.enable_scan()  # to enable scan in the nn.Module
        student_params = student_model.convert_unroll_to_scan(student_params)  # to convert the unrolled params to scan

    # Initialize our student state
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    total_train_steps = int(training_args.max_steps)

    # Create learning rate schedule
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        total_train_steps,
        training_args.lr_scheduler_type,
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
    student_state = TrainState.create(
        apply_fn=student_model.__call__,
        params=student_params,
        tx=adamw,
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

    cur_step = int(jax.device_get(student_state.step))

    # save weights in HF Transformers format
    if jax.process_index() == 0:
        student_model.disable_scan()
        student_state_params = student_model.convert_scan_to_unroll(student_state.params)
        student_params = jax.device_get(student_state_params)
        student_model.save_pretrained(
            os.path.join(training_args.output_dir, f"checkpoint-{cur_step}"), params=student_params
        )
        if training_args.push_to_hub:
            repo.push_to_hub(
                commit_message=f"Saving weights of step {cur_step}",
                blocking=False,
            )


if __name__ == "__main__":
    main()
