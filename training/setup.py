# Copyright 2023 The HuggingFace Team. All rights reserved.
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


import os

import setuptools

_deps = [
    "torch>=1.10",
    "transformers>=4.35.1",
    "datasets[audio]>=2.14.7",
    "accelerate>=0.24.1",
    "jiwer",
    "evaluate>=0.4.1",
    "wandb",
    "tensorboard",
    "nltk",
]

_extras_dev_deps = [
    "ruff==0.1.5",
]

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="distil_whisper",
    description="Toolkit for distilling OpenAI's Whisper model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=_deps,
    extras_require={
        "dev": [_extras_dev_deps],
    },
)

