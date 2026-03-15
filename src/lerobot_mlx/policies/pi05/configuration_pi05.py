#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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
"""Pi0.5 policy configuration -- MLX port.

Pi0.5 is identical to Pi0 but with QUANTILES normalization and a larger
tokenizer max length. Inherits from Pi0Config.
"""
from dataclasses import dataclass

from lerobot_mlx.policies.pi0.configuration_pi0 import Pi0Config


@dataclass
class Pi05Config(Pi0Config):
    """Pi0.5 config -- same architecture as Pi0 with quantile normalization.

    Key differences from Pi0:
        - normalization_mode: "quantiles" (vs "mean_std" in Pi0)
        - tokenizer_max_length: 200 (vs 48 in Pi0)
    """

    normalization_mode: str = "quantiles"  # vs "mean_std" in Pi0
    tokenizer_max_length: int = 200  # vs 48 in Pi0
