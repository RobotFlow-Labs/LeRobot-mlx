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
"""Pi0.5 -- identical to Pi0 with QUANTILES normalization.

Pi0.5 inherits the full Pi0 architecture (PaliGemma VLM + Gemma action expert
+ flow matching action head) and only changes the normalization strategy from
mean_std to quantiles. This thin wrapper ensures proper config/class naming.
"""

from lerobot_mlx.policies.pi0.modeling_pi0 import Pi0FlowMatching, Pi0Policy
from lerobot_mlx.policies.pi05.configuration_pi05 import Pi05Config


class Pi05Policy(Pi0Policy):
    """Pi0.5 uses the same architecture as Pi0 with quantile normalization.

    The only behavioural difference is the normalization_mode field on the
    config (``"quantiles"`` instead of ``"mean_std"``). All model weights,
    forward pass, and action generation logic are identical to Pi0.
    """

    config_class = Pi05Config
    name = "pi05"

    def __init__(self, config: Pi05Config | None = None):
        if config is None:
            config = Pi05Config()
        super().__init__(config)
        # Store normalization mode from config for easy access
        self._normalization_mode = getattr(config, "normalization_mode", "quantiles")
