#!/usr/bin/env python

# Copyright 2025 Bryson Jones and The HuggingFace Inc. team. All rights reserved.
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

# lerobot_policy_mtdit — in-repo plugin backporting the multi_task_dit policy from the
# lerobot 0.5.2 checkout (/home/admin_07/project_repos/lerobot_AICchallange/lerobot,
# commit fc6c94c) onto the PINNED lerobot 0.4.4. Activate with
#   PYTHONPATH=<repo>/src  +  --policy.discover_packages_path=lerobot_policy_mtdit
# (on BOTH fresh and resume invocations). Importing this package triggers
# @PreTrainedConfig.register_subclass("multi_task_dit"); 0.4.4's factory then resolves
# modeling/processor modules from the configuration module's name, so the module names
# below must stay verbatim and the package name must not contain "configuration_" or
# "processor_". See docs/PINS.md and docs/decisions.md for the backport record.
from .configuration_multi_task_dit import MultiTaskDiTConfig
from .modeling_multi_task_dit import MultiTaskDiTPolicy
from .processor_multi_task_dit import make_multi_task_dit_pre_post_processors

__all__ = ["MultiTaskDiTConfig", "MultiTaskDiTPolicy", "make_multi_task_dit_pre_post_processors"]
