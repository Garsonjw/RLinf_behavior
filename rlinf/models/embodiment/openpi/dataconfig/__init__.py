# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# openpi model configs

import dataclasses
import difflib
from typing import Optional
from enum import Enum
from enum import auto
import openpi.transforms as _transforms
import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from collections.abc import Sequence
from openpi.training.config import (
    AssetsConfig,
    DataConfig,
    TrainConfig,
)

from rlinf.models.embodiment.openpi.dataconfig.behavior_dataconfig import (
     LeRobotBehaviorDataConfig
)
from rlinf.models.embodiment.openpi.dataconfig.calvin_dataconfig import (
    LeRobotCalvinDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.franka_dataconfig import (
    CustomDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.libero_dataconfig import (
    LeRobotLiberoDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.maniskill_dataconfig import (
    LeRobotManiSkillDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.metaworld_dataconfig import (
    LeRobotMetaworldDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.robocasa_dataconfig import (
    LeRobotRobocasaDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.robotwin_aloha_dataconfig import (
    LeRobotAlohaDataConfig,
)

class DroidActionSpace(Enum):
    """Action space for DROID dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()

@dataclasses.dataclass(frozen=True)
class BehaviorDataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None

    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None

    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None

    # Only used for B1K data loader.
    behavior_dataset_root: str = None

    # Action space for DROID dataset.
    action_space: DroidActionSpace | None = None

    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None

    # episodes index to use for training
    episodes_index: list[int] | None = None

    # tasks to use for training
    tasks: list[str] | None = None

    # tasks to use for training
    modalities: list[str] = dataclasses.field(default_factory=lambda: ["rgb"])

    # tolerance decoding
    tolerance_s: float = 1e-4

    # fine-grained level of orchestrators to use for training
    fine_grained_level: int = (0,)  # 0, 1, 2

    # whether to return seg instance
    return_seg_instance: bool = False

    # type of rgb to use for training
    train_rgb_type: str = "regular"  # regular | box | point

    # skill list to use for training
    skill_list: list[str] = dataclasses.field(default_factory=lambda: ["all"])

_CONFIGS = [
    TrainConfig(
        name="pi0_libero",
        model=pi0_config.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_libero/assets"),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi0_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_libero/assets"),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
    ),
    TrainConfig(
        name="pi0_maniskill",
        model=pi0_config.Pi0Config(),
        data=LeRobotManiSkillDataConfig(
            repo_id="physical-intelligence/maniskill",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_base"),
            extra_delta_transform=True,
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
        seed=0,
        batch_size=32,
        num_workers=8,
        num_train_steps=200,  # 1_000, #30_000
        log_interval=5,  # 25,
        save_interval=50,  # 200,
    ),
    TrainConfig(
        name="pi05_maniskill",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),  # discrete_state_input=False: stateless policy, True: with state policy
        data=LeRobotManiSkillDataConfig(
            repo_id="physical-intelligence/maniskill",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi05_maniskill/assets"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        seed=0,
        batch_size=256,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_workers=8,
        num_train_steps=5_000,
        log_interval=5,
        save_interval=250,
    ),
    TrainConfig(
        name="pi0_metaworld",
        model=pi0_config.Pi0Config(action_horizon=5),
        data=LeRobotMetaworldDataConfig(
            repo_id="lerobot/metaworld_mt50",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_metaworld/assets"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi0_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
    ),
    TrainConfig(
        name="pi05_metaworld",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=5, discrete_state_input=False
        ),
        data=LeRobotMetaworldDataConfig(
            repo_id="lerobot/metaworld_mt50",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_metaworld/assets"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
    ),
    TrainConfig(
        name="pi0_calvin",
        model=pi0_config.Pi0Config(action_horizon=5),
        data=LeRobotCalvinDataConfig(
            repo_id="InternRobotics/InternData-Calvin_ABC",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_calvin/assets"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi0_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_calvin",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=5, discrete_state_input=False
        ),
        data=LeRobotCalvinDataConfig(
            repo_id="InternRobotics/InternData-Calvin_ABC",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_calvin/assets"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_robocasa",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=LeRobotRobocasaDataConfig(
            repo_id="physical-intelligence/robocasa",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_robocasa/assets"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi0_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_aloha_robotwin",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="robotwin/place_empty_cup_random",
            base_config=DataConfig(
                prompt_from_task=True
            ),  # we need language instruction
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_robotwin/assets"),
            extra_delta_transform=True,  # True for delta action, False for abs_action
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
    ),
    # TrainConfig(
    #     name="pi0_behavior",
    #     model=pi0_config.Pi0Config(),
    #     data=LeRobotBehaviorDataConfig(
    #         repo_id="physical-intelligence/behavior",
    #         base_config=DataConfig(prompt_from_task=True),
    #         assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_behavior/assets"),
    #         extra_delta_transform=True,
    #     ),
    #     weight_loader=weight_loaders.CheckpointWeightLoader(
    #         "checkpoints/jax/pi0_base/params"
    #                 ),
    #     pytorch_weight_path="checkpoints/torch/pi0_base",
    #     num_train_steps=30_000,
    # ),

    TrainConfig(
        name="pi05_behavior",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotBehaviorDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=BehaviorDataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi50_behavior/assets"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("path_to_your_pretrained_checkpoint"),
        num_train_steps=20_000,
        pytorch_weight_path="checkpoints/torch/pi05_base",
    ),
    
    TrainConfig(
        name="pi0_custom",
        model=pi0_config.Pi0Config(),
        data=CustomDataConfig(
            repo_id="physical-intelligence/custom_dataset",
            base_config=DataConfig(
                prompt_from_task=True
            ),  # we need language instruction
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_base/assets"),
            extra_delta_transform=False,  # True for delta action, False for abs_action
            action_train_with_rotation_6d=False,  # User can add extra config in custom dataset
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def _override_with_model_path(config: TrainConfig, model_path: str) -> TrainConfig:
    """Return a copy of the config with assets/weight paths set from model_path."""
    data_config = config.data
    if (
        dataclasses.is_dataclass(data_config)
        and hasattr(data_config, "assets")
        and dataclasses.is_dataclass(data_config.assets)
    ):
        data_config = dataclasses.replace(
            data_config,
            assets=dataclasses.replace(data_config.assets, assets_dir=model_path),
        )

    replace_kwargs = {
        "data": data_config,
        "pytorch_weight_path": model_path,
    }
    if dataclasses.is_dataclass(config) and any(
        field.name == "assets_dirs" for field in dataclasses.fields(config)
    ):
        replace_kwargs["assets_dirs"] = model_path

    return dataclasses.replace(config, **replace_kwargs)


def get_openpi_config(
    config_name: str, model_path: Optional[str] = None, batch_size: Optional[int] = None
) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(
            config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0
        )
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    config = _CONFIGS_DICT[config_name]
    if model_path is not None:
        config = _override_with_model_path(config, model_path)
    if batch_size is not None:
        config = dataclasses.replace(config, batch_size=batch_size)

    return config
