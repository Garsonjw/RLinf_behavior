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
import dataclasses

import einops
import numpy as np
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES

from openpi import transforms
from openpi.models import model as _model


def make_behavior_example() -> dict:
    """Creates a random input example for the Behavior policy.
    Aligned with openpi-comet B1kInputs format.
    """
    return {
        "observation/egocentric_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(23),  # 23-dim state after extract_state_from_proprio
        "prompt": "do something",
    }

def extract_state_from_proprio(proprio_data):
    """
    We assume perfect correlation for the two gripper fingers.
    """
    # extract joint position
    base_qvel = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["base_qvel"]]  # 3
    trunk_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["trunk_qpos"]]  # 4
    arm_left_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["arm_left_qpos"]]  #  7
    arm_right_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["arm_right_qpos"]]  #  7
    left_gripper_width = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_left_qpos"]].sum(axis=-1, keepdims=True)  # 1
    right_gripper_width = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_right_qpos"]].sum(axis=-1, keepdims=True)  # 1
    return np.concatenate([
        base_qvel,
        trunk_qpos,
        arm_left_qpos,
        # left_gripper_width,
        arm_right_qpos,
        left_gripper_width, # NOTE: we rearrange the gripper from 21 to 14 to match the action space
        right_gripper_width,
    ], axis=-1)

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    elif image.shape[0] == 2 and image.shape[1] == 3:
        image = einops.rearrange(image, "n c h w -> n h w c")
    return image


@dataclasses.dataclass(frozen=True)
class BehaviorInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format.
    Aligned with openpi-comet B1kInputs format for compatibility.
    """

    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    meta_image_keys: list[str] = dataclasses.field(default_factory=list)

    depth_as_pcd: bool = False

    def __call__(self, data: dict) -> dict:
        # Extract state from proprio data (aligned with openpi-comet B1kInputs)
        proprio_data = data["observation/state"]
        state = extract_state_from_proprio(proprio_data)

        # Parse images - aligned with openpi-comet B1kInputs format
        # Uses separate keys for each camera instead of stacked wrist images
        base_image = _parse_image(data["observation/egocentric_camera"])  # [h, w, c]
        wrist_image_left = _parse_image(data["observation/wrist_image_left"])  # [h, w, c]
        wrist_image_right = _parse_image(data["observation/wrist_image_right"])  # [h, w, c]

        # Determine image naming based on model type (aligned with openpi-comet)
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image_left, wrist_image_right)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, wrist_image_left, wrist_image_right)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        # Create inputs dict
        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BehaviorOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.
    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """
    action_dim: int = 23
    def __call__(self, data: dict) -> dict:
        # Only return the first 23 dims.
        return {"actions": np.asarray(data["actions"][:, :self.action_dim])}