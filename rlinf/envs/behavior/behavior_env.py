import json
import os

import cv2
import gymnasium as gym
import numpy as np
import torch
from av.container import Container
from av.stream import Stream
from omegaconf import OmegaConf, open_dict
from omnigibson.envs import VectorEnvironment 
from omnigibson.learning.utils.eval_utils import (
    TASK_INDICES_TO_NAMES,
    ROBOT_CAMERA_NAMES,
    PROPRIOCEPTION_INDICES,
    flatten_obs_dict,
    TASK_NAMES_TO_INDICES,
)
from omnigibson.learning.utils.obs_utils import (
    create_video_writer,
    write_video,
)
from omnigibson.macros import gm
import omnigibson.utils.transform_utils as T
from PIL import Image

from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor
from rlinf.utils.logging import get_logger

from omnigibson.envs import Environment, EnvironmentWrapper
from omnigibson.learning.utils.eval_utils import HEAD_RESOLUTION, WRIST_RESOLUTION
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
import omnigibson as og

# 导入 DISABLED_TRANSITION_RULES，与 openpi-comet eval_custom.py 对齐
from gello.robots.sim_robot.og_teleop_cfg import DISABLED_TRANSITION_RULES

logger = create_module_logger("RGBWrapper")

# ============================================================
# 全局 macros 设置：与 openpi-comet eval_custom.py 完全对齐
# ============================================================
gm.HEADLESS = True
gm.ENABLE_FLATCACHE = True  # 与 openpi-comet 对齐：启用 flatcache 加速渲染
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True
# 注意：移除 gm.ENABLE_OBJECT_STATES = True，openpi-comet 推理时不设置此项

# Image resize target size (match openpi-comet)
RESIZE_SIZE = 224
ROLLOUT_CAMERA_NAMES = ("head", "left_wrist", "right_wrist")


# ---------- PIL-based resize_with_pad (与 openpi-comet openpi_client.image_tools 完全一致) ----------
def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method=Image.BILINEAR) -> Image.Image:
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)
    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    return zero_image


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """PIL-based resize_with_pad, 与 openpi-comet openpi_client.image_tools.resize_with_pad 一致。"""
    if images.shape[-3:-1] == (height, width):
        return images
    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([np.asarray(_resize_with_pad_pil(Image.fromarray(im), height, width, method=method)) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])

__all__ = ["BehaviorEnv"]


class RGBWrapper(EnvironmentWrapper):
    """
    Match eval sensor settings (resolution + head aperture), then reload obs space.
    """

    def __init__(self, env: Environment):
        super().__init__(env=env)

        robot = env.robots[0]

        def resolve_sensor_key(camera_name: str) -> str:
            # take the last part after '::' as suffix
            suffix = camera_name.split("::")[-1]  # e.g. "robot_r1:left_realsense_link:Camera:0" OR "left_realsense_link:Camera:0"

            keys = list(robot.sensors.keys())

            # 1) exact match
            if suffix in robot.sensors:
                return suffix

            # 2) if suffix has "robot_r1:" prefix, strip it
            if ":" in suffix:
                maybe_stripped = suffix.split(":", 1)[1]  # drop "robot_r1:"
                if maybe_stripped in robot.sensors:
                    return maybe_stripped

            # 3) suffix match (handles "robot_xxx:left_realsense_link:Camera:0")
            cand = [k for k in keys if k.endswith(suffix)]
            if cand:
                cand.sort(key=len)
                return cand[0]

            # 4) suffix match after stripping "robot_r1:"
            if ":" in suffix:
                stripped = suffix.split(":", 1)[1]
                cand2 = [k for k in keys if k.endswith(stripped)]
                if cand2:
                    cand2.sort(key=len)
                    return cand2[0]

            # 5) fail: print some keys for debugging
            sample = keys[:30]
            raise KeyError(
                f"[RGBWrapper] Cannot resolve sensor key for camera_name={camera_name!r} "
                f"(suffix={suffix!r}). robot.sensors keys sample: {sample}"
            )

        for camera_id, camera_name in ROBOT_CAMERA_NAMES["R1Pro"].items():
            sensor_key = resolve_sensor_key(camera_name)

            if camera_id == "head":
                robot.sensors[sensor_key].horizontal_aperture = 40.0
                robot.sensors[sensor_key].image_height = HEAD_RESOLUTION[0]
                robot.sensors[sensor_key].image_width = HEAD_RESOLUTION[1]
            else:
                robot.sensors[sensor_key].image_height = WRIST_RESOLUTION[0]
                robot.sensors[sensor_key].image_width = WRIST_RESOLUTION[1]

        env.load_observation_space()
        logger.info("Reloaded observation space!")
        
        # ---- 必须加：VecEnv.reset 会传 get_obs=... 进来 ----
    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    # ---- 可选但建议：step 也透传，避免以后 n_render_iterations 等参数出问题 ----
    def step(self, action, *args, **kwargs):
        return self.env.step(action, *args, **kwargs)

    def observation_spec(self, *args, **kwargs):
        return self.env.observation_spec(*args, **kwargs)

class BehaviorEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        self.cfg = cfg

        self.num_envs = num_envs
        self.ignore_terminations = cfg.ignore_terminations
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.record_metrics = record_metrics
        self._is_start = True

        self.logger = get_logger()

        self.auto_reset = cfg.auto_reset
        if self.record_metrics:
            self._init_metrics()

        # record total number and success number of trials and trial time
        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0

        self._init_env()

        # video
        self._video_writer = None
        self._rollout_video_writers = None
        self.video_cnt = 0
        if self.cfg.video_cfg.save_video:
            os.makedirs(str(self.cfg.video_cfg.video_base_dir), exist_ok=True)
            self._create_video_writer()
            self._create_rollout_video_writers()

        # cache obs like Evaluator does (policy sees self.obs)
        self.obs = None
        
        # ============================================================
        # 任务实例加载配置（与 openpi-comet eval_custom.py 对齐）
        # ============================================================
        self.use_task_instances = getattr(cfg, 'use_task_instances', False)
        self.random_task_instance = getattr(cfg, 'random_task_instance', True)
        
        # 初始化可用的任务实例 ID 列表
        task_instance_ids = getattr(cfg, 'task_instance_ids', None)
        if task_instance_ids is not None:
            self.available_instance_ids = list(task_instance_ids)
        else:
            # 默认使用所有训练实例 (0-199)
            self.available_instance_ids = list(range(200))
        
        # 顺序测试时的实例索引计数器
        self._instance_counter = 0
        
        self.logger.info(f"Task instance loading: use_task_instances={self.use_task_instances}, "
                        f"random_task_instance={self.random_task_instance}, "
                        f"available_instance_ids={len(self.available_instance_ids)} instances")

    def _load_tasks_cfg(self):
        with open_dict(self.cfg):
            self.cfg.omnigibson_cfg["task"]["activity_name"] = TASK_INDICES_TO_NAMES[self.cfg.task_idx]

        # Read task description
        task_description_path = os.path.join(os.path.dirname(__file__), "behavior_task.jsonl")
        with open(task_description_path, "r") as f:
            text = f.read()
            task_description = [json.loads(x) for x in text.strip().split("\n") if x]
        task_description_map = {task_description[i]["task_name"]: task_description[i]["task"] for i in range(len(task_description))}
        self.task_description = task_description_map[self.cfg.omnigibson_cfg["task"]["activity_name"]]

    def _init_env(self):
        # ============================================================
        # 与 openpi-comet eval_custom.py 对齐：禁用特定的 transition rules
        # ============================================================
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False
        
        # 对齐 Evaluator：任务名要先写进 cfg，然后创建 env
        self._load_tasks_cfg()
        self.env = VectorEnvironment(
            self.num_envs,
            OmegaConf.to_container(self.cfg.omnigibson_cfg, resolve=True),
        )

        # ---- 关键：给每个 sub-env 套上 RGBWrapper（保持并行逻辑不变） ----
        # 兼容不同 VectorEnvironment 实现（你贴的 OG 实现是 self.envs）
        subenv_list = None
        for attr in ("envs", "_envs", "environments", "_environments"):
            if hasattr(self.env, attr):
                subenv_list = getattr(self.env, attr)
                break
        if subenv_list is None:
            raise RuntimeError(
                "Cannot find sub-env list inside VectorEnvironment. "
                "Tried attrs: envs/_envs/environments/_environments"
            )

        for i in range(len(subenv_list)):
            # 直接原地替换成 wrapper（wrapper 仍然可 step/reset/observation_spec）
            subenv_list[i] = RGBWrapper(subenv_list[i])

    # -----------------------------
    # Helpers to align with Evaluator
    # -----------------------------
    def _get_subenv(self, env_idx: int):
        """
        Best-effort access to the underlying single Environment for env_idx.
        VectorEnvironment implementations differ; try common attribute names.
        """
        for attr in ("envs", "_envs", "environments", "_environments"):
            if hasattr(self.env, attr):
                sub = getattr(self.env, attr)
                try:
                    return sub[env_idx]
                except Exception:
                    pass
        # Some impls expose a getter
        if hasattr(self.env, "get_env"):
            try:
                return self.env.get_env(env_idx)
            except Exception:
                pass
        return None

    def _get_robot_from_subenv(self, subenv):
        """
        Align with Evaluator.load_robot(): robot named 'robot_r1'
        """
        if subenv is None:
            return None
        try:
            return subenv.scene.object_registry("name", "robot_r1")
        except Exception:
            return None

    def load_task_instance(self, instance_id: int, env_idx: int = 0) -> None:
        """
        与 openpi-comet eval_custom.py 的 load_task_instance 对齐。
        加载特定任务实例的配置（机器人位置、物体状态等）。
        
        Args:
            instance_id (int): 任务实例 ID
            env_idx (int): 要加载的子环境索引，默认为 0
        """
        subenv = self._get_subenv(env_idx)
        if subenv is None:
            self.logger.warning(f"Cannot get subenv for env_idx={env_idx}, skipping load_task_instance")
            return
        
        # 获取底层环境（如果是 wrapper 的话需要展开）
        env = subenv.env if hasattr(subenv, 'env') else subenv
        
        robot = self._get_robot_from_subenv(subenv)
        if robot is None:
            self.logger.warning(f"Cannot get robot for env_idx={env_idx}, skipping load_task_instance")
            return
        
        try:
            scene_model = env.task.scene_name
            tro_filename = env.task.get_cached_activity_scene_filename(
                scene_model=scene_model,
                activity_name=env.task.activity_name,
                activity_definition_id=env.task.activity_definition_id,
                activity_instance_id=instance_id,
            )
            
            tro_file_path = os.path.join(
                get_task_instance_path(scene_model),
                f"json/{scene_model}_task_{env.task.activity_name}_instances/{tro_filename}-tro_state.json",
            )
            
            with open(tro_file_path) as f:
                tro_state = recursively_convert_to_torch(json.load(f))
            
            for tro_key, tro_state_item in tro_state.items():
                if tro_key == "robot_poses":
                    presampled_robot_poses = tro_state_item
                    robot_pos = presampled_robot_poses[robot.model_name][0]["position"]
                    robot_quat = presampled_robot_poses[robot.model_name][0]["orientation"]
                    robot.set_position_orientation(robot_pos, robot_quat)
                    
                    # Write robot poses to scene metadata
                    env.scene.write_task_metadata(key=tro_key, data=tro_state_item)
                else:
                    env.task.object_scope[tro_key].load_state(tro_state_item, serialized=False)
            
            # 确保所有任务相关物体稳定（与 openpi-comet 对齐）
            for _ in range(25):
                og.sim.step_physics()
                for entity in env.task.object_scope.values():
                    if not entity.is_system and entity.exists:
                        entity.keep_still()
            
            env.scene.update_initial_file()
            env.scene.reset()
            
            self.logger.info(f"Loaded task instance {instance_id} for env_idx={env_idx}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load task instance {instance_id} for env_idx={env_idx}: {e}")

    def load_task_instances_for_all_envs(self, instance_ids: list[int]) -> None:
        """
        为所有子环境加载任务实例。
        
        Args:
            instance_ids (list[int]): 每个子环境对应的任务实例 ID 列表，长度应等于 num_envs
        """
        assert len(instance_ids) == self.num_envs, \
            f"instance_ids length ({len(instance_ids)}) must equal num_envs ({self.num_envs})"
        
        for env_idx, instance_id in enumerate(instance_ids):
            self.load_task_instance(instance_id, env_idx)

    def _compute_cam_rel_poses(self, env_idx: int):
        """
        Align with Evaluator._preprocess_obs(): build robot_r1::cam_rel_poses from synced camera parameters.
        If unavailable, return None (do not crash training).
        """
        subenv = self._get_subenv(env_idx)
        robot = self._get_robot_from_subenv(subenv)
        if robot is None:
            return None

        try:
            base_pose = robot.get_position_orientation()
            cam_rel_poses = []
            for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
                # camera_name like "robot_r1::zed_link:Camera:0"
                sensor_key = camera_name.split("::")[1]
                camera = robot.sensors[sensor_key]
                direct_cam_pose = camera.camera_parameters.get("cameraViewTransform", None)

                if direct_cam_pose is None:
                    cam_pose = camera.get_position_orientation()
                    cam_rel_poses.append(torch.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
                    continue

                direct_cam_pose = np.array(direct_cam_pose)
                if np.allclose(direct_cam_pose, np.zeros(16)):
                    cam_pose = camera.get_position_orientation()
                    cam_rel_poses.append(torch.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
                else:
                    mat = np.reshape(direct_cam_pose, [4, 4]).T
                    cam_pose = T.mat2pose(torch.tensor(np.linalg.inv(mat), dtype=torch.float32))
                    cam_rel_poses.append(torch.cat(T.relative_pose_transform(*cam_pose, *base_pose)))

            return torch.cat(cam_rel_poses, dim=-1)
        except Exception:
            return None

    def _preprocess_obs_single(self, env_idx: int, obs: dict) -> dict:
        """
        Align with Evaluator._preprocess_obs(): flatten + add cam_rel_poses + task_id
        """
        flat = flatten_obs_dict(obs)

        # add cam_rel_poses if possible
        cam_rel = self._compute_cam_rel_poses(env_idx)
        if cam_rel is not None:
            flat["robot_r1::cam_rel_poses"] = cam_rel

        # add task_id (Evaluator uses TASK_NAMES_TO_INDICES[self.cfg.task.name])
        # Here we derive task name from cfg.omnigibson_cfg["task"]["activity_name"]
        task_name = self.cfg.omnigibson_cfg["task"]["activity_name"]
        if task_name in TASK_NAMES_TO_INDICES:
            flat["task_id"] = torch.tensor([TASK_NAMES_TO_INDICES[task_name]], dtype=torch.int64)
        return flat

    # -----------------------------
    # Obs extraction / wrapping (align semantics)
    # -----------------------------
    def _extract_obs_image(self, raw_obs):
        """
        Align with Evaluator: use ROBOT_CAMERA_NAMES + '::rgb' from flattened obs.
        Return structure kept as BehaviorEnv expects (main_images / wrist_images / state).
        """
        # raw_obs is per-env nested dict, preprocess to flat dict
        # NOTE: env_idx unknown here; we will pass env_idx in _wrap_obs via closure
        raise RuntimeError("_extract_obs_image should be called via _extract_obs_image_with_idx")

    def _extract_obs_image_with_idx(self, env_idx: int, raw_obs: dict):
        flat = self._preprocess_obs_single(env_idx, raw_obs)

        # ---- helper: pick rgb key by suffix, prefer robot_*::robot_*:... over external sensors ----
        def pick_rgb_key_by_suffix(flat_dict: dict, suffix: str) -> str | None:
            # 1) prefer keys that look like robot_<id>::robot_<id>...<suffix>
            robot_candidates = []
            other_candidates = []
            for k in flat_dict.keys():
                if not k.endswith(suffix):
                    continue
                if k.startswith("robot_") and ("::robot_" in k):
                    robot_candidates.append(k)
                else:
                    other_candidates.append(k)

            # deterministic preference: shorter key first
            if robot_candidates:
                robot_candidates.sort(key=len)
                return robot_candidates[0]
            if other_candidates:
                other_candidates.sort(key=len)
                return other_candidates[0]
            return None

        # These suffixes match your actual keys sample:
        # robot_mkexiw::robot_mkexiw:zed_link:Camera:0::rgb
        head_key = pick_rgb_key_by_suffix(flat, ":zed_link:Camera:0::rgb")
        left_key = pick_rgb_key_by_suffix(flat, ":left_realsense_link:Camera:0::rgb")
        right_key = pick_rgb_key_by_suffix(flat, ":right_realsense_link:Camera:0::rgb")

        if head_key is None or left_key is None or right_key is None:
            keys_sample = list(flat.keys())[:30]
            raise AssertionError(
                "Missing rgb keys in obs. Need suffixes: "
                "':zed_link:Camera:0::rgb', ':left_realsense_link:Camera:0::rgb', ':right_realsense_link:Camera:0::rgb'. "
                f"Found head_key={head_key}, left_key={left_key}, right_key={right_key}. "
                f"Got keys sample: {keys_sample}"
            )

        # Infer robot prefix from the chosen head_key, e.g. "robot_mkexiw"
        # head_key looks like: "<robot>::<robot>:zed_link:Camera:0::rgb"
        robot_prefix = head_key.split("::", 1)[0]  # "robot_mkexiw"

        # ---- Proprio: prefer "<robot_prefix>::proprio", else any key containing 'proprio' ----
        state = None
        preferred_proprio = f"{robot_prefix}::proprio"
        if preferred_proprio in flat:
            state = flat[preferred_proprio]
        else:
            proprio_keys = [k for k in flat.keys() if "proprio" in k]
            if proprio_keys:
                # prefer same robot prefix if possible
                proprio_keys.sort(
                    key=lambda x: (
                        0 if x.startswith(robot_prefix + "::") else 1,
                        len(x),
                    )
                )
                state = flat[proprio_keys[0]]

        assert state is not None, (
            "state is not found in the observation which is required for the behavior training."
        )

        # Keep state slicing behavior
        if isinstance(state, np.ndarray):
            state_t = torch.from_numpy(state)
        else:
            state_t = state if torch.is_tensor(state) else torch.tensor(state)

        # Images: convert to numpy uint8 [H,W,3] then PIL resize (与 openpi-comet 完全一致)
        def to_u8_np(x):
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            x = np.asarray(x)
            if x.dtype != np.uint8:
                mx = float(x.max()) if x.size > 0 else 0.0
                if mx <= 1.0:
                    x = (x * 255.0).astype(np.uint8)
                else:
                    x = np.clip(x, 0, 255).astype(np.uint8)
            return x[..., :3]

        zed_np = to_u8_np(flat[head_key])
        left_np = to_u8_np(flat[left_key])
        right_np = to_u8_np(flat[right_key])

        # Resize images to RESIZE_SIZE x RESIZE_SIZE using PIL (与 openpi-comet 完全一致)
        zed_image = torch.from_numpy(resize_with_pad(zed_np, RESIZE_SIZE, RESIZE_SIZE).copy())
        left_image = torch.from_numpy(resize_with_pad(left_np, RESIZE_SIZE, RESIZE_SIZE).copy())
        right_image = torch.from_numpy(resize_with_pad(right_np, RESIZE_SIZE, RESIZE_SIZE).copy())

        # Store the last flattened obs for env0 so _write_video can match Evaluator behavior
        if env_idx == 0:
            self.obs = flat

        # Return format aligned with openpi-comet B1kInputs
        return {
            "egocentric_camera": zed_image,  # [RESIZE_SIZE, RESIZE_SIZE, C]
            "wrist_image_left": left_image,  # [RESIZE_SIZE, RESIZE_SIZE, C]
            "wrist_image_right": right_image,  # [RESIZE_SIZE, RESIZE_SIZE, C]
            "state": state_t,  # full proprio state for extract_state_from_proprio
            "flat": flat,
        }


    def _wrap_obs(self, obs_list):
        """
        Wrap list of per-env raw obs into batched dict, aligned with openpi-comet B1kInputs format.
        Images are resized to RESIZE_SIZE x RESIZE_SIZE (224x224) to match openpi-comet.
        """
        extracted_obs_list = []
        for env_idx, obs in enumerate(obs_list):
            extracted_obs_list.append(self._extract_obs_image_with_idx(env_idx, obs))

        # Format aligned with openpi-comet B1kInputs expectations
        obs = {
            "egocentric_camera": torch.stack([x["egocentric_camera"] for x in extracted_obs_list], dim=0),  # [N_ENV, 224, 224, C]
            "wrist_image_left": torch.stack([x["wrist_image_left"] for x in extracted_obs_list], dim=0),  # [N_ENV, 224, 224, C]
            "wrist_image_right": torch.stack([x["wrist_image_right"] for x in extracted_obs_list], dim=0),  # [N_ENV, 224, 224, C]
            "task_descriptions": [self.task_description for _ in range(self.num_envs)],
            "states": torch.stack([x["state"] for x in extracted_obs_list], dim=0),  # [N_ENV, proprio_dim]
        }
        return obs

    # -----------------------------
    # Gym API
    # -----------------------------
    def reset(self, instance_ids: list[int] | None = None):
        """
        重置环境。
        
        Args:
            instance_ids (list[int] | None): 可选，每个子环境要加载的任务实例 ID。
                如果提供，会在 reset 后加载对应的任务实例配置（与 openpi-comet 评估对齐）。
                长度应等于 num_envs。
                如果为 None 且 use_task_instances=True，则根据 random_task_instance 配置自动选择。
        
        Returns:
            obs: 观测
            infos: 信息字典
        """
        raw_obs, infos = self.env.reset()
        
        # 确定要加载的任务实例 ID
        if instance_ids is None and self.use_task_instances:
            # 根据配置自动选择任务实例
            if self.random_task_instance:
                # 随机选择任务实例（训练用）
                instance_ids = [
                    np.random.choice(self.available_instance_ids) 
                    for _ in range(self.num_envs)
                ]
            else:
                # 顺序选择（测试/评估用）
                # 使用计数器按顺序遍历所有实例
                instance_ids = []
                for _ in range(self.num_envs):
                    idx = self._instance_counter % len(self.available_instance_ids)
                    instance_ids.append(self.available_instance_ids[idx])
                    self._instance_counter += 1
                
                # 检查是否完成一轮完整测试
                if self._instance_counter >= len(self.available_instance_ids):
                    self.logger.info(f"Completed testing all {len(self.available_instance_ids)} instances, "
                                    f"starting next round...")
                    
            self.logger.info(f"Auto-selected task instance IDs: {instance_ids} "
                           f"(counter={self._instance_counter}/{len(self.available_instance_ids)})")
        
        # 如果指定了任务实例 ID，加载对应的任务配置
        if instance_ids is not None:
            self.load_task_instances_for_all_envs(instance_ids)
            # 加载任务实例后需要重新获取观测
            raw_obs = self._get_obs_after_instance_load()
        
        obs = self._wrap_obs(raw_obs)

        # keep metrics behavior same as your old code
        rewards = torch.zeros(self.num_envs, dtype=torch.float32)
        infos = self._record_metrics(rewards, infos)
        self._reset_metrics()
        return obs, infos
    
    def _get_obs_after_instance_load(self):
        """
        在加载任务实例后获取观测。
        由于 load_task_instance 会修改场景状态，需要重新获取观测。
        
        注意：不能调用 env.reset() 因为会覆盖刚加载的实例状态。
        使用零动作 step 来获取当前观测。
        """
        try:
            # 方法1：尝试从 VectorEnvironment 获取观测（不重置）
            if hasattr(self.env, 'get_obs'):
                result = self.env.get_obs()
                # 处理返回值可能是 tuple (obs, info) 的情况
                if isinstance(result, tuple):
                    return result[0]
                return result
        except Exception as e:
            self.logger.warning(f"get_obs() failed: {e}, trying zero-action step")
        
        # 方法2：执行零动作 step 来获取观测（不会显著改变状态）
        try:
            action_dim = getattr(self.cfg, 'action_dim', 23)  # 默认 23 维动作
            zero_action = np.zeros((self.num_envs, action_dim), dtype=np.float32)
            raw_obs, _, _, _, _ = self.env.step(zero_action)
            return raw_obs
        except Exception as e:
            self.logger.error(f"Failed to get obs after instance load: {e}")
            raise


    def step(self, actions=None) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Align with Evaluator:
        - step env with n_render_iterations=1 if supported
        - write video from self.obs (policy-side obs), not raw_obs parsing
        """
        raw_obs, rewards, terminations, truncations, infos = self.env.step(actions)

        obs = self._wrap_obs(raw_obs)

        if self.cfg.video_cfg.save_video:
            self._write_video_from_extracted(obs)   # 直接用提取后的 obs

        infos = self._record_metrics(rewards, infos)
        if self.ignore_terminations:
            terminations[:] = False

        return (
            obs,
            to_tensor(rewards),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        chunk_size = chunk_actions.shape[1]
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []

        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_rewards, terminations, truncations, infos = self.step(actions)
            chunk_rewards.append(step_rewards)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(past_dones, extracted_obs, infos)

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations

        return (
            extracted_obs,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    @property
    def device(self):
        # 按你要求：device 逻辑保持原样
        return "cuda"

    @property
    def elapsed_steps(self):
        return torch.tensor(self.cfg.max_episode_steps)

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    # -----------------------------
    # Video writer (same API, aligned internals)
    # -----------------------------
    @property
    def video_writer(self) -> tuple[Container, Stream]:
        return self._video_writer

    @video_writer.setter
    def video_writer(self, video_writer: tuple[Container, Stream]) -> None:
        if self._video_writer is not None:
            (container, stream) = self._video_writer
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        self._video_writer = video_writer

    @property
    def rollout_video_writers(self) -> dict[str, tuple[Container, Stream]] | None:
        return self._rollout_video_writers

    @rollout_video_writers.setter
    def rollout_video_writers(self, rollout_video_writers: dict[str, tuple[Container, Stream]] | None) -> None:
        if self._rollout_video_writers is not None:
            for _, (container, stream) in self._rollout_video_writers.items():
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
        self._rollout_video_writers = rollout_video_writers

    def _create_video_writer(self) -> None:
        output_dir = os.path.join(self.cfg.video_cfg.video_base_dir, f"seed_{self.seed_offset}")
        os.makedirs(output_dir, exist_ok=True)
        video_name = os.path.join(output_dir, f"{self.video_cnt}.mp4")
        self.video_writer = create_video_writer(
            fpath=video_name,
            resolution=(448, 672),
        )

    def _create_rollout_video_writers(self) -> None:
        output_dir = os.path.join(
            self.cfg.video_cfg.video_base_dir,
            f"seed_{self.seed_offset}",
            f"rollout_{self.video_cnt}",
        )
        os.makedirs(output_dir, exist_ok=True)
        writers: dict[str, tuple[Container, Stream]] = {}
        for camera_name in ROLLOUT_CAMERA_NAMES:
            resolution = HEAD_RESOLUTION if camera_name == "head" else WRIST_RESOLUTION
            writers[camera_name] = create_video_writer(
                fpath=os.path.join(output_dir, f"{camera_name}.mp4"),
                resolution=resolution,
            )
        self.rollout_video_writers = writers

    def flush_video(self, video_sub_dir: str = None) -> None:
        if self.cfg.video_cfg.save_video:
            self.video_writer = None
            self.rollout_video_writers = None
            self.video_cnt += 1
            self._create_video_writer()
            self._create_rollout_video_writers()

    def _write_video_from_extracted(self, extracted_obs: dict) -> None:

        # env0 - use new key names aligned with openpi-comet
        head = extracted_obs["egocentric_camera"][0]     # [H,W,C] uint8 torch
        left = extracted_obs["wrist_image_left"][0]      # [H,W,C] uint8 torch
        right = extracted_obs["wrist_image_right"][0]    # [H,W,C] uint8 torch

        def to_np(x):
            x = x.detach().cpu().numpy()
            if x.dtype != np.uint8:
                mx = float(x.max()) if x.size > 0 else 0.0
                x = (x * 255.0).astype(np.uint8) if mx <= 1.0 else np.clip(x,0,255).astype(np.uint8)
            return np.ascontiguousarray(x)

        left = cv2.resize(to_np(left), (224, 224))
        right = cv2.resize(to_np(right), (224, 224))
        head = cv2.resize(to_np(head), (448, 448))

        frame = np.expand_dims(np.hstack([np.vstack([left, right]), head]), 0)
        write_video(frame, video_writer=self.video_writer, batch_size=1, mode="rgb")
        self._write_rollout_from_flat()

    def _write_rollout_from_flat(self) -> None:
        if self.rollout_video_writers is None or self.obs is None:
            return

        def to_np_u8(x):
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            else:
                x = np.asarray(x)
            if x.dtype != np.uint8:
                mx = float(x.max()) if x.size > 0 else 0.0
                if mx <= 1.0:
                    x = (x * 255.0).astype(np.uint8)
                else:
                    x = np.clip(x, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(x[..., :3])

        for camera_name in ROLLOUT_CAMERA_NAMES:
            key = ROBOT_CAMERA_NAMES["R1Pro"][camera_name] + "::rgb"
            if key not in self.obs:
                continue
            frame = to_np_u8(self.obs[key])
            write_video(
                frame[None, ...],
                video_writer=self.rollout_video_writers[camera_name],
                batch_size=1,
                mode="rgb",
            )


    # -----------------------------
    # Metrics: keep as your current version unless your "correct code" includes different semantics
    # -----------------------------
    def _init_metrics(self):
        self.success_once = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.fail_once = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.returns = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.prev_step_reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
        else:
            mask = torch.ones(self.num_envs, dtype=bool, device=self.device)
        self.prev_step_reward[mask] = 0.0
        if self.record_metrics:
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0

    def _record_metrics(self, rewards, infos):
        info_lists = []
        for env_idx, (reward, info) in enumerate(zip(rewards, infos)):
            episode_info = {
                "success": info.get("done", {}).get("success", False),
                "episode_length": info.get("episode_length", 0),
            }
            self.returns[env_idx] += reward
            if "success" in info:
                self.success_once[env_idx] = (self.success_once[env_idx] | info["success"])
                episode_info["success_once"] = self.success_once[env_idx].clone()
            if "fail" in info:
                self.fail_once[env_idx] = self.fail_once[env_idx] | info["fail"]
                episode_info["fail_once"] = self.fail_once[env_idx].clone()
            episode_info["return"] = self.returns[env_idx].clone()
            episode_info["episode_len"] = self.elapsed_steps.clone()
            episode_info["reward"] = (episode_info["return"] / episode_info["episode_len"])
            if self.ignore_terminations and "success" in info:
                episode_info["success_at_end"] = info["success"]
            info_lists.append(episode_info)

        infos = {"episode": to_tensor(list_of_dict_to_dict_of_list(info_lists))}
        return infos

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = extracted_obs.copy()
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        final_info = infos.copy()
        if getattr(self, "use_fixed_reset_state_ids", False):
            options.update(episode_id=self.reset_state_ids[env_idx])
        extracted_obs, infos = self.reset()
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def update_reset_state_ids(self):
        pass

    # ============================================================
    # 任务实例测试辅助方法
    # ============================================================
    def reset_instance_counter(self):
        """重置任务实例计数器，用于重新开始顺序测试"""
        self._instance_counter = 0
        self.logger.info("Task instance counter reset to 0")
    
    def get_instance_progress(self) -> tuple[int, int]:
        """
        获取当前测试进度
        
        Returns:
            (current_count, total_count): 当前已测试数量和总数量
        """
        return self._instance_counter, len(self.available_instance_ids)
    
    def is_testing_complete(self) -> bool:
        """检查是否已完成一轮完整的顺序测试"""
        return self._instance_counter >= len(self.available_instance_ids)
    
    def get_remaining_instances(self) -> list[int]:
        """获取剩余未测试的实例 ID 列表"""
        start_idx = self._instance_counter % len(self.available_instance_ids)
        return self.available_instance_ids[start_idx:]
