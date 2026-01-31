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
from openpi.shared.image_tools import resize_with_pad_torch

from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor
from rlinf.utils.logging import get_logger

from omnigibson.envs import Environment, EnvironmentWrapper
from omnigibson.learning.utils.eval_utils import HEAD_RESOLUTION, WRIST_RESOLUTION
from omnigibson.utils.ui_utils import create_module_logger

logger = create_module_logger("RGBWrapper")
# Make sure object states are enabled
gm.HEADLESS = True
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

# Image resize target size (match openpi-comet)
RESIZE_SIZE = 224

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
        self.video_cnt = 0
        if self.cfg.video_cfg.save_video:
            os.makedirs(str(self.cfg.video_cfg.video_base_dir), exist_ok=True)
            self._create_video_writer()

        # cache obs like Evaluator does (policy sees self.obs)
        self.obs = None

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

        # Images: keep as torch uint8 [H,W,3]
        def to_u8_t(x):
            if torch.is_tensor(x):
                if x.dtype != torch.uint8:
                    if x.numel() > 0 and float(x.max()) <= 1.0:
                        x = (x * 255.0).to(torch.uint8)
                    else:
                        x = x.to(torch.uint8)
                return x[..., :3]
            x = np.asarray(x)
            if x.dtype != np.uint8:
                mx = float(x.max()) if x.size > 0 else 0.0
                if mx <= 1.0:
                    x = (x * 255.0).astype(np.uint8)
                else:
                    x = np.clip(x, 0, 255).astype(np.uint8)
            return torch.from_numpy(x[..., :3])

        zed_image = to_u8_t(flat[head_key])
        left_image = to_u8_t(flat[left_key])
        right_image = to_u8_t(flat[right_key])

        # Resize images to RESIZE_SIZE x RESIZE_SIZE (match openpi-comet)
        zed_image = resize_with_pad_torch(zed_image, RESIZE_SIZE, RESIZE_SIZE)
        left_image = resize_with_pad_torch(left_image, RESIZE_SIZE, RESIZE_SIZE)
        right_image = resize_with_pad_torch(right_image, RESIZE_SIZE, RESIZE_SIZE)

        # Store the last flattened obs for env0 so _write_video can match Evaluator behavior
        if env_idx == 0:
            self.obs = flat

        return {
            "main_images": zed_image,  # [RESIZE_SIZE, RESIZE_SIZE, C]
            "wrist_images": torch.stack([left_image, right_image], dim=0),  # [2, RESIZE_SIZE, RESIZE_SIZE, C]
            "state": state_t[:32],  # [32]
            "flat": flat,
        }


    def _wrap_obs(self, obs_list):
        """
        Wrap list of per-env raw obs into batched dict, but extraction semantics aligned to Evaluator.
        Images are resized to RESIZE_SIZE x RESIZE_SIZE (224x224) to match openpi-comet.
        """
        extracted_obs_list = []
        for env_idx, obs in enumerate(obs_list):
            extracted_obs_list.append(self._extract_obs_image_with_idx(env_idx, obs))

        obs = {
            "main_images": torch.stack([x["main_images"] for x in extracted_obs_list], dim=0),  # [N_ENV, 224, 224, C]
            "wrist_images": torch.stack([x["wrist_images"] for x in extracted_obs_list], dim=0),  # [N_ENV, 2, 224, 224, C]
            "task_descriptions": [self.task_description for _ in range(self.num_envs)],
            "states": torch.stack([x["state"] for x in extracted_obs_list], dim=0),  # [N_ENV, 32]
        }
        return obs

    # -----------------------------
    # Gym API
    # -----------------------------
    def reset(self):
        raw_obs, infos = self.env.reset()
        obs = self._wrap_obs(raw_obs)

        # keep metrics behavior same as your old code
        rewards = torch.zeros(self.num_envs, dtype=torch.float32)
        infos = self._record_metrics(rewards, infos)
        self._reset_metrics()
        return obs, infos


    def step(self, actions=None) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Align with Evaluator:
        - step env with n_render_iterations=1 if supported
        - write video from self.obs (policy-side obs), not raw_obs parsing
        """
        
        # try:
            # try to pass n_render_iterations like Evaluator
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

    def _create_video_writer(self) -> None:
        output_dir = os.path.join(self.cfg.video_cfg.video_base_dir, f"seed_{self.seed_offset}")
        os.makedirs(output_dir, exist_ok=True)
        video_name = os.path.join(output_dir, f"{self.video_cnt}.mp4")
        self.video_writer = create_video_writer(
            fpath=video_name,
            resolution=(448, 672),
        )

    def flush_video(self, video_sub_dir: str = None) -> None:
        if self.cfg.video_cfg.save_video:
            self.video_writer = None
            self.video_cnt += 1
            self._create_video_writer()

    def _write_video_from_extracted(self, extracted_obs: dict) -> None:

        # env0
        head = extracted_obs["main_images"][0]          # [H,W,C] uint8 torch
        wrists = extracted_obs["wrist_images"][0]       # [2,H,W,C] uint8 torch
        left = wrists[0]
        right = wrists[1]

        def to_np(x):
            x = x.detach().cpu().numpy()
            if x.dtype != np.uint8:
                mx = float(x.max()) if x.size > 0 else 0.0
                x = (x * 255.0).astype(np.uint8) if mx <= 1.0 else np.clip(x,0,255).astype(np.uint8)
            return np.ascontiguousarray(x)

        left = cv2.resize(to_np(left), (224,224))
        right = cv2.resize(to_np(right), (224,224))
        head = cv2.resize(to_np(head), (448,448))

        frame = np.expand_dims(np.hstack([np.vstack([left, right]), head]), 0)
        write_video(frame, video_writer=self.video_writer, batch_size=1, mode="rgb")


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
