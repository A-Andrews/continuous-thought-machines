from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np
import importlib.util
import os
import sys


class VGDLGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        game_name: str,
        game_folder: str,
        obs_size: int = 84,
        grayscale: bool = True,
        flatten: bool = False,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self._game_name = game_name
        self._game_folder = game_folder
        self._obs_size = obs_size
        self._grayscale = grayscale
        self._flatten = flatten
        self._render_mode = render_mode

        self._env = _get_vgdl_env_class(game_folder)(game_name, game_folder)
        self._env.set_level(0)

        self.action_space = gym.spaces.Discrete(len(self._env.actions))

        if grayscale:
            obs_shape = (1, obs_size, obs_size)
        else:
            obs_shape = (3, obs_size, obs_size)

        if flatten:
            obs_shape = (int(np.prod(obs_shape)),)

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

    def _obs_from_render(self) -> np.ndarray:
        frame = self._env.render()
        frame = cv2.resize(
            frame, (self._obs_size, self._obs_size), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = frame.mean(axis=2, keepdims=True)
        frame = np.transpose(frame, (2, 0, 1))
        obs = frame.astype(np.float32) / 255.0
        if self._flatten:
            obs = obs.reshape(-1)
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._env.reset()
        return self._obs_from_render(), {}

    def step(self, action):
        reward, ended, win = self._env.step(int(action))
        obs = self._obs_from_render()
        terminated = bool(ended)
        truncated = False
        info = {"win": bool(win)}
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()


_VGDL_ENV_CLASS = None


def _get_vgdl_env_class(game_folder: str):
    global _VGDL_ENV_CLASS
    if _VGDL_ENV_CLASS is not None:
        return _VGDL_ENV_CLASS

    rc_rl_root = os.path.abspath(os.path.join(game_folder, os.pardir))
    if rc_rl_root not in sys.path:
        sys.path.insert(0, rc_rl_root)

    utils_path = os.path.join(rc_rl_root, "utils.py")
    if not os.path.exists(utils_path):
        raise FileNotFoundError(f"RC_RL utils.py not found at {utils_path}")

    spec_utils = importlib.util.spec_from_file_location("rc_rl_utils", utils_path)
    if spec_utils is None or spec_utils.loader is None:
        raise ImportError(f"Failed to create module spec for {utils_path}")
    rc_utils = importlib.util.module_from_spec(spec_utils)
    spec_utils.loader.exec_module(rc_utils)

    previous_utils = sys.modules.get("utils")
    sys.modules["utils"] = rc_utils
    try:
        vgdl_env_path = os.path.join(rc_rl_root, "VGDLEnv.py")
        spec_env = importlib.util.spec_from_file_location("rc_rl_VGDLEnv", vgdl_env_path)
        if spec_env is None or spec_env.loader is None:
            raise ImportError(f"Failed to create module spec for {vgdl_env_path}")
        vgdl_env_module = importlib.util.module_from_spec(spec_env)
        spec_env.loader.exec_module(vgdl_env_module)
        _VGDL_ENV_CLASS = vgdl_env_module.VGDLEnv
    finally:
        if previous_utils is not None:
            sys.modules["utils"] = previous_utils
        else:
            sys.modules.pop("utils", None)

    return _VGDL_ENV_CLASS
