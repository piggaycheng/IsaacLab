from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol, Any, cast

import gymnasium as gym
import numpy as np
import torch


class PMTG(Protocol):
    """Protocol for a Policies Modulating Trajectory Generator.

    A PMTG maps (latent, t, optionally state) -> action for a vectorized batch of envs.
    Inputs and outputs are torch.Tensors on the env's device.
    """

    action_dim: int

    def reset(self, env_ids: torch.Tensor | None = None):
        ...

    def __call__(
        self,
        latent: torch.Tensor,  # (num_envs, latent_dim)
        t: torch.Tensor,  # (num_envs,) time in seconds
        state: dict | None = None,
    ) -> torch.Tensor:  # (num_envs, action_dim)
        ...


@dataclass
class SinusoidPMTG:
    """A simple sinusoidal PMTG: a = A*sin(2*pi*f*t + phi) + b.

    The policy output (latent) modulates amplitude, phase, and bias via linear maps.
    """

    action_dim: int
    latent_dim: int
    freq_hz: float = 1.0
    amp_scale: float = 1.0
    bias_scale: float = 0.0
    phase_scale: float = math.pi
    device: torch.device | None = None

    def __post_init__(self):
        # lightweight linear maps (no gradients unless user wants to optimize wrapper)
        rng = torch.Generator().manual_seed(1234)
        self.W_amp = torch.randn(self.latent_dim, self.action_dim, generator=rng) * 0.1
        self.W_bias = torch.randn(self.latent_dim, self.action_dim, generator=rng) * 0.1
        self.W_phase = torch.randn(self.latent_dim, self.action_dim, generator=rng) * 0.1
        self.to(self.device or torch.device("cpu"))

    def to(self, device: torch.device):
        self.device = device
        self.W_amp = self.W_amp.to(device)
        self.W_bias = self.W_bias.to(device)
        self.W_phase = self.W_phase.to(device)
        return self

    def reset(self, env_ids: torch.Tensor | None = None):
        # stateless in this minimal example
        return

    @torch.no_grad()
    def __call__(self, latent: torch.Tensor, t: torch.Tensor, state: dict | None = None) -> torch.Tensor:
        latent = latent.to(self.device)
        t = t.to(self.device)
        amp = torch.tanh(latent @ self.W_amp) * self.amp_scale
        bias = torch.tanh(latent @ self.W_bias) * self.bias_scale
        phase = (latent @ self.W_phase) * self.phase_scale
        omega = 2 * math.pi * self.freq_hz
        # shape alignment
        omega_t = omega * t.unsqueeze(-1)
        a = amp * torch.sin(omega_t + phase) + bias
        return a


class PMTGActionWrapper(gym.ActionWrapper):
    """Gym wrapper that inserts a PMTG between policy action and env action.

    Contract:
    - Expects policy to output a latent of shape (num_envs, latent_dim).
    - Wrapper converts latent -> action via provided PMTG using env time.
    - Respects vectorized IsaacLab envs with ManagerBasedRLEnv semantics.
    """

    def __init__(
        self,
        env: gym.Env,
        pmtg: PMTG,
        latent_space: gym.Space | None = None,
        passthrough_mask: torch.Tensor | None = None,
    ):
        super().__init__(env)
        self.device = getattr(env, "device", torch.device("cpu"))
        self.pmtg = pmtg
        # best-effort device move without strict typing
        if hasattr(self.pmtg, "to"):
            cast(Any, self.pmtg).to(self.device)

        # The policy will see latent_space as action_space; default to Box(-1,1, latent_dim)
        if latent_space is None:
            latent_dim = getattr(pmtg, "latent_dim", None)
            if latent_dim is None:
                raise ValueError("latent_space not provided and pmtg.latent_dim missing")
            low = -np.ones(latent_dim, dtype=np.float32)
            high = np.ones(latent_dim, dtype=np.float32)
            latent_single = gym.spaces.Box(low=low, high=high)
        else:
            latent_single = latent_space

        # batch the latent space to match env's vectorization
        self.action_space = gym.vector.utils.batch_space(latent_single, self.num_envs)

        # keep original env action space for internal mapping
        self.env_action_space = env.action_space

        # optional mask to mix PMTG output with direct latent (for partial joints etc.)
        if passthrough_mask is not None:
            self.passthrough_mask = passthrough_mask.to(self.device)
        else:
            self.passthrough_mask = None

        # time tracking per env
        self._t = torch.zeros(self.num_envs, device=self.device)
        # attempt to pull step_dt from IsaacLab env
        self._dt = getattr(env, "step_dt", 0.02)

    # IsaacLab vector envs expose num_envs
    @property
    def num_envs(self) -> int:
        return getattr(self.env, "num_envs", 1)

    def reset(self, **kwargs):
        obs_info = self.env.reset(**kwargs)
        # gymnasium reset returns (obs, info); older gym may return obs only
        if isinstance(obs_info, tuple) and len(obs_info) == 2:
            obs, info = obs_info
        else:
            obs, info = obs_info, {}
        self._t.zero_()
        if hasattr(self.pmtg, "reset"):
            self.pmtg.reset(None)
        return obs, info

    def step(self, action):
        # action is latent produced by the policy
        if isinstance(action, np.ndarray):
            latent = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        elif isinstance(action, torch.Tensor):
            latent = action.to(self.device)
        else:
            latent = torch.tensor(action, dtype=torch.float32, device=self.device)

        # compose state if useful
        state = None
        env_any = cast(Any, self.env)
        obs_buf = getattr(env_any, "obs_buf", None)
        if obs_buf is not None:
            state = {"obs": obs_buf}
        ep_len = getattr(env_any, "episode_length_buf", None)
        if ep_len is not None and isinstance(ep_len, torch.Tensor):
            steps = ep_len.to(self.device).to(torch.float32)
            # keep self._t authoritative; steps may reset lazily
            state = {**(state or {}), "steps": steps}

        env_action = self.pmtg(latent, self._t, state)

        if self.passthrough_mask is not None:
            # blend: a = mask*env_action + (1-mask)*latent_projected (if shapes match)
            if latent.shape[-1] == env_action.shape[-1]:
                env_action = self.passthrough_mask * env_action + (1 - self.passthrough_mask) * latent

        # convert to env-expected type
        if isinstance(self.env_action_space, gym.spaces.Box) and isinstance(env_action, torch.Tensor):
            env_action = env_action.clamp(
                min=torch.as_tensor(self.env_action_space.low, device=self.device, dtype=env_action.dtype),
                max=torch.as_tensor(self.env_action_space.high, device=self.device, dtype=env_action.dtype),
            )

        if isinstance(env_action, torch.Tensor):
            env_action_np = env_action.detach().cpu().numpy()
        else:
            env_action_np = np.asarray(env_action, dtype=np.float32)

        # step underlying env
        result = self.env.step(env_action_np)

        # gymnasium returns 5-tuple; be robust to 4-tuple
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            # fallback: classic gym (obs, reward, done, info)
            obs, reward, done, info = result  # type: ignore[misc]
            terminated, truncated = done, np.zeros_like(done)

        # advance time after env step
        self._t = self._t + self._dt

        # per-env resets: zero time where episode ended
        term_t = torch.as_tensor(terminated, device=self.device)
        trunc_t = torch.as_tensor(truncated, device=self.device)
        reset_mask = (term_t.bool() | trunc_t.bool()).view(-1)
        if reset_mask.any():
            self._t[reset_mask] = 0.0
            if hasattr(self.pmtg, "reset"):
                # if PMTG wants per-env reset, pass indices
                env_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1)
                self.pmtg.reset(env_ids)

        return obs, reward, terminated, truncated, info

    def env_method(self, method_name: str, *args, **kwargs):
        # helper to reach underlying env APIs from algorithms
        return getattr(self.env, method_name)(*args, **kwargs)
