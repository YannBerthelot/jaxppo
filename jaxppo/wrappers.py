"""Wrappers for environment"""

from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import State
from flax import struct
from gymnax.environments import environment, spaces


class GymnaxWrapper:
    """Base class for Gymnax wrappers."""

    def __init__(self, env: environment.Environment):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class BraxWrapper:
    """Base class for Gymnax wrappers."""

    def __init__(self, env: environment.Environment):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def observation_space(self, params) -> spaces.Box:
        """Get the observation space from a gymnax env given its params"""
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        """Reset the environment and flatten the observation"""
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        """Step the environment and flatten the observation"""
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    """Logging buffer"""

    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        """Reset the environment and log the state of the env"""
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)  # type: ignore[call-arg]
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        """Step the environment and log the env state, episode return, episode length and timestep"""
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(  # type: ignore[call-arg]
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


class ClipAction(GymnaxWrapper):
    """Continus action clipping wrapper"""

    def __init__(self, env, low=-1.0, high=1.0):
        """Set the high and low bounds"""
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        """Step the environment while clipping the action first"""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, action, params)


class ClipActionBrax(BraxWrapper):
    """Continus action clipping wrapper"""

    def __init__(self, env, low=-1.0, high=1.0):
        """Set the high and low bounds"""
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, state, action):
        """Step the environment while clipping the action first"""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(state, action)


class TransformObservation(GymnaxWrapper):
    """Observation modifying wrapper"""

    def __init__(self, env, transform_obs):
        """Set the observation transformation"""
        super().__init__(env)
        self.transform_obs = transform_obs

    def reset(self, key, params=None):
        """Reset the env and return the transformed obs"""
        obs, state = self._env.reset(key, params)
        return self.transform_obs(obs), state

    def step(self, key, state, action, params=None):
        """Step the env and return the transformed obs"""
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self.transform_obs(obs), state, reward, done, info


class TransformReward(GymnaxWrapper):
    """Reward modifying wrapper"""

    def __init__(self, env, transform_reward):
        super().__init__(env)
        self.transform_reward = transform_reward

    def step(self, key, state, action, params=None):
        """Step the env and return the transformed reward"""
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return obs, state, self.transform_reward(reward), done, info


class VecEnv(GymnaxWrapper):
    """Vectorized an environment by vectorizing step and reset"""

    def __init__(self, env):
        """Override reset and step"""
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


@struct.dataclass
class NormalizeVecObsEnvState:
    """Carry variables necessary for online normalization"""

    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: State


@struct.dataclass
class NormalizeVecObsEnvStateBrax:
    """Carry variables necessary for online normalization"""

    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    pipeline_state: Optional[State]
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)


class NormalizeVecObservation(GymnaxWrapper):
    """Wrapper for online normalization of observations"""

    def reset(self, key, params=None):
        """Reset the environment and return the normalized obs"""
        obs, state = self._env.reset(key, params)
        state = NormalizeVecObsEnvState(
            mean=jnp.zeros_like(obs),
            var=jnp.ones_like(obs),
            count=1e-4,
            env_state=state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=state.env_state,
        )

        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

    def step(self, key, state, action, params=None):
        """Step the environment and return the normalized obs"""
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.var + 1e-8),
            state,
            reward,
            done,
            info,
        )


class NormalizeVecObservationBrax(BraxWrapper):
    """Wrapper for online normalization of observations"""

    def reset(self, key):
        """Reset the environment and return the normalized obs"""
        env_state = self._env.reset(key)
        obs = env_state.obs

        wrapped_state = NormalizeVecObsEnvStateBrax(
            mean=jnp.zeros_like(obs),
            var=jnp.ones_like(obs),
            count=1e-4,
            reward=env_state.reward,
            done=env_state.done,
            metrics=env_state.metrics,
            pipeline_state=env_state.pipeline_state,
            info=env_state.info,
            obs=env_state.obs,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - wrapped_state.mean
        tot_count = wrapped_state.count + batch_count

        new_mean = wrapped_state.mean + delta * batch_count / tot_count
        m_a = wrapped_state.var * wrapped_state.count
        m_b = batch_var * batch_count
        M2 = (
            m_a
            + m_b
            + jnp.square(delta) * wrapped_state.count * batch_count / tot_count
        )
        new_var = M2 / tot_count
        new_count = tot_count
        new_wrapped_state = NormalizeVecObsEnvStateBrax(
            mean=new_mean,
            var=new_var,
            count=new_count,
            reward=env_state.reward,
            done=env_state.done,
            metrics=env_state.metrics,
            pipeline_state=env_state.pipeline_state,
            info=env_state.info,
            obs=(env_state.obs - new_mean) / jnp.sqrt(new_var + 1e-8),
        )

        return new_wrapped_state

    def step(self, wrapped_state, action):
        """Step the environment and return the normalized obs"""
        unwrapped_state = State(
            wrapped_state.pipeline_state,
            wrapped_state.obs,
            wrapped_state.reward,
            wrapped_state.done,
            wrapped_state.metrics,
            wrapped_state.info,
        )
        env_state = self._env.step(unwrapped_state, action)
        obs = env_state.obs
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - wrapped_state.mean
        tot_count = wrapped_state.count + batch_count

        new_mean = wrapped_state.mean + delta * batch_count / tot_count
        m_a = wrapped_state.var * wrapped_state.count
        m_b = batch_var * batch_count
        M2 = (
            m_a
            + m_b
            + jnp.square(delta) * wrapped_state.count * batch_count / tot_count
        )
        new_var = M2 / tot_count
        new_count = tot_count

        new_wrapped_state = NormalizeVecObsEnvStateBrax(
            mean=new_mean,
            var=new_var,
            count=new_count,
            reward=env_state.reward,
            done=env_state.done,
            metrics=env_state.metrics,
            pipeline_state=env_state.pipeline_state,
            info=env_state.info,
            obs=(env_state.obs - wrapped_state.mean)
            / jnp.sqrt(wrapped_state.var + 1e-8),
        )
        return new_wrapped_state


@struct.dataclass
class NormalizeVecRewEnvState:
    """Carry variables necessary for online normalization"""

    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: State


@struct.dataclass
class NormalizeVecRewEnvStateBrax:
    """Carry variables necessary for online normalization"""

    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    pipeline_state: Optional[State]
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)


class NormalizeVecReward(GymnaxWrapper):
    """Wrapper for online normalization of rewards"""

    def __init__(self, env, gamma):
        """Set gamma"""
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        """Reset the environment and return the normalized reward"""
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        """Step the environment and return the normalized reward"""
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info


class NormalizeVecRewardBrax(BraxWrapper):
    """Wrapper for online normalization of rewards"""

    def __init__(self, env, gamma):
        """Set gamma"""
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key):
        """Reset the environment and return the normalized reward"""
        env_state = self._env.reset(key)
        batch_count = env_state.obs.shape[0]

        state = NormalizeVecRewEnvStateBrax(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            reward=env_state.reward,
            done=env_state.done,
            metrics=env_state.metrics,
            pipeline_state=env_state.pipeline_state,
            info=env_state.info,
            obs=env_state.obs,
        )
        return state

    def step(self, wrapped_state, action):
        """Step the environment and return the normalized reward"""
        unwrapped_state = State(
            wrapped_state.pipeline_state,
            wrapped_state.obs,
            wrapped_state.reward,
            wrapped_state.done,
            wrapped_state.metrics,
            wrapped_state.info,
        )
        env_state = self._env.step(unwrapped_state, action)
        obs, reward, done = (
            env_state.obs,
            env_state.reward,
            env_state.done,
        )
        return_val = wrapped_state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - wrapped_state.mean
        tot_count = wrapped_state.count + batch_count

        new_mean = wrapped_state.mean + delta * batch_count / tot_count
        m_a = wrapped_state.var * wrapped_state.count
        m_b = batch_var * batch_count
        M2 = (
            m_a
            + m_b
            + jnp.square(delta) * wrapped_state.count * batch_count / tot_count
        )
        new_var = M2 / tot_count
        new_count = tot_count

        new_wrapped_state = NormalizeVecRewEnvStateBrax(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            reward=reward / jnp.sqrt(new_var + 1e-8),
            done=env_state.done,
            metrics=env_state.metrics,
            pipeline_state=env_state.pipeline_state,
            info=env_state.info,
            obs=env_state.obs,
        )
        return new_wrapped_state


def get_wrappers(mode: str = "gymnax"):
    if mode == "gymnax":
        return ClipAction, NormalizeVecObservation, NormalizeVecReward
    return ClipActionBrax, NormalizeVecObservationBrax, NormalizeVecRewardBrax
