"""Integration tests for the SB3 agent wrappers (SAC + PPO).

These require the optional ``train`` extra (``pip install -e '.[train]'``);
they are skipped cleanly via :func:`pytest.importorskip` on machines without
torch / stable-baselines3 (e.g. Intel-mac dev boxes, lean CI). The wrappers
themselves stay importable without the extra (lazy SB3 imports), so the
``import``-level checks below run everywhere.
"""
from __future__ import annotations

import numpy as np
import pytest
from hydra import compose, initialize

from controllers.ppo_agent import PPOAgent
from controllers.sac_agent import SACAgent
from envs.rocket_landing_env import OBS_DIM, RocketLandingEnv

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture
def cfg():
    # Force a single CPU worker + tiny rollout so the smoke-train is fast.
    with initialize(config_path="../configs", version_base=None):
        c = compose(
            config_name="train",
            overrides=[
                "compute.n_envs=1",
                "compute.batch_size=8",
                "compute.buffer_size=200",
                "compute.n_steps=16",
                "agent.learning_starts=8",
            ],
        )
        return c


def test_agent_modules_import_without_sb3() -> None:
    """Wrappers must be importable even without the train extra installed."""
    assert SACAgent(cfg=None)._model is None  # noqa: SLF001
    assert PPOAgent(cfg=None)._model is None  # noqa: SLF001


def test_predict_before_learn_raises() -> None:
    """Calling predict() before learn()/load() is a clear error, not a crash."""
    with pytest.raises(RuntimeError):
        SACAgent(cfg=None).predict(np.zeros(OBS_DIM))
    with pytest.raises(RuntimeError):
        PPOAgent(cfg=None).predict(np.zeros(OBS_DIM))


@pytest.mark.parametrize("agent_cls", [SACAgent, PPOAgent])
def test_learn_predict_save_load_roundtrip(agent_cls, cfg, tmp_path) -> None:
    """Smoke: train a handful of steps, predict an in-bounds action, and
    round-trip through save/load."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")

    # Keep the run tiny and off wandb.
    cfg.total_steps = 64
    cfg.results_dir = str(tmp_path)

    env = RocketLandingEnv(cfg)
    agent = agent_cls(cfg)
    agent.learn(env, total_steps=int(cfg.total_steps))

    obs, _ = env.reset(seed=0)
    action = agent.predict(obs, deterministic=True)
    assert action.shape == (3,)
    assert env.action_space.contains(action.astype(env.action_space.dtype))

    ckpt = tmp_path / "ckpt"
    agent.save(str(ckpt))
    restored = agent_cls.load(str(ckpt))
    action2 = restored.predict(obs, deterministic=True)
    assert action2.shape == (3,)
    np.testing.assert_allclose(action, action2, rtol=1e-5, atol=1e-6)
