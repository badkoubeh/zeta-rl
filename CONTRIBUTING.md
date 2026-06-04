# Contributing — Zeta RL

Development guidelines and architecture decisions for this project.
Read this before contributing changes, suggestions, or additions.

---

## How to Contribute (humans and AI agents)

This is an open-source project. Contributions are welcome from human
developers and from AI coding agents alike — **both follow the same standards
described in this document.** An AI coding agent operating on this repository
acts as a contributor and is held to the guidelines here, no differently from
a human contributor. Tooling choices are private to each contributor and must
not leak into the project (see *Commits and pull requests* below).

### Workflow

1. Work on a branch — never commit directly to `main`.
2. Keep changes scoped, and keep the import chain intact (see
   *Module Dependency Rules*).
3. Add or update tests for every change (see *Testing*). The suite must pass
   and coverage must stay at or above the configured threshold.
4. If you change signal flow, the observation/action/adversary spaces, the
   dynamics interface, or the reward structure, update the README diagram
   (see *Diagram Maintenance*) — the pre-commit hook enforces this locally.
5. Never hardcode values that belong in `configs/`.

### Commits and pull requests

- Write clear, imperative commit messages describing *what* changed and *why*.
- **Keep the repository tool-agnostic.** Do not add AI-tool branding,
  "Generated with …" notes, or `Co-Authored-By` trailers naming an AI tool to
  commits, code, or docs. Commit under your own identity.
- Open a pull request against `main`; CI runs tests, coverage, and the diagram
  check.

### For AI coding agents specifically

- Treat this file as your contract: read it fully before editing anything.
- Do not introduce tool-specific files into the tracked repo. Keep any
  agent/editor configuration under git-ignored paths (e.g. `.claude/`).
- Refer to the project owner as "the maintainer" or "a contributor" in code
  and docs — never as "the user".
- Surface assumptions, trade-offs, and limitations honestly (see
  *Known Limitations*).

---

## Project Summary

Zeta RL is a framework for **physics-grounded RL simulation environments for
robust control under uncertainty** — dynamics derived from first principles and
agents evaluated across disturbances and model uncertainty. The framework is
domain-agnostic by design (see *Adding an environment* below); the **current
reference environment is 6-DOF rocket landing**, which exercises the full stack:

- Physics derived from first principles (not a tutorial copy)
- Adversarial training loop (agent vs learned disturbance adversary)
- Full robustness evaluation matrix across disturbance types
- PID vs SAC vs PPO head-to-head benchmark

The architecture decisions below describe the rocket-landing reference
environment. Additional environments (e.g. autonomous driving, LLM
post-training) are a roadmap item, not yet implemented.

---

**Write code at this level.** No tutorial-quality implementations. No hand-waving
on physics. Assume deep familiarity with control theory, RL, and production ML.

---

## Architecture Decisions (DO NOT change without discussion)

### Dynamics
- Abstract base class `RocketDynamics` with `step()` and `get_params()`
- `ModerateFidelityDynamics` is the current implementation (6-DOF rigid body)
- High fidelity upgrade path exists via config flag — do not break this
- `dynamics/` is self-contained. Only `envs/` may import from it

### Observation Space (17-dim)
`[x, y, z, vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, Tx, Ty, Tz, fuel_mass, fuel_remaining]`

### Action Space
Continuous 3D thrust vector `[Tx, Ty, Tz]` ∈ `[-1, 1]³`

### Algorithms
SAC + PPO — both trained, compared empirically. Do not remove either.

### Robustness Strategy
Adversarial training: agent vs learned adversary (disturbance
injector). Alternating gradient updates. Adversary action space:
`[wind_x, wind_y, wind_z, noise_magnitude, mass_offset]`

### Reward
Hybrid: dense shaping every step + sparse terminal bonus/penalty.
All coefficients in `configs/reward.yaml` — never hardcode reward weights.

### Curriculum
Automatic difficulty annealing over training progress (0.0 → 1.0).
Drop height, wind budget, adversary weight all anneal via config-driven scheduler.

---

## Adding an environment (roadmap)

Zeta RL ships one reference environment today (rocket landing). New environments
are welcome and should plug into the existing layering rather than fork it. The
extension surface a new environment builds against:

- **Dynamics** — implement the `RocketDynamics`-style contract in `dynamics/`
  (`step()` + `get_params()`). A domain-neutral `BaseDynamics` split is a planned
  refactor; until then, follow the existing abstract base.
- **Environment** — expose a standard Gymnasium `gym.Env` in `envs/` and register
  it with a versioned id (the rocket env registers `RocketLanding-v0`).
- **Reward / curriculum** — keep all coefficients in `configs/`; never hardcode.
- **Controllers** — `SACAgent` / `PPOAgent` are domain-agnostic and adapt to any
  observation/action space; only domain-specific baselines (like the rocket PID)
  need new code.

If you are planning a second environment, open an issue first so we can align on
the shared abstractions (env registry, domain config group) before the code lands.

---

## Diagram Maintenance

The closed-loop control diagram in `README.md` (under `## Architecture`) is the
authoritative visual specification of the system topology. **Update both the
diagram and any affected dimensions/units in the surrounding text whenever you
change:**

1. Action space — dimensions, semantics, or scaling
2. Observation space — dimensions, slot semantics, or normalisation bounds
3. Adversary action or observation space
4. Dynamics interface — `RocketDynamics.step()` signature, fidelity tiers
5. Reward decomposition — dense vs sparse vs terminal structure
6. Modules inserted into the loop — world model, recurrent policy, additional sensors

Trigger paths (the `pre-commit` hook at `scripts/check_diagram_sync.py`
enforces this locally; the `.github/workflows/diagram-check.yml` workflow
surfaces a soft warning on pull requests):

- `dynamics/**/*.py`
- `envs/rocket_landing_env.py`, `envs/__init__.py`
- `controllers/**/*.py`
- `adversary/**/*.py`
- `configs/{env,reward,adversary}.yaml`

When in doubt: would a reviewer reading **only the README diagram** get the
right mental model of what was just changed? If not, update the diagram.

---

## Coding Standards (enforce strictly)

```python
# Type hints on all functions
def step(self, state: State, action: Action, dt: float) -> State:

# NumPy-style docstrings with units
def compute_thrust(self, throttle: float) -> np.ndarray:
    """
    Compute thrust vector from normalised throttle command.

    Parameters
    ----------
    throttle : float
        Normalised throttle in [-1, 1].

    Returns
    -------
    np.ndarray
        Thrust vector [Tx, Ty, Tz] in Newtons.
    """
```

- No hardcoded values — everything in `configs/`
- No `print()` for logging — use Python `logging` module
- All random ops use seeded `np.random.Generator`, never `np.random.seed()`
- Units must be documented in docstrings (SI units throughout)
- Tests for every physics function in `tests/`

---

## Config System

Hydra-managed YAML configs. Entry point:

```bash
python experiments/train.py --config-name train
python experiments/train.py dynamics.fidelity=high        # override
python experiments/train.py agent=ppo                     # swap agent
```

Config files:
- `configs/train.yaml` — top-level, composes others
- `configs/env.yaml` — environment + dynamics params
- `configs/reward.yaml` — all reward weights
- `configs/adversary.yaml` — adversary hyperparams
- `configs/agent/sac.yaml` — SAC hyperparams
- `configs/agent/ppo.yaml` — PPO hyperparams

---

## Experiment Tracking

All runs logged to wandb. Required logged values:
- Every reward component separately (not just total reward)
- Curriculum progress (current difficulty level)
- Adversary loss alongside agent loss
- Episode metrics: landing success, touchdown velocity, fuel used
- Evaluation metrics: robustness matrix results

wandb project name: `zeta-rl`
Run naming convention: `{agent}_{fidelity}_{adversarial|nominal}_{seed}`
Example: `sac_moderate_adversarial_42`

---

## Module Dependency Rules

```
configs → dynamics → envs → controllers/adversary → experiments
```

**Never** import upward in this chain.
**Never** import from `experiments/` in any other module.
`experiments/` contains entrypoints only — no business logic.

---

## Testing

```bash
pytest tests/ -v                         # run all
pytest tests/test_physics.py -v         # physics correctness only
pytest tests/ --cov=dynamics --cov=envs  # coverage
```

Required test coverage:
- `dynamics/`: energy conservation, thrust bounds, mass depletion
- `envs/`: observation shape, action clipping, reward range
- `controllers/`: PID output bounds, agent load/save

---

## Reproduction

Single command to reproduce any result:

```bash
python experiments/train.py --config-name train seed=42
```

All results, checkpoints, and videos saved to `results/{run_name}/`.

---

## Robustness Evaluation

```bash
python experiments/evaluate_robustness.py checkpoint=results/sac_moderate_adversarial_42/
```

Outputs:
- `results/robustness_matrix.csv` — full disturbance sweep table
- `results/side_by_side.mp4` — naive vs robust agent video
- wandb table logged automatically

---

## Current Phase

> Update this section as phases complete.

- [x] Phase 1 — Foundation (Days 1–4) — Track A infrastructure + Track B physics core complete; `notebooks/physics_derivation.ipynb` deferred (maintainer-authored)
- [ ] Phase 2 — Controllers (Days 5–10)
- [ ] Phase 3 — Adversarial + Robustness (Days 11–18)
- [ ] Phase 4 — Polish (Days 19–21)

---

## Known Limitations (be honest)

Document in `README.md` as work progresses:
- Adversarial training may be unstable — fallback is domain randomisation
- Moderate fidelity omits aerodynamics and gimbal actuator dynamics
- Training is CPU/single-GPU; no distributed training
- No real hardware validation
