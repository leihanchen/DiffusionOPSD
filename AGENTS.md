# Repository Guidelines

## Project Structure & Module Organization

- `src/` is installed as `diffusionopsd`; it contains reward scorers, shared training utilities, `diffusers_patch/` pipeline adaptations, and `video/` geometry and Wan policy modules.
- `scripts/` contains training, evaluation, checkpoint setup, and Slurm entry points. `config/` holds Python experiment presets; `opd/` contains multi-teacher distillation baselines.
- `tests/video/` holds pytest tests. `data/` contains prompt splits and preparation notes; `assets/` holds documentation images.
- `containers/` provides the Apptainer definition. `docs/superpowers/` records design specifications and implementation plans.

## Build, Test, and Development Commands

Run commands from the repository root. Use Python 3.10+; the README recommends 3.10–3.11. Install CUDA-matched PyTorch before training dependencies.

- `pip install -e ".[dev]"` installs the editable package, pytest, and Ruff.
- `pip install -e ".[rewards,dev]"` adds standard reward dependencies; follow README instructions for additional reward packages and checkpoints.
- `python -m pytest tests/video -q` runs the video unit suite.
- `ruff check src config scripts opd tests` checks Python lint rules; restrict paths to changed files for focused checks.
- `python scripts/prepare_pickapic_prompts.py` prepares the Pick-a-Pic prompt splits.
- `SMOKE_TEST=1 NPROC=8 UPDATES=1 bash scripts/train_public.sh sd35 hpsv2` runs a short training check requiring eight CUDA GPUs and model/reward weights.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and uppercase constants. Ruff configures a 120-character line length. Follow nearby code, annotate tensor shapes and coordinate conventions, and keep experiment settings in `config/` rather than hardcoding them into reusable modules.

## Testing Guidelines

Name files `test_*.py` and functions `test_*`. Prefer small tensors and stub estimators for unit tests; check numerical behavior, shapes, and gradient flow when relevant. Run individual files during development, then the video suite for shared changes. No numeric coverage threshold is configured. Reward integration checks use `scripts/smoke_reward_gradient.py` and require the selected reward environment.

## Commit & Pull Request Guidelines

Follow recent history: `feat(video): ...`, `fix(env): ...`, or `docs: ...`, using concise imperative summaries. Keep commits focused. PRs should explain behavior changes, link relevant issues/specifications, and report validation commands, outcomes, and GPU limitations. For training changes, include the backbone, reward, configuration, and comparable metrics or sample outputs.

## Configuration & Artifacts

Use `REWARD_CKPT_PATH`, `VIDEO_REWARD_CKPT_PATH`, and `HF_HOME` for local weights and caches. Keep credentials, checkpoints, generated outputs, and logs out of commits; respect `.gitignore`.
