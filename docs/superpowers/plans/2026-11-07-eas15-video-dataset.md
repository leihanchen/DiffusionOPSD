# EAS-15 video dataset implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Prepare audited VideoGPA/VideoREPA datasets, wire them into Wan/Cosmos, and propose a draft PR linked to EAS-15.

**Architecture:** A pinned source recipe drives a CPU-only preparation module and CLI. Caption and provenance records are grouped before splitting, then written as trainer-compatible text files with verification metadata. Opt-in training presets consume the verified export and expose a model-free validation path.

**Tech Stack:** Python 3.10+, standard-library JSON/CSV/hashlib/urllib, existing ml_collections configs, pytest and Ruff.

**Spec:** `docs/superpowers/specs/2026-11-07-eas15-video-dataset-design.md`.

**Status:** Proposed plan for user review. Product implementation begins after design review. Review choices include the 32K source default, full accepted union, filtering thresholds, 90/5/5 component split, and optional Wan batch alignment.

## Global constraints

- Python 3.10+, four-space indentation, 120-character Ruff line length.
- Pin upstream revisions and input byte hashes. Use no new required dependencies.
- Keep source downloads, generated datasets, checkpoints and logs outside Git.
- Dataset preparation and validation require no video downloads, model weights or CUDA.
- Follow the spec's exact filtering rules, split seed 42, grouping semantics and output contract.
- Preserve legacy configuration and resume behavior while adding explicit dataset presets.
- Produce and validate a real full dataset release before claiming EAS-15 acceptance.

## Review focus

- Distinct captions from one original scene/video must remain together, including components joined by duplicated text.
- VideoREPA 32K/64K overlap must be measured before selecting one manifest per recipe.
- Reserved evaluation text can exclude an entire connected source group.
- Downloads, forced rebuilds and retries must leave an existing verified release usable.
- Wan's four-prompt scheduler and saved state must agree with the effective dataset count.

## File responsibilities

- Create `data/video_posttrain_recipe.json` for versioned sources, byte hashes, filters and split rules.
- Create `data/video_posttrain/README.md` for preparation, usage terms, audit results and trainer commands.
- Create `src/video/prompt_dataset.py` for source parsing, normalization, grouping, release writing and verification.
- Create `scripts/prepare_video_posttrain_prompts.py` as the preparation/audit CLI.
- Create `tests/video/test_prompt_dataset.py` for small fixture-based dataset regression tests.
- Create `config/wan22_video_posttrain.py` and `config/cosmos_video_posttrain.py` as opt-in presets.
- Modify `scripts/train_opsd_video_wan.py`, `scripts/train_opsd_video_wan_epoch.py`, `scripts/train_opsd_video_cosmos.py` and their relevant tests for shared file validation, model-free checks and dataset metadata.
- Modify `config/wan22_ti2v_epoch.py` only as needed to declare existing dataset expectations explicitly.
- Modify `.gitignore` for generated exports and raw source cache, allowing the dataset README.
- Create a compact tracked `data/video_posttrain_audit.json` with actual aggregate counts/hashes, with generated row-level reports kept ignored.

## Task 1 — Audit and pin complete upstream inputs

**Interface:** `data/video_posttrain_recipe.json` declares `format_version`, `dataset_version`, `sources`, `default_variant`, `filters`, `split` and `reserved_inputs`. Each source file has repository, revision, relative path, URL, SHA-256 and measured row count. Local reserved files also carry byte hashes.

- [ ] Read the spec and issue. Confirm branch/worktree state and run the existing baseline video suite in the validated environment. Capture environment limitations before code changes.
- [ ] Fetch all five VideoGPA JSONs and both VideoREPA CSVs with bounded network timeouts to ignored cache paths. Verify repository revisions through GitHub. Record byte hashes from full successful downloads.
- [ ] Parse full inputs, measure exact rows, unique captions, source groups and overlap between the VideoREPA manifests. Verify columns/types and reject unknown schema. Report any discrepancy with prior stated counts.
- [ ] Inspect 20 seeded captions per source plus repeated/long captions. Define a fixed, documented camera/object-motion keyword vocabulary for descriptive tags. Treat matches as heuristics, permit `unknown`, and avoid inferring ground-truth motion.
- [ ] Write the pinned recipe using the spec's default 32K variant. Record a 64K variant selecting the alternate CSV. Include provenance and access/usage-term references.
- [ ] Commit source recipe and initial dataset documentation as `docs(data): pin VideoGPA and VideoREPA source recipes`.

## Task 2 — Parse, normalize, filter and split with provenance

**Interfaces:** In `src/video/prompt_dataset.py`, `load_source_records(recipe: dict, cache_dir: Path) -> list[dict]` produces source records; `build_release(records: list[dict], recipe: dict, reserved: dict[str, list[str]]) -> dict` produces split records and aggregate exclusions. `normalize_prompt(text: str) -> str` and `source_group(source: str, identifier: str) -> str` own canonicalization and grouping.

- [ ] Add fixture tests for NFKC/whitespace, non-string/empty inputs, 4/5/250/251-token boundaries, control characters and repetitive 4-gram boundaries. Assert original text remains in provenance and every rejected source row has a reason.
- [ ] Add identifier tests with embedded underscores, multiple segments of the same video, invalid suffixes and multiple frame folders in a DL3DV scene. Assert stable original-video/scene group IDs.
- [ ] Add splitting tests where identical text bridges two groups and both sources, a reserved caption touches one member, and input rows are reversed. Assert zero train/validation/test group or normalized-text overlap, complete alias accounting, reserved-component removal and byte-stable output order.
- [ ] Add tests for 32K/64K overlapping inputs, empty accepted sources, tiny connected components and unknown motion. The chosen variant must consume one VideoREPA selection. Fail actionably if a required training/evaluation split cannot be formed.
- [ ] Run the new tests and confirm they expose missing implementation. Implement the exact spec rules with deterministic connected components and source-stratified component assignment. Sort duplicate aliases and records independently of source input order.
- [ ] Run `python -m pytest tests/video/test_prompt_dataset.py -q`, then focused Ruff. Commit as `feat(data): build provenance-aware video prompt splits`.

## Task 3 — Materialize and verify complete releases

**Interfaces:** `materialize(recipe_path: Path, output_dir: Path, cache_dir: Path, *, offline: bool = False, force: bool = False, variant: str = "32k") -> dict` returns the verified release report. `verify_release(output_dir: Path) -> dict` checks all exported hashes, counts, metadata alignment and split isolation.

- [ ] Add CLI options `--recipe`, `--output`, `--cache-dir`, `--variant {32k,64k}`, `--offline`, `--force` and `--verify-only`. Default recipe/output paths resolve from the repository root. Errors report the offending source/file and recovery command.
- [ ] Add tests for corrupted cached inputs, source download failure, differing existing outputs, failed forced rebuild, output tampering and stale staging directories. Verify successful repeated builds produce identical output bytes and a failed replacement preserves a verified existing release. Concurrent attempts must be serialized or rejected by an exclusive output lock with actionable stale-lock recovery.
- [ ] Implement verified input caching, staged writes and atomic publication/rollback. Include `manifest.jsonl`, line-aligned split exports, per-source exports, `report.json` and output hashes. Keep hashes deterministic and exclude timestamps and absolute paths from the release identity.
- [ ] Add `.gitignore` entries for generated exports and source cache with an explicit exception for `data/video_posttrain/README.md`.
- [ ] Materialize the full default release. Run `--verify-only`, rebuild to a second ignored directory offline and compare file hashes. Audit 20 retained/rejected examples per source and tag where available. Save actual counts, exclusions, source proportions, cache/output sizes, elapsed preparation time and hashes in the tracked aggregate audit.
- [ ] Commit as `feat(data): materialize verified video post-training datasets`.

## Task 4 — Wire normal Wan/Cosmos entry points and model-free checks

**Interfaces:** `load_training_prompts(train_path: Path, eval_path: Path, *, expected_count: int | None = None, release_dir: Path | None = None) -> tuple[list[str], list[str], dict]` validates files and returns dataset identity. `wan_training_selection(prompts: list[str]) -> tuple[list[str], list[str]]` returns the deterministic multiple-of-four selection and omitted tail, also exported during preparation as `wan_train.txt`. New presets use the same effective file for comparable Wan/Cosmos runs.

- [ ] Add tests showing legacy file-only configs load normally, new presets validate release metadata, stale metadata fails, and counts 0/1/3/4/5/7/8 have the declared Wan behavior. Assert all accepted data remains in the full export and omitted rows are reported.
- [ ] Add entry-point tests invoking `--validate-dataset-only` for simple Wan, epoch Wan and Cosmos. Stub model constructors/CUDA/distributed setup to raise if reached. Assert config paths and real prepared fixture files are read, effective counts/hashes are printed, and a corrupt or overlapping release fails before model loading. Support this flag before heavy optional imports if needed.
- [ ] Add the presets. `wan22_video_posttrain.py` derives placement/LoRA settings from the epoch preset and sets expected prompt count, one-pass updates, save and evaluation intervals explicitly. `cosmos_video_posttrain.py` derives its model/reward settings from `cosmos_nano.py`. Both train on `wan_train.txt` and evaluate on `validation.txt`.
- [ ] Replace hardcoded 200/50/evaluation checks in the epoch trainer with config-derived dataset and update validations. Preserve four logical prompts and four rollouts per prompt. Validate one-pass order length and coverage on resume, retaining prompt/config hash protections. Add regression tests for legacy 200-prompt/50-update behavior and a larger prepared dataset. Reject mismatched resume datasets.
- [ ] Use the shared prompt validation path in normal training and model-free mode. Persist dataset/recipe identity alongside existing run/checkpoint metadata for each trainer. Test saved metadata with stubbed training infrastructure.
- [ ] Run all affected config, scheduler, loader and resume tests. Commit as `feat(video): load prepared datasets in Wan and Cosmos`.

## Task 5 — Acceptance audit and draft PR

- [ ] Run full-release verification and model-free entry-point checks for all three trainers with the materialized dataset. Record exact commands and resulting counts/hashes in the README and audit.
- [ ] Run `python -m pytest tests/video -q` in the validated runtime and focused `ruff check` on changed Python files. Investigate failures before delivery. Report unavailable environment checks explicitly.
- [ ] Review the diff against every EAS-15 acceptance item. Check ignored files remain ignored and no generated caption corpus, video, cache, model weight or secret is staged. Verify current worktree/branch/base and repository remotes.
- [ ] Commit final documentation, push `codex/eas-15-video-training-dataset` to `leihanchen/DiffusionOPSD`, and create a draft PR targeting `main` titled `feat(data): prepare VideoGPA and VideoREPA prompts for video OPSD`. Link EAS-15 in the description.
- [ ] Include pinned references, exact observed counts and exclusions, final source mixture, split/grouping rules, output hashes, reproduction and loader commands, test evidence and remaining limitations. Verify the PR state and clean tracked worktree. Keep issue closure for verified acceptance and user review.

## Acceptance mapping

| EAS-15 requirement | Evidence |
| --- | --- |
| Reproducible preparation and pinned provenance | Tasks 1–3, offline second-build hash comparison |
| Counts, mixture, exclusions and split sizes | Full-source audit and tracked aggregate report |
| Scene/video and exact-text split isolation | Connected-component fixtures and full-release verification |
| Wan and Cosmos loading | Task 4 normal shared loading path and Task 5 entry-point checks |
| Artifact distinctions and usage terms | Dataset README, recipe and row-level source provenance |
| New worktree and proposed PR | Created branch/worktree and verified draft PR |
