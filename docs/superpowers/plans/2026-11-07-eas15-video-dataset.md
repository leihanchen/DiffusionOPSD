# EAS-15 combined Hugging Face dataset implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Publish VideoGPA's 3,147 training captions and both exact VideoREPA 32K/64K selections as `cvis-tmu/VideoGPA-VideoREPA`, then propose a draft PR.

**Architecture:** A standalone Python CLI reads pinned JSON/CSV sources, builds two-column HF datasets, exports Parquet, and publishes one dataset repository with exact-source configurations and a default combined configuration.

**Tech Stack:** Python 3.10+, existing `datasets` and `huggingface_hub`, standard-library JSON/CSV/hashlib, pytest and Ruff.

**Spec:** `docs/superpowers/specs/2026-11-07-eas15-video-dataset-design.md`.

## Global constraints

- Columns are `text` and `video_source`, both strings.
- Preserve original text and exact membership of each source selection.
- Combined configuration removes exact duplicate rows by `(text, video_source)`.
- Relative paths use `DL3DV/` and `OpenVid/` prefixes. Preserve VideoGPA frame-directory references and document their meaning.
- Pin source revisions and hashes. Keep credentials, source caches and generated data outside Git.
- Use the existing EAS-15 worktree and branch. Publish to the `cvis-tmu` namespace as authorized by the user.

## Files

- Create `scripts/prepare_video_posttrain_dataset.py` for collection, conversion, validation and optional upload.
- Create `data/video_posttrain_sources.json` for pinned source URLs, revisions and SHA-256 hashes.
- Create `data/video_posttrain/README.md` for HF dataset-card content and preparation/loading instructions.
- Create `data/video_posttrain_audit.json` for measured counts and content hashes.
- Create `tests/video/test_posttrain_dataset.py` for conversion and overlap regression coverage.
- Modify `.gitignore` to exclude source caches and generated Parquet while retaining documentation.

## Task 1 — Collect the exact sources

- [x] Download VideoGPA's four training caption JSONs at commit `91356baa923e2f6f6991e3f3377fa1b60cc441da`.
- [x] Download VideoREPA's `openvid_3w2.csv` and `openvid_6w4.csv` at commit `8d581dc301e18546a1808436c724f59d259d09f9`.
- [x] Verify complete-file hashes against the design's audit. Require 3,147 VideoGPA rows, 32,000 smaller-selection rows and 64,000 larger-selection rows.
- [x] Record source provenance and upstream usage terms in the recipe and dataset card.

## Task 2 — Build and verify the HF dataset

**CLI:** `python scripts/prepare_video_posttrain_dataset.py --output data/video_posttrain/generated`. Add `--offline` for verified cached sources and `--repo-id cvis-tmu/VideoGPA-VideoREPA --push-to-hub` for publication.

- [x] Add fixture tests for exact text preservation, JSON/CSV field mapping, valid relative paths, invalid schemas and corrupted cached input.
- [x] Add overlap tests proving source configurations preserve their rows, shared VideoREPA rows appear once in `combined`, and different source videos with identical captions remain distinct.
- [x] Implement deterministic conversion with exactly `text` and `video_source` string columns. Export one Parquet per configuration and declare each configuration's single `train` split in the card.
- [x] Verify full outputs have 3,147 / 32,000 / 64,000 / 67,801 rows for `videogpa` / `videorepa_32k` / `videorepa_64k` / `combined` respectively. Require 31,346 shared VideoREPA rows.
- [x] Rebuild offline and compare row content and hashes. Record aggregate results and a small manual path/text audit.
- [x] Run focused pytest and Ruff. Commit the implementation and documentation.

## Task 3 — Publish, reload and propose the PR

- [x] Check the authenticated HF account can publish under `cvis-tmu`, and check the target repository for existing content before making changes.
- [x] Upload all four Parquet configurations and the dataset card as one dataset commit. Set `combined` as default. Record the returned repository URL and commit SHA.
- [x] Reload all configurations from that SHA with `datasets.load_dataset`, checking schema, count and complete row equality against the local release.
- [x] Document `load_dataset("cvis-tmu/VideoGPA-VideoREPA", "combined", split="train")` plus the exact-source configuration names and path resolution examples.
- [x] Inspect the final Git diff and ensure only code, source recipe, aggregate audit and documentation are staged. Push the EAS-15 branch and open a draft PR against `main` in `leihanchen/DiffusionOPSD` titled `feat(data): publish combined VideoGPA and VideoREPA dataset`.
- [x] Link EAS-15 and the published dataset. Report exact counts, overlap policy, source hashes, HF revision, tests and any access or path-resolution limitations.

## Acceptance

The published dataset reloads at a pinned HF revision with exactly two string columns and the verified row counts above. The source configurations reproduce upstream training selections. The combined configuration contains their exact-row union. The repository contains reproducible preparation/publication code, and a draft PR is available for review.

## Delivery

Published [cvis-tmu/VideoGPA-VideoREPA](https://huggingface.co/datasets/cvis-tmu/VideoGPA-VideoREPA) at `95f93059b25b2fbe4759d8e6be3aa36b30e85b62`. All configuration and default-view row comparisons passed. Public anonymous access was verified. Full video suite passed 242 tests with two dependency warnings, and focused Ruff passed.

Draft [PR 5](https://github.com/leihanchen/DiffusionOPSD/pull/5) targets `main`. The development worktree is preserved for review.

## VideoGPA original test subset update

The user requested publishing the original 1K evaluation subset as test. The pinned `captions_1K.json` source supplies exactly 1,000 rows in both `combined/test` and `videogpa/test`. The columns remain `text` and `video_source`. Training Parquet byte hashes match the initial release. Normalized training/test captions and DL3DV scene IDs are checked for overlap. Publication requires an explicit current parent revision, preserving training content and rejecting concurrent updates.

Published HF revision `a6909e889f5bb134778aaf6910f44cf18c7b8457` was reloaded, with independent anonymous source-to-HF equality for every test record. The dataset card and preparation script reproduce both splits.

## Combined metadata update

The user requested `sample_id`, `source_dataset`, `source_subsets`, `source_group_id`, and `media_type` in the combined data. Both combined splits now have seven columns. Exact-source configurations retain their two-column schema, with a dedicated combined test Parquet for schema consistency. IDs hash the compact UTF-8 JSON array `[text, video_source]`. Membership lists retain both VideoREPA subsets where applicable. Group IDs identify original DL3DV scenes or OpenVid source videos. The dataset card specifies all conventions.

Published revision `b5c9967cb7801ceec0b3581839d065131eefc5be` reloaded successfully. Independent checks cover all 67,801 training and 1,000 test rows, unique IDs, exact memberships and source groups. Offline rebuilding reproduces all output hashes. Original caption/path content is preserved.
