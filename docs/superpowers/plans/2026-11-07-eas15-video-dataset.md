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

- [ ] Download VideoGPA's four training caption JSONs at commit `91356baa923e2f6f6991e3f3377fa1b60cc441da`.
- [ ] Download VideoREPA's `openvid_3w2.csv` and `openvid_6w4.csv` at commit `8d581dc301e18546a1808436c724f59d259d09f9`.
- [ ] Verify complete-file hashes against the design's audit. Require 3,147 VideoGPA rows, 32,000 smaller-selection rows and 64,000 larger-selection rows.
- [ ] Record source provenance and upstream usage terms in the recipe and dataset card.

## Task 2 — Build and verify the HF dataset

**CLI:** `python scripts/prepare_video_posttrain_dataset.py --output data/video_posttrain/generated`. Add `--offline` for verified cached sources and `--repo-id cvis-tmu/VideoGPA-VideoREPA --push-to-hub` for publication.

- [ ] Add fixture tests for exact text preservation, JSON/CSV field mapping, valid relative paths, invalid schemas and corrupted cached input.
- [ ] Add overlap tests proving source configurations preserve their rows, shared VideoREPA rows appear once in `combined`, and different source videos with identical captions remain distinct.
- [ ] Implement deterministic conversion with exactly `text` and `video_source` string columns. Export one Parquet per configuration and declare each configuration's single `train` split in the card.
- [ ] Verify full outputs have 3,147 / 32,000 / 64,000 / 67,801 rows for `videogpa` / `videorepa_32k` / `videorepa_64k` / `combined` respectively. Require 31,346 shared VideoREPA rows.
- [ ] Rebuild offline and compare row content and hashes. Record aggregate results and a small manual path/text audit.
- [ ] Run focused pytest and Ruff. Commit the implementation and documentation.

## Task 3 — Publish, reload and propose the PR

- [ ] Check the authenticated HF account can publish under `cvis-tmu`, and check the target repository for existing content before making changes.
- [ ] Upload all four Parquet configurations and the dataset card as one dataset commit. Set `combined` as default. Record the returned repository URL and commit SHA.
- [ ] Reload all configurations from that SHA with `datasets.load_dataset`, checking schema, count and complete row equality against the local release.
- [ ] Document `load_dataset("cvis-tmu/VideoGPA-VideoREPA", "combined", split="train")` plus the exact-source configuration names and path resolution examples.
- [ ] Inspect the final Git diff and ensure only code, source recipe, aggregate audit and documentation are staged. Push the EAS-15 branch and open a draft PR against `main` in `leihanchen/DiffusionOPSD` titled `feat(data): publish combined VideoGPA and VideoREPA dataset`.
- [ ] Link EAS-15 and the published dataset. Report exact counts, overlap policy, source hashes, HF revision, tests and any access or path-resolution limitations.

## Acceptance

The published dataset reloads at a pinned HF revision with exactly two string columns and the verified row counts above. The source configurations reproduce upstream training selections. The combined configuration contains their exact-row union. The repository contains reproducible preparation/publication code, and a draft PR is available for review.
