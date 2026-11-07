# EAS-15 video training dataset design

Proposed design for review, November 7, 2026. Linked issue [EAS-15](https://linear.app/cvis-tmu/issue/EAS-15/create-new-training-dataset-with-videogpa-and-videorepa).

## Intent

Prepare reproducible VideoGPA and VideoREPA caption datasets for online Wan/Cosmos OPSD rollouts using the GeoFlow reward. Preserve original source identifiers so later experiments can retrieve corresponding real clips. Deliver the preparation code, a materialized dataset with audited counts, and a draft implementation PR.

## Current evidence

- Worktree `/lustre06/project/6062393/leihan/DiffusionOPSD-eas15-dataset`, branch `codex/eas-15-video-training-dataset`, based on committed main `389fb8904b1f51ddcc785bf8c5726c4e22adcdec`.
- VideoGPA reference `Hongyang-Du/VideoGPA@91356baa923e2f6f6991e3f3377fa1b60cc441da`. JSON maps scene/frame paths to captions. Complete downloads confirm 3,147 training-source entries and 1,000 evaluation-source entries, each with a distinct scene ID. No scene or normalized-text overlap was found between these source splits.
- VideoREPA reference `aHapBean/VideoREPA@8d581dc301e18546a1808436c724f59d259d09f9`. Complete downloads confirm 32,000 and 64,000 rows. Whitespace-normalized, case-folded caption counts are 31,973 and 63,929 respectively. Columns are `video,caption,seconds`. These are raw audit counts before the proposed NFKC normalization, quality filters and scene grouping.
- VideoREPA's 32K/64K figures describe two selections from OpenVid. Audit their overlap and choose one, avoiding double-counting their union by default.
- Both sources contain descriptive captions with static scenes and repetitive language. Automatic rewrite would change the data and require a separately versioned generation recipe.
- Wan uses `prompt_fn_kwargs.path` and `eval_prompts`. Cosmos uses `train_prompts` and `eval_prompts`. The Wan epoch entry point also hardcodes 200 prompts, 50 updates, and evaluation every 25 updates. Its scheduler processes four logical prompts per update.

## Approaches and proposed choice

1. Preserve all quality-accepted captions from a chosen VideoREPA subset and VideoGPA, with deterministic source-aware splitting and separate source exports. This gives auditable coverage and leaves experimental downsampling explicit. Recommended.
2. Select a fixed 5K balanced training mixture. This controls cost and source balance, but requires choosing quotas before measuring caption quality and eligible group sizes.

The proposed first version uses option 1 with VideoREPA's 32K manifest. Export the 64K source as an optional recipe variant after auditing overlap. Counts and source ratios are measured outputs. Motion/camera keyword tags are descriptive heuristics and remain visible in metadata.

## Dataset contract

- Fetch five VideoGPA JSON files and both VideoREPA CSVs into an ignored source cache. Pin revisions and SHA-256 digests in a tracked JSON recipe after the initial audit. Support rebuilding offline from verified cached inputs.
- Normalize captions using Unicode NFKC and whitespace collapse. Preserve the original string. Reject empty/non-string captions, control characters other than whitespace, captions with fewer than 5 or more than 250 whitespace-delimited tokens, and captions with a repeated 4-token sequence occupying at least 30% of their tokens. Define this last fraction as four times the maximum non-overlapping occurrence count divided by token count, requiring at least three occurrences. Record every rejection reason.
- Deduplication key is normalized case-folded text. Keep provenance for all duplicate aliases. VideoGPA group is its DL3DV scene hash. OpenVid group is the original video ID obtained by validating and removing the final `_segment_starttoend.mp4` suffix, preserving underscores inside the original ID. Reject unparseable identifiers.
- Form connected groups using both source scene/video identity and exact normalized text. A duplicate joining two source groups joins their whole split component. Exclude entire components touching reserved evaluation content.
- Reserve VideoGPA 1K scenes and existing `data/video_motion/test.txt` normalized prompts before training selection. Existing benchmark exclusions also include `data/drawbench/test.txt`. Exclude matching components from the training pool. Report overlap in exclusions.
- Within each source stratum, order components by SHA-256 of `seed:component_id`, seed 42, and allocate 90%/5%/5% of components to train/validation/test using floor for the first two and the remainder for test. A component containing both sources uses the `mixed` stratum. Report actual row proportions because groups vary in size.
- Export `train.txt`, `validation.txt`, `test.txt`, `videogpa_eval.txt`, per-source split files, `manifest.jsonl`, and `report.json` under ignored `data/video_posttrain/`. Each text line maps to one manifest record containing split, line index, prompt, source aliases, scene/video groups, heuristic tags, source revision and source row/key.
- Report raw rows, distinct captions, parsed groups, rejection counts, reserved overlaps, duplicate aliases, split counts by source/tag, output byte hashes, recipe hash, and the final training source ratio. Tag unknown motion explicitly.
- Record VideoGPA evaluation provenance separately. Components colliding with legacy benchmark text are excluded from the optional VideoGPA evaluation export too. Document any remaining overlap between evaluation collections rather than combining them silently.
- Publish only the preparation code, pinned recipe, aggregate audit report and documentation in Git. Large source files, generated captions/manifests and videos use ignored paths. No external upload is part of this change.
- A verified existing output is reusable. A differing output requires `--force`. Stage a complete output directory and publish it only after validation, with rollback on replacement failure. A failed download or validation must leave an existing valid release usable.

## Integration

Add opt-in Wan epoch and Cosmos presets targeting the new training split and validation split. Keep test data for final evaluation. Presets verify materialized files and obtain their dataset metadata from a shared lightweight helper.

For Wan, generalize the fixed dataset-size check to a declared expected count and four-prompt scheduling contract. The default dataset export keeps every accepted training row. If the count is not divisible by four, the Wan preset selects a deterministic prefix of `4 * floor(n / 4)` prompts into a separate Wan training file, reporting the 0–3 excluded rows and their identifiers. Cosmos can use that same file for matched comparisons. Do not duplicate prompts to fill a batch. Require at least four selected training prompts. One-pass update count is selected count divided by four. Save/evaluation frequencies are explicit positive configuration values. Retain existing resume hash checks and validate schedule completeness when loading state.

Add a dataset-only validation flag to the Wan epoch, simple Wan and Cosmos training entry points. It must load the config, read and validate the actual text files, and return before CUDA checks, distributed setup, checkpoint access, model loading or reward construction. Reuse this path during normal training. Validate existing legacy datasets without requiring a new manifest. For prepared datasets require recipe/manifest/output hashes and disjointness checks.

Record dataset revision, recipe hash, effective training file hash/count and validation hash/count in saved run metadata. Cosmos currently already saves text hashes. Wan epoch already saves prompt/config hashes. Extend their existing records without weakening resume checks. Record a dataset identity for the simple Wan run too.

## Validation and delivery

Test normalization, rejection boundaries, identifier parsing, cross-source duplicates joining components, reserved overlaps, reordered source input, source-subset overlap, tampered cache/output, interrupted rebuild, split isolation and small/uneven datasets with tiny local fixtures. Exercise the real dataset-only trainer paths using stubs where heavy imports require them. Materialize the full release twice and compare output hashes. Inspect a seeded sample from each source and motion tag, including rejected captions, and report quality limitations.

Run focused lint, dataset tests and the full video test suite in the existing validated runtime if available. No full training is needed for data acceptance. A model-free loader smoke verifies data wiring, with any GPU validation explicitly distinguished.

Open a draft PR against originating `main` in `leihanchen/DiffusionOPSD`, titled `feat(data): prepare VideoGPA and VideoREPA prompts for video OPSD`, link EAS-15, and include exact counts, exclusions, source mixture, dataset hashes, reproducibility commands, tests and limitations. Preserve issue completion until its acceptance checks pass.

## Sources and usage

- [VideoGPA captions](https://github.com/Hongyang-Du/VideoGPA/tree/91356baa923e2f6f6991e3f3377fa1b60cc441da/dl3dv_video_captions), [paper](https://arxiv.org/abs/2601.23286).
- [VideoREPA manifests](https://github.com/aHapBean/VideoREPA/tree/8d581dc301e18546a1808436c724f59d259d09f9/finetune/openvid), [paper](https://arxiv.org/abs/2505.23656).
- [OpenVid HF data](https://huggingface.co/datasets/nkp37/OpenVid-1M). Real source clips can be retrieved from archives 30–49 if a later experiment requires them. Record dataset-specific terms and inherited source terms separately from repository code licenses.

## Review decisions

The author proposes the 32K default, full accepted source union, filtering thresholds, seed 42, component-level 90/5/5 split, and opt-in training presets. These are proposed implementation choices for the user to review. Final accepted counts and release hashes remain to be measured in implementation.

## Complete input audit

Full files fetched successfully on November 7, 2026. The two VideoREPA selections share 31,346 video filenames. The smaller selection contains 654 filenames absent from the larger selection, so it is not a strict subset. Keep variants distinct and measure overlap by original-video group during implementation too.

| Input file | Raw rows | SHA-256 |
| --- | --- | --- |
| VideoGPA `captions_1K.json` | 1,000 | `e965cc30e05c89cfa6bc6a591c0547073b68ee3c62b3138519538f76a51c2d4e` |
| VideoGPA `captions_8K.json` | 840 | `e24750d337d5f04a3f5d74926ff95ffb576f2bcf503281d2997a415a6ee47bd8` |
| VideoGPA `captions_9K.json` | 900 | `9fe07d253dd0f57890730a7b45c9fe96e1154fbabec8a5d633eb99db7bdde720` |
| VideoGPA `captions_10K.json` | 909 | `7c14c085ad0a986edaf8a88dbe859fe4920f09585f785d9a199d2593098a76b1` |
| VideoGPA `captions_11K.json` | 498 | `b7e29128eb65e1ce81a4b007eca42f745204bb4f79d8ca79e7787c7b894eb7af` |
| VideoREPA `openvid_3w2.csv` | 32,000 | `7c116ee8363cb3506896be698b839b280bcdd4386c2ab1c3f73a9a3478231eb8` |
| VideoREPA `openvid_6w4.csv` | 64,000 | `7b088e3b5a76c0156967926492cc53d107f579f6fe01e2a6180337239c974956` |
