# EAS-15 combined Hugging Face dataset design

Updated November 7, 2026 following the user's simplified scope. “VideoGPT” is interpreted as VideoGPA in the context of EAS-15.

## Deliverable

Collect VideoGPA's 3,147 released training captions and both exact VideoREPA training selections, 32K and 64K. Publish one Hugging Face dataset repository under `cvis-tmu`, with string columns `text` and `video_source`. Deliver the reproducible preparation/upload script and a draft PR linked to [EAS-15](https://linear.app/cvis-tmu/issue/EAS-15/create-new-training-dataset-with-videogpa-and-videorepa).

Dataset ID is `cvis-tmu/VideoGPA-VideoREPA`. Use a single `train` split and four named configurations so the exact upstream selections remain accessible.

| Configuration | Rows | Content |
| --- | --- | --- |
| `videogpa` | 3,147 | VideoGPA 8K/9K/10K/11K caption files |
| `videorepa_32k` | 32,000 | Exact `openvid_3w2.csv` selection |
| `videorepa_64k` | 64,000 | Exact `openvid_6w4.csv` selection |
| `combined` (default) | 67,801 | Union of all three configurations, deduplicated by exact `(text, video_source)` |

Complete-file audit confirms 31,346 identical VideoREPA rows appear in both selections, yielding 64,654 distinct VideoREPA rows. Prefixing source paths distinguishes DL3DV and OpenVid. Preserve caption strings exactly, including their original whitespace, and preserve distinct videos sharing a caption.

## Columns

- `text` is the original VideoGPA JSON value or VideoREPA CSV `caption`.
- `video_source` is a POSIX relative source reference. Use `DL3DV/<original JSON key>` for VideoGPA and `OpenVid/<video filename>` for VideoREPA. Reject absolute paths and parent traversal.

VideoGPA's released keys identify extracted-frame directories, for example `8K/<scene hash>/images_8`. Preserve those references and document that resolving them yields source frames. An MP4 filename must be obtained from upstream metadata before any conversion to an actual video-file path. OpenVid references identify MP4 files in the extracted source collection. Paths are resolved against a user-provided source-data root. The HF artifact contains caption/path records in Parquet.

## Preparation and publication

Use the pinned repositories and byte hashes below. Fetch the four VideoGPA training JSONs and the two VideoREPA CSVs. Validate their schemas, counts and source paths. Export each configuration deterministically and record source hashes, configuration row counts and output hashes in a small manifest. Retain the raw caption content and exact source membership.

Publish the Parquet files and a dataset card in one HF dataset commit, with all four configurations declared and `combined` marked as default. The card explains sources, path conventions, overlap, citations and applicable upstream usage terms. Check namespace access and the intended repository before upload. The user explicitly authorized uploading to the `cvis-tmu` HF account. Reuse local credentials without logging tokens.

After publication, reload every configuration from the returned HF commit and compare schemas, row counts and row content with local outputs. Include `load_dataset` usage and the verified HF URL/revision in the PR.

## Implementation boundaries

The deliverable is source collection, deterministic conversion and HF publication. Training integration, quality filtering, rewritten captions, new train/validation/test assignments and motion tags are deferred. Existing training code can consume `text` in a later task.

Use the existing worktree `/lustre06/project/6062393/leihan/DiffusionOPSD-eas15-dataset` and branch `codex/eas-15-video-training-dataset`. Store source cache and generated exports in ignored paths. Commit the preparation/upload script, pinned source recipe, aggregate audit and usage documentation. Propose the implementation as a draft PR against `main` in `leihanchen/DiffusionOPSD`.

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

## Published release

Published [cvis-tmu/VideoGPA-VideoREPA](https://huggingface.co/datasets/cvis-tmu/VideoGPA-VideoREPA) at commit `95f93059b25b2fbe4759d8e6be3aa36b30e85b62`. All configurations and the default combined view were reloaded and verified against the complete local release. Independent source-to-Parquet equality and an offline second-build hash comparison passed.

## VideoGPA original test subset update

The user requested publishing the original 1K evaluation subset as test. The pinned `captions_1K.json` source supplies exactly 1,000 rows in both `combined/test` and `videogpa/test`. The columns remain `text` and `video_source`. Training Parquet byte hashes match the initial release. Normalized training/test captions and DL3DV scene IDs are checked for overlap. Publication requires an explicit current parent revision, preserving training content and rejecting concurrent updates.

Published HF revision `a6909e889f5bb134778aaf6910f44cf18c7b8457` was reloaded, with independent anonymous source-to-HF equality for every test record. The dataset card and preparation script reproduce both splits.

## Combined metadata update

The user requested `sample_id`, `source_dataset`, `source_subsets`, `source_group_id`, and `media_type` in the combined data. Both combined splits now have seven columns. Exact-source configurations retain their two-column schema, with a dedicated combined test Parquet for schema consistency. IDs hash the compact UTF-8 JSON array `[text, video_source]`. Membership lists retain both VideoREPA subsets where applicable. Group IDs identify original DL3DV scenes or OpenVid source videos. The dataset card specifies all conventions.

Published revision `b5c9967cb7801ceec0b3581839d065131eefc5be` reloaded successfully. Independent checks cover all 67,801 training and 1,000 test rows, unique IDs, exact memberships and source groups. Offline rebuilding reproduces all output hashes. Original caption/path content is preserved.
