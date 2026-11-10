# EAS-17: use the combined HF dataset for video OPSD

## Approved scope

Default Wan (simple and distributed epoch) and Cosmos to
`cvis-tmu/VideoGPA-VideoREPA`, configuration `combined`, revision
`b5c9967cb7801ceec0b3581839d065131eefc5be`.
Use 67,801 training records and 1,000 held-out test records. Keep all seven
published fields and distinct sample IDs even when captions repeat.
Generate from text; `video_source` is provenance, not downloaded conditioning video.

Keep existing optimizer update budgets. Distributed Wan cycles a seeded
permutation into four-record batches without dropping the uneven tail.
Cosmos cycles the same permutation using its durable sampling-attempt cursor.
Simple Wan retains random record subsets per update with a local seeded RNG.
Record dataset identity and hashes in run metadata; resumable trainers reject
changed data or sampling configuration. Preserve explicit legacy file configs.
Validate the dataset before model setup, including a CPU-only command, and
check the offline cache before Slurm workers start. No training job is submitted.

## Implementation checklist

- [x] Shared pinned loader, integrity checks, split checks, record identity and scheduling.
- [x] Wan and Cosmos defaults, trainer and evaluation integration.
- [x] Login-node prefetch and offline Slurm preflight.
- [x] Real published dataset checks through all three validation entrypoints.
- [x] CPU regression tests, shell syntax and focused lint.
- [x] Independent final review and local commit.

## Verification focus

Check duplicate-caption identity, uneven-tail cycling at 4/8/16 workers,
resume mismatches, offline failures, source metadata in evaluation outputs,
and that validation-only entrypoints avoid model imports.

## Execution notes

- Initial new-loader test failed before implementation (module absent); the
  first four loader/scheduler tests then passed. Expanded validation tests
  exposed an Arrow union-schema case, fixed by validating fields before hashes.
- Integration tests exposed the image base config's string `dataset` field.
  Wan explicitly removes that inherited field before assigning its structured
  video dataset config; image presets remain unchanged.
- Real HF prefetch and offline validation succeeded with 67,801/1,000 records.
  All three training entrypoints validate before model imports or CUDA setup.
- Final independent review identified standalone evaluation provenance loss
  and expansion of the default evaluation budget. A regression test reproduced
  the missing provenance; the implementation now exports source records,
  scores, seeds and dataset identity. Default standalone evaluation retains
  64 records, with `--num-prompts 0` explicitly selecting the full test split.
- Simple Wan retains policy-only checkpoint compatibility and saves dataset
  metadata in adjacent `.dataset.json` sidecars. It has no training-state resume.
- No GPU updates or Slurm jobs were run for this dataset integration.

## Final validation

- Full video suite: `python -m pytest tests/video -q` in the existing isolated
  Cosmos container runtime: **270 passed**, two existing dependency warnings.
- Focused Ruff checks and syntax checks for all four modified Slurm scripts passed.
- Pinned real dataset prefetch, offline cache check and all three
  `--validate-dataset-only` commands passed. Budgets remain 100,000 for generic
  Wan and 50 for distributed Wan and Cosmos; debug presets retain their budget.
- Final review findings were addressed; no deferred review findings remain.
- Logs are local under `.cache/eas17/` and are not committed.
