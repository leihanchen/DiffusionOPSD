# Video training data

Wan and Cosmos now default to the `combined` configuration of
[cvis-tmu/VideoGPA-VideoREPA](https://huggingface.co/datasets/cvis-tmu/VideoGPA-VideoREPA)
at revision `b5c9967cb7801ceec0b3581839d065131eefc5be`:
67,801 training records and 1,000 held-out test records. The published seven
fields are retained in sampling/evaluation logs. Training uses `text` only;
`video_source` records the relative upstream media path and does not cause
videos to be downloaded or used as conditioning.

On the login node, populate the same `HF_HOME` used by the job (default `$PWD/hf`):

```bash
HF_HUB_OFFLINE=0 bash scripts/run_cosmos_python.sh scripts/prefetch_video_dataset.py
```

Check the cache without network access or GPUs:

```bash
bash scripts/run_cosmos_python.sh scripts/prefetch_video_dataset.py --check-only
bash scripts/run_cosmos_python.sh scripts/train_opsd_video_wan.py --validate-dataset-only
bash scripts/run_cosmos_python.sh scripts/train_opsd_video_wan_epoch.py --validate-dataset-only
bash scripts/run_cosmos_python.sh scripts/train_opsd_video_cosmos.py --validate-dataset-only
```

Both commands accept `--config path/to/preset.py` to validate a particular
preset. The validation-only command reads the preset as written; it does not
accept training-only options or dotted config overrides. Slurm entrypoints
check the cache before launching workers. A missing or corrupted cache is an
error, never a fallback to old prompt files. Only the small Parquet metadata
files and audit/source records are fetched.

Dataset files, ordered row hashes, schema, sample IDs, and train/test scene and
caption separation are checked. Captions may repeat within training when they
refer to different videos: statistics use `sample_id`, not caption text.
Distributed Wan uses seed 42 to shuffle record indices and cycles that order
into four-record updates, including the uneven dataset tail. Cosmos uses the
same seeded order, advancing its checkpointed cursor on each sampling attempt
(including rejected attempts). Simple Wan samples distinct record indices per
update using a local seed-42 RNG. Existing update budgets and evaluation counts
are unchanged; a 50-update run does not traverse the entire training split.

Run manifests store the revision, hashes and sampling settings. Distributed Wan
also saves the dataset identity and order in its checkpoint; Cosmos includes
identity in its checkpoint config signature. Their resume checks reject changed
data/settings. Simple Wan continues to save policy snapshots with `.dataset.json` provenance sidecars and does not
support training-state resume. Start a new run when switching an old run to HF.
For an explicitly legacy experiment, set `dataset.source = "files"` and retain
the old train/eval path fields. Configs without dataset settings also use files.
Standalone Wan evaluation defaults to the first 64 HF test records, preserving
the previous generation budget. Use `--num-prompts 0` for the full test split or
a positive count for a smaller run. `--prompts path.txt` explicitly selects a
local evaluation file (all its records by default). Evaluation JSON includes
the dataset identity and individual source records, seeds and scores.
