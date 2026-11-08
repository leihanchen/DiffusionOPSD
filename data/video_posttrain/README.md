---
language:
- en
pretty_name: VideoGPA and VideoREPA training captions
task_categories:
- text-to-video
configs:
- config_name: combined
  default: true
  data_files:
  - split: train
    path: data/combined.parquet
  - split: test
    path: data/videogpa_test.parquet
- config_name: videogpa
  data_files:
  - split: train
    path: data/videogpa.parquet
  - split: test
    path: data/videogpa_test.parquet
- config_name: videorepa_32k
  data_files:
  - split: train
    path: data/videorepa_32k.parquet
- config_name: videorepa_64k
  data_files:
  - split: train
    path: data/videorepa_64k.parquet
---

# VideoGPA and VideoREPA training captions

This dataset combines the released VideoGPA training captions with the exact
32K and 64K OpenVid selections released by VideoREPA. Every configuration has a `train` split. The `combined` and `videogpa`
configurations also expose the original 1,000-caption VideoGPA evaluation subset
as `test`. Both splits have two string columns.

| Column | Meaning |
| --- | --- |
| `text` | Original caption, preserving its whitespace and wording |
| `video_source` | Relative source path, prefixed with `DL3DV/` or `OpenVid/` |

| Configuration | Train | Test |
| --- | ---: | ---: |
| `combined` (default) | 67,801 | 1,000 |
| `videogpa` | 3,147 | 1,000 |
| `videorepa_32k` | 32,000 | — |
| `videorepa_64k` | 64,000 | — |

The VideoREPA selections share 31,346 identical caption/path pairs. The smaller
selection includes 654 clips absent from the larger selection. `combined`
contains their union plus VideoGPA, removing only exact duplicate `(text,
video_source)` pairs. Different videos with the same caption remain distinct.
Source configurations preserve upstream membership and ordering. Combined rows
follow VideoGPA, then the 32K selection, then previously unseen 64K rows.

## Load

```python
from datasets import load_dataset

dataset = load_dataset("cvis-tmu/VideoGPA-VideoREPA", split="train")
test = load_dataset("cvis-tmu/VideoGPA-VideoREPA", split="test")
# Access either exact VideoREPA selection or the VideoGPA captions.
small = load_dataset("cvis-tmu/VideoGPA-VideoREPA", "videorepa_32k", split="train")
large = load_dataset("cvis-tmu/VideoGPA-VideoREPA", "videorepa_64k", split="train")
gpa = load_dataset("cvis-tmu/VideoGPA-VideoREPA", "videogpa", split="train")
```

Use `revision="<dataset commit SHA>"` to reproduce a particular published release.

## Resolve source paths

The downloadable artifacts contain captions and path references. Obtain source
media separately from the upstream collections and arrange it beneath a local
source-data root.

- `OpenVid/<filename>.mp4` identifies a source video in the extracted OpenVid
  collection. VideoREPA's download recipe uses OpenVid archives 30–49.
- `DL3DV/8K/<scene hash>/images_8` illustrates a VideoGPA reference. These
  upstream keys identify extracted-frame directories. Resolve them to frames
  using the DL3DV layout. Mapping them to MP4 files requires upstream metadata.

```python
from pathlib import Path

source_path = Path("/path/to/source-data") / dataset[0]["video_source"]
```

## Provenance

VideoGPA sources are `dl3dv_video_captions/captions_{8K,9K,10K,11K}.json` at
commit `91356baa923e2f6f6991e3f3377fa1b60cc441da`. Its `captions_1K.json` evaluation captions supply the `test` split. All
1,000 records preserve their source order, caption text and frame-directory keys.
The release checks DL3DV scene IDs and normalized caption text for training/test
overlap, including caption comparisons against the combined training collection.
[VideoGPA repository](https://github.com/Hongyang-Du/VideoGPA),
[VideoGPA paper](https://arxiv.org/abs/2601.23286).

VideoREPA sources are `finetune/openvid/openvid_3w2.csv` and
`finetune/openvid/openvid_6w4.csv` at commit
`8d581dc301e18546a1808436c724f59d259d09f9`.
[VideoREPA repository](https://github.com/aHapBean/VideoREPA),
[VideoREPA paper](https://arxiv.org/abs/2505.23656),
[OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M).

`sources.json` records input revisions, row counts and SHA-256 hashes.
`audit.json` records output file hashes and ordered content hashes for each
configuration. Content hashes use UTF-8 JSON records with sorted keys,
`ensure_ascii=False`, default JSON separators, and one trailing newline per row.

## Usage terms and limitations

Follow the upstream terms for captions and source media. OpenVid's dataset card
lists CC-BY-4.0, describes research and non-commercial use, and requires following
the licenses of its contributing source collections. VideoREPA code is
Apache-2.0. VideoGPA's README displays an MIT code badge, while a separate caption
license was not identified in the inspected release. These code licenses do not
establish a blanket license for this combined caption collection or source media.
[OpenVid terms](https://huggingface.co/datasets/nkp37/OpenVid-1M#license),
[VideoREPA code license](https://github.com/aHapBean/VideoREPA/blob/main/LICENSE),
[VideoGPA release](https://github.com/Hongyang-Du/VideoGPA).

Captions are machine-generated and can contain repetition or inaccurate details.
This release preserves the original training selections. It provides training
records without adding quality filters, rewritten captions, or new evaluation
partitions. The two VideoREPA configurations overlap and must be treated as
alternative source selections when making comparisons.

## Reproduce and publish

From the DiffusionOPSD repository, install its normal dataset dependencies and run

```bash
python scripts/prepare_video_posttrain_dataset.py
# Rebuild from verified cached inputs in a separate output directory.
python scripts/prepare_video_posttrain_dataset.py --offline --output outputs/video-posttrain-rebuilt
# Publish with a Hugging Face account permitted to write under cvis-tmu.
python scripts/prepare_video_posttrain_dataset.py --offline \
  --repo-id cvis-tmu/VideoGPA-VideoREPA --push-to-hub
# To add the test split to a previous release, also supply its exact HF commit.
# --expected-hub-revision <previous SHA>
```

Publication uploads all configurations and metadata together and verifies complete
row content after reloading every configuration from the returned commit. An
existing matching release is verified again. Updating an existing release requires
`--expected-hub-revision <current SHA>` and identical training content hashes.
The Hub parent-commit guard rejects concurrent updates. For an independent remote verification, pass
`--verify-hub-revision <SHA>`.
