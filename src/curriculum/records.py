"""Validated G1 prompt records and content-addressed manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import unicodedata

SUBTYPES = frozenset({'orbit', 'dolly', 'lateral'})
SOURCES = frozenset({'initial', 'retained', 'edited', 'rehearsal', 'new'})
SPLITS = frozenset({'train', 'development', 'test'})


def normalized_text(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    lineage_id: str
    parent_id: str | None
    split: str
    text: str
    subtype: str
    attributes: dict[str, str]
    source: str
    round_index: int

    def __post_init__(self):
        if any(not isinstance(x, str) or not x.strip() for x in (self.prompt_id, self.lineage_id, self.text)):
            raise ValueError('IDs and prompt text must be nonempty strings')
        if self.parent_id is not None and (not isinstance(self.parent_id, str) or not self.parent_id.strip()):
            raise ValueError('Invalid parent ID')
        if self.split not in SPLITS or self.source not in SOURCES or self.subtype not in SUBTYPES:
            raise ValueError('Unsupported split, source, or G1 subtype')
        if type(self.round_index) is not int or self.round_index < 0:
            raise ValueError('Round index must be a nonnegative integer')
        if not isinstance(self.attributes, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in self.attributes.items()
        ):
            raise ValueError('Attributes must map strings to strings')


def validate_manifest(records: list[PromptRecord], history: list[PromptRecord]) -> None:
    if not records:
        raise ValueError('Manifest must not be empty')
    known, lineage_splits, texts = {}, {}, {}
    for record in [*history, *records]:
        if record.lineage_id in lineage_splits and lineage_splits[record.lineage_id] != record.split:
            raise ValueError('A lineage cannot cross partitions')
        lineage_splits[record.lineage_id] = record.split
    for record in history:
        previous = known.get(record.prompt_id)
        if previous and (previous.text, previous.lineage_id, previous.split) != (
            record.text, record.lineage_id, record.split
        ):
            raise ValueError('History changes an immutable prompt identity')
        known[record.prompt_id] = record
        texts[normalized_text(record.text)] = record.prompt_id
    ids, current_texts = set(), set()
    for record in records:
        key = normalized_text(record.text)
        if record.prompt_id in ids or key in current_texts:
            raise ValueError('Duplicate prompt ID or normalized text')
        ids.add(record.prompt_id)
        current_texts.add(key)
        previous = known.get(record.prompt_id)
        if record.source in {'retained', 'rehearsal'}:
            if previous is None or record.round_index <= previous.round_index:
                raise ValueError('Retained/rehearsal records require an earlier prompt')
            if (record.text, record.lineage_id, record.split, record.parent_id, record.subtype, record.attributes) != (
                previous.text, previous.lineage_id, previous.split, previous.parent_id,
                previous.subtype, previous.attributes
            ):
                raise ValueError('An unchanged prompt source cannot modify its content')
        else:
            if previous is not None or key in texts:
                raise ValueError('New prompt identity or text already exists')
            if record.source == 'edited':
                parent = known.get(record.parent_id)
                if parent is None or record.round_index <= parent.round_index:
                    raise ValueError('Edit requires an earlier parent')
                if (record.lineage_id, record.split) != (parent.lineage_id, parent.split):
                    raise ValueError('Edits must inherit lineage and partition')
            elif record.parent_id is not None or record.lineage_id != record.prompt_id:
                raise ValueError('Initial/new prompts must establish their own lineage')


def manifest_hash(records: list[PromptRecord]) -> str:
    payload = json.dumps([asdict(r) for r in records], sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def read_manifest(path: str | Path) -> list[PromptRecord]:
    """Read record structure; callers validate ancestry against the full history."""
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('Manifest must contain a nonempty JSON array')
    return [PromptRecord(**row) for row in rows]
