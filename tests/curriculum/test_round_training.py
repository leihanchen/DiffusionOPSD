import random

import pytest
import torch

from diffusionopsd.video.cosmos_training import (
    RoundCursor, load_training_state, save_training_state, validate_round_transition,
)


def test_complete_one_pass():
    order = list(range(32))
    random.Random(42).shuffle(order)
    cursor = RoundCursor(order)
    visited = []
    while (index := cursor.take()) is not None:
        visited.append(index)
    assert sorted(visited) == list(range(32))
    assert len(visited) * 4 == 128
    assert cursor.take() is None
    with pytest.raises(ValueError):
        RoundCursor([0, 0])


def test_rejected_batch_advances_without_update(tmp_path):
    cursor = RoundCursor([0, 1])
    student, ema = torch.tensor([1.]), torch.tensor([.9])
    cursor.take()
    assert not cursor.record_batch(0)
    path = tmp_path / 'state.pt'
    save_training_state(path, adapters={'default': student, 'old': ema}, optimizer={}, update=0,
                        config={}, extra={'round_cursor': cursor.state_dict()})
    state = load_training_state(path, config={})
    restored = RoundCursor(**state['extra']['round_cursor'])
    assert restored.take() == 1
    assert torch.equal(state['adapters']['default'], student)
    assert torch.equal(state['adapters']['old'], ema)
    assert state['update'] == 0
    assert restored.record_batch(0) is False  # A fully rejected manifest still finishes.
    assert restored.take() is None


def test_resume_matches_uninterrupted(tmp_path):
    torch.manual_seed(42)
    cursor = RoundCursor([2, 0, 1])
    assert cursor.take() == 2
    path = tmp_path / 'state.pt'
    save_training_state(path, adapters={}, optimizer={}, update=1, config={'x': 1},
                        extra={'round_cursor': cursor.state_dict()})
    expected = torch.rand(4)
    state = load_training_state(path, config={'x': 1})
    assert torch.equal(torch.rand(4), expected)
    restored = RoundCursor(**state['extra']['round_cursor'])
    assert [restored.take(), restored.take(), restored.take()] == [0, 1, None]


def test_transition_preserves_invariants_requires_completed_previous_round():
    old = {'model': 'nano', 'learning_rate': 1e-5, 'train_prompt_sha256': 'a',
           'curriculum_round': 0, 'curriculum_manifest_hash': 'a'}
    new = old | {'train_prompt_sha256': 'b', 'curriculum_manifest_hash': 'b', 'curriculum_round': 1}
    state = {'config': old, 'extra': {'round_cursor': {'order': [0, 1], 'next_index': 2}}}
    validate_round_transition(state, new)
    with pytest.raises(ValueError):
        validate_round_transition(state, new | {'learning_rate': 2e-5})
    with pytest.raises(ValueError):
        validate_round_transition(state, new | {'curriculum_round': 2})
    state['extra']['round_cursor']['next_index'] = 1
    with pytest.raises(ValueError):
        validate_round_transition(state, new)
