import numpy as np
import pytest
from diffusionopsd.stat_tracking import PerPromptStatTracker


@pytest.mark.parametrize("global_std", [False, True])
@pytest.mark.parametrize("channels", [1, 2])
def test_repeated_prompt_uses_history_across_updates(global_std, channels):
    tracker = PerPromptStatTracker(global_std=global_std)
    first = np.array([1., 3., 10.])
    second = np.array([5., 14., 7.])
    if channels == 2:
        first = np.stack([first, first * 2], axis=1)
        second = np.stack([second, second * 2], axis=1)
    tracker.update(["a", "a", "b"], first)
    actual = tracker.update(["a", "b", "a"], second)
    expected = np.empty_like(second)
    for prompt, indices, old_indices in [("a", [0, 2], [0, 1]), ("b", [1], [2])]:
        history = np.concatenate([first[old_indices], second[indices]])
        std = np.std(second if global_std else history, axis=0) + 1e-4
        expected[indices] = (second[indices] - np.mean(history, axis=0)) / std
    np.testing.assert_allclose(actual, expected)
    assert tracker.get_stats() == (3., 2)
    tracker.clear()
    reset = tracker.update(["a", "a"], first[:2])
    np.testing.assert_allclose(reset, (first[:2] - first[:2].mean(axis=0)) / (first[:2].std(axis=0) + 1e-4))
