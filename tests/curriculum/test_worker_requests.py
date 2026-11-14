"""Additional request-isolation gates for the shared VLM boundary."""
import pytest

from diffusionopsd.curriculum.vlm import build_messages


def test_proposal_rejects_private_development_rows():
    r = {'mode': 'propose', 'payload': {'domain': 'G1', 'round_index': 1, 'count': 1,
         'split': 'train', 'sources': {'new': 1}, 'parents': [],
         'feedback': {'development': {'private': {'prompt_id': 'hidden', 'text': 'held-out'}}}}}
    with pytest.raises(ValueError):
        build_messages(r)
