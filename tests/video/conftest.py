"""Keep native H3 tests separate from the base video runtime."""

import pytest


def pytest_collection_modifyitems(items):
    h3_items = [item for item in items if item.path.name.startswith('test_h3_')]
    if not h3_items:
        return

    from diffusionopsd.video.h3_conditioning import runtime_identity

    try:
        runtime_identity()
    except (ImportError, ValueError):
        skip = pytest.mark.skip(reason='Native H3 tests require scripts/run_minimax_h3_python.sh')
        for item in h3_items:
            item.add_marker(skip)
