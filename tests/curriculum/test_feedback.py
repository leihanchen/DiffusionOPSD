import pytest

from diffusionopsd.curriculum.feedback import allocate_next_round, summarize_round


def row(i, geo=.5, **kwargs):
    return dict(prompt_id=str(i), subtype='orbit', split='train', geo=geo, valid=True) | kwargs


def test_uncertain_scores_stay_unknown():
    report = summarize_round([row(0, float('nan')), row(1, .8), row(1, .4)], [])
    assert report['prompts']['0']['mean_geo'] is None
    assert report['prompts']['1']['mean_geo'] == pytest.approx(.6)
    assert report['prompts']['1']['count'] == 2
    assert 'pass' not in report['prompts']['1']
    assert report['prompts']['1']['observations'] == {}


def test_test_rows_excluded():
    with pytest.raises(ValueError):
        summarize_round([], [row(0, split='test')])
    with pytest.raises(ValueError):
        summarize_round([row(0, split='test')], [])


def test_allocation_budget_and_fallback():
    ids = [str(i) for i in range(32)]
    summary = summarize_round([row(i, i / 32) for i in range(32)], [])
    allocation = allocate_next_round(ids, summary, 42)
    assert [len(allocation[k]) for k in ('retained', 'edit_parents', 'rehearsal')] == [8, 12, 4]
    assert allocation['new_count'] == 8
    assert not set(allocation['retained']) & set(allocation['rehearsal'])
    assert set(allocation['retained']) == {str(i) for i in range(8)}
    assert allocation == allocate_next_round(ids, summary, 42)
    small = allocate_next_round(['0'], summarize_round([row(0)], []), 42)
    assert sum(len(small[k]) for k in ('retained', 'edit_parents', 'rehearsal')) + small['new_count'] == 32
    assert small['shortfall'] > 0


def test_development_is_aggregate_only():
    report = summarize_round([row(0)], [row('private', split='development')])
    assert 'private' not in str(report['development'])
    assert report['development']['orbit']['count'] == 1


def test_development_contract_is_frozen_and_disjoint():
    from diffusionopsd.curriculum.records import PromptRecord
    from diffusionopsd.curriculum.feedback import development_contract

    p = PromptRecord('d', 'd', None, 'development', 'A static scene', 'orbit', {}, 'initial', 0)
    contract = development_contract([p], [], seed=42, seeds_per_prompt=2, expected_count=1)
    assert [q['seed'] for q in contract['queries']] == [1042, 1043]
    assert contract == development_contract([p], [], seed=42, seeds_per_prompt=2, expected_count=1)
    train = PromptRecord('t', 'd', None, 'train', 'A related scene', 'orbit', {}, 'initial', 0)
    with pytest.raises(ValueError):
        development_contract([p], [train], seed=42, seeds_per_prompt=2, expected_count=1)
