from copy import deepcopy
import json

import pytest
import torch
from PIL import Image

from diffusionopsd.curriculum.vlm import build_messages, run_requests, validate_local_checkpoint


def request(mode='propose'):
    return {'mode': mode, 'model_id': 'google/gemma-4-12B-it', 'model_revision': 'a' * 40,
            'seed': 42, 'payload': {'domain': 'G1', 'round_index': 0, 'split': 'train',
                                    'count': 1, 'sources': {'initial': 1}, 'parents': [], 'feedback': {}}}


class Processor:
    def __init__(self, result, tokens=8):
        self.result, self.tokens, self.messages = result, tokens, []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        self.messages.append(deepcopy(messages))
        return {'input_ids': torch.ones(1, self.tokens, dtype=torch.long)}

    def decode(self, tokens, **kwargs):
        assert kwargs['skip_special_tokens'] is False
        return 'encoded'

    def parse_response(self, response, **kwargs):
        return {'role': 'assistant', 'content': json.dumps(self.result)}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.options = []

    @property
    def device(self):
        return self.weight.device

    def generate(self, input_ids, **kwargs):
        assert not torch.is_grad_enabled()
        self.options.append(kwargs)
        return torch.cat([input_ids, torch.tensor([[2, 3]])], dim=1)


def test_worker_uses_frozen_model_and_fresh_role_requests():
    candidate = {'prompt_id': 'p', 'lineage_id': 'p', 'parent_id': None, 'split': 'train',
                 'text': 'Stationary sculpture; camera orbits.', 'subtype': 'orbit', 'attributes': {},
                 'source': 'initial', 'round_index': 0}
    processor, model = Processor([candidate]), Model()
    responses = run_requests([request(), request()], model, processor)
    assert not model.training and not model.weight.requires_grad
    assert len(processor.messages) == 2
    assert all(len(messages) == 2 for messages in processor.messages)
    assert model.options[0]['temperature'] == .7
    assert model.options[0]['max_new_tokens'] == 2048
    assert responses[0]['usage']['input_tokens'] == 8
    assert responses[0]['usage']['output_tokens'] == 2


def test_oversize_input_rejected_before_generation():
    model = Model()
    with pytest.raises(ValueError, match='token'):
        run_requests([request()], model, Processor([], tokens=8193))
    assert not model.options


def test_mixed_model_revisions_rejected():
    other = request()
    other['model_revision'] = 'b' * 40
    with pytest.raises(ValueError):
        run_requests([request(), other], Model(), Processor([]))


def test_diagnosis_hides_numeric_feedback_and_requires_ordered_frames(tmp_path):
    p = tmp_path / 'frame.png'
    Image.new('RGB', (20, 10)).save(p)
    r = request('diagnose')
    r['payload'] = {'prompt': 'A stationary statue', 'requirements': ['stationary'],
                    'evidence': {'frame_count': 1, 'native_fps': 24.,
                                 'frames': [{'frame_id': 0, 'timestamp': 0., 'path': str(p)}]},
                    'scores': {'geo': 123}, 'suggested_edit': 'make easy'}
    messages = build_messages(r)
    assert '123' not in str(messages) and 'make easy' not in str(messages)
    result = {'requirements': [{'requirement_id': 'stationary', 'status': 'unknown',
                                'frame_ids': [], 'reason': 'Only one frame.'}]}
    model = Model()
    run_requests([r], model, Processor(result))
    assert model.options[0]['do_sample'] is False
    assert model.options[0]['max_new_tokens'] == 1024
    r['payload']['evidence']['frames'][0]['path'] = 'https://example.com/image.png'
    with pytest.raises(ValueError):
        build_messages(r)


def test_local_checkpoint_requires_pinned_identity(tmp_path):
    with pytest.raises(ValueError):
        validate_local_checkpoint(tmp_path, 'google/gemma-4-12B-it', 'main')
    (tmp_path / 'opsd_revision.txt').write_text('a' * 40)
    (tmp_path / 'config.json').write_text('{"model_type":"gemma4_unified"}')
    assert validate_local_checkpoint(tmp_path, 'google/gemma-4-12B-it', 'a' * 40) == tmp_path.resolve()
    with pytest.raises(ValueError):
        validate_local_checkpoint(tmp_path, 'google/gemma-4-12B-it', 'b' * 40)


def test_usage_records_selected_sizes_and_visual_tokens(tmp_path):
    path = tmp_path / 'frame.png'
    Image.new('RGB', (600, 300)).save(path)
    r = request('diagnose')
    r['payload'] = {'prompt': 'A statue', 'requirements': ['stationary'],
                    'evidence': {'frame_count': 1, 'native_fps': 24.,
                                 'frames': [{'frame_id': 0, 'timestamp': 0., 'path': str(path)}]}}
    response = {'requirements': [{'requirement_id': 'stationary', 'status': 'unknown',
                                  'frame_ids': [], 'reason': 'Single view'}]}

    class VisualProcessor(Processor):
        def apply_chat_template(self, messages, **kwargs):
            inputs = super().apply_chat_template(messages, **kwargs)
            # Complete processor-returned modality metadata; count actual image positions.
            inputs['mm_token_type_ids'] = torch.tensor([[0, 1, 1, 1, 1, 0, 0, 0]])
            return inputs

    usage = run_requests([r], Model(), VisualProcessor(response))[0]['usage']
    assert usage['visual_tokens'] == 4
    assert usage['selected_frame_sizes'] == [[448, 224]]


def test_loader_requires_offline_pinned_bf16_and_one_explicit_device(tmp_path, monkeypatch):
    import transformers
    import scripts.curriculum_vlm_worker as worker

    calls = []

    class Factory:
        def __init__(self, kind):
            self.kind = kind

        def from_pretrained(self, path, **kwargs):
            calls.append((self.kind, path, kwargs))
            return Model() if self.kind == 'model' else Processor([])

    (tmp_path / 'opsd_revision.txt').write_text('a' * 40)
    (tmp_path / 'config.json').write_text('{"model_type":"gemma4_unified"}')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    # Patch the actual loader boundary; LazyModule may resolve public exports again.
    monkeypatch.setattr(transformers.AutoModelForMultimodalLM, 'from_pretrained', Factory('model').from_pretrained)
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', Factory('processor').from_pretrained)
    model, _ = worker.load_frozen_model(tmp_path, 'google/gemma-4-12B-it', 'a' * 40, 'cuda:1')
    assert not model.training and not model.weight.requires_grad
    assert calls[0][2] == {'local_files_only': True, 'trust_remote_code': False}
    assert calls[1][2] == {'local_files_only': True, 'trust_remote_code': False,
                           'dtype': torch.bfloat16, 'device_map': {'': 'cuda:1'}}
    with pytest.raises(ValueError):
        worker.load_frozen_model(tmp_path, 'google/gemma-4-12B-it', 'a' * 40, 'cuda:2')


@pytest.mark.parametrize('answer', ['{"value":1}', '```json\n{"value":1}\n```'])
def test_json_answer_allows_only_one_complete_document_or_fence(answer):
    from diffusionopsd.curriculum.vlm import parse_json_answer

    result, normalized = parse_json_answer(answer)
    assert result == {'value': 1}
    assert normalized == answer.startswith('```')


@pytest.mark.parametrize('answer', ['Here is the answer: {"value":1}',
                                  '```json\n{}\n```\nExtra commentary', '{} {}',
                                  '```python\n{}\n```'])
def test_json_answer_rejects_prose_multiple_documents_and_wrong_fence(answer):
    from diffusionopsd.curriculum.vlm import parse_json_answer

    with pytest.raises(ValueError):
        parse_json_answer(answer)


def test_diagnosis_numeric_string_frame_ids_are_not_coerced(tmp_path):
    from diffusionopsd.curriculum.vlm import InvalidModelResponse

    path = tmp_path / 'frame.png'
    Image.new('RGB', (20, 10)).save(path)
    r = request('diagnose')
    r['payload'] = {'prompt': 'A statue', 'requirements': ['stationary'],
                    'evidence': {'frame_count': 1, 'native_fps': 24.,
                                 'frames': [{'frame_id': 0, 'timestamp': 0., 'path': str(path)}]}}
    result = {'requirements': [{'requirement_id': 'stationary', 'status': 'observed_pass',
                               'frame_ids': ['0'], 'reason': 'Visible object'}]}
    with pytest.raises(InvalidModelResponse, match='Evidence'):
        run_requests([r], Model(), Processor(result))
