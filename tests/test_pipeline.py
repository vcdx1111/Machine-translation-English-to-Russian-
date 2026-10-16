import json
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from src.data import clean, group_key, read_raw, split_rows
from src.model import MultiHeadAttention, Transformer, source_mask, target_mask
from src.tokenizer import BOS, EOS, PAD, train_tokenizer
from src.train import Noam, collate, run, run_epoch
from src.translate import decode_ids


def tiny_model():
    return Transformer(32, d_model=16, heads=4, layers=1, d_ff=32, dropout=0, max_length=16)


def test_masks():
    tokens = torch.tensor([[BOS, 8, PAD]])
    assert source_mask(tokens).shape == (1, 1, 1, 3)
    assert target_mask(tokens)[0, 0].tolist() == [[True, False, False], [True, True, False], [True, True, False]]


def test_shapes_and_causality():
    torch.manual_seed(42)
    model = tiny_model().eval()
    src = torch.tensor([[BOS, 4, EOS, PAD], [BOS, 5, 6, EOS]])
    tgt = torch.tensor([[BOS, 7, 8], [BOS, 9, 10]])
    first = model(src, tgt)
    changed = tgt.clone()
    changed[:, -1] = 15
    assert first.shape == (2, 3, 32)
    torch.testing.assert_close(first[:, :2], model(src, changed)[:, :2])
    # Appending masked source padding cannot change predictions.
    torch.testing.assert_close(first, model(torch.cat([src, torch.zeros(2, 2, dtype=torch.long)], 1), tgt))


def test_attention_mask_blocks_value():
    torch.manual_seed(3)
    attn = MultiHeadAttention(8, 2, 0)
    q, kv = torch.randn(1, 2, 8), torch.randn(1, 3, 8)
    mask = torch.tensor([[[[True, True, False]]]])
    result = attn(q, kv, kv, mask)
    changed = kv.clone()
    changed[:, 2] = 1e4
    torch.testing.assert_close(result, attn(q, changed, changed, mask))


def test_clean_and_grouped_split(tmp_path):
    path = tmp_path / 'raw.txt'
    path.write_text('Hi.\tПривет.\tcredit\nHi.\tПривет.\tcredit\nbad\n\t\tcredit\n', encoding='utf-8')
    rows, widths, malformed = read_raw(path)
    assert widths == {3: 3, 1: 1} and malformed == 1
    cleaned, reasons = clean(rows)
    assert len(cleaned) == 1
    assert reasons == {'duplicate_pair': 1, 'empty': 1}
    rows = [{'en': f'Sentence {i}', 'ru': 'Привет'} for i in range(100)]
    rows += [{'en': 'SENTENCE 1!', 'ru': 'Здравствуйте'}]
    splits = split_rows(rows)
    assert {k: sorted(r['en'] for r in v) for k, v in splits.items()} == {k: sorted(r['en'] for r in v) for k, v in split_rows(list(reversed(rows))).items()}
    keys = [{group_key(r['en']) for r in splits[s]} for s in ('train', 'val', 'test')]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])


def test_tokenizer(tmp_path):
    rows = [{'en': 'Hello world!', 'ru': 'Привет мир!'}, {'en': 'Hello!', 'ru': 'Привет!'}]
    tok = train_tokenizer(rows, tmp_path / 'sp', 64)
    assert tok.encode('Hello world!')[0] == BOS
    assert tok.encode('Hello world!')[-1] == EOS
    assert tok.decode(tok.encode('Привет мир!')) == 'Привет мир!'
    assert len(tok.encode('Hello world!', 4)) == 4
    assert tok.sp.pad_id() == PAD


def test_training_step():
    torch.set_num_threads(2)
    model = tiny_model()
    before = model.output.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0)
    scheduler = Noam(optimizer, 16, 4)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    batch = collate([([BOS, 4, EOS], [BOS, 5, EOS]), ([BOS, 6, EOS], [BOS, 7, 8, EOS])])
    loss = run_epoch(model, [batch], nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=.1),
                     torch.device('cpu'), optimizer, scheduler, scaler)
    assert 0 < loss < 100 and scheduler.steps == 1
    assert not torch.equal(before, model.output.weight)


class FakeDecoder(nn.Module):
    """Greedy picks token 4; width two finds the globally better 5 -> EOS."""
    def encode(self, src):
        return src, None

    def decode(self, tgt, memory, mask, last_only=False):
        logits = torch.full((tgt.size(0), tgt.size(1), 7), -20.)
        for i in range(tgt.size(0)):
            if tgt.size(1) == 1:
                logits[i, -1, 4], logits[i, -1, 5] = 0., -.1
            elif tgt[i, -1] == 5:
                logits[i, -1, EOS] = 10
            else:
                logits[i, -1, EOS], logits[i, -1, 6] = 0., -.01
        return logits


def test_decoding_search_and_eos():
    src = torch.tensor([[BOS, 4, EOS]])
    model = FakeDecoder()
    assert decode_ids(model, src, max_length=3, beam_size=1) == [BOS, 4, EOS]
    assert decode_ids(model, src, max_length=3, beam_size=2, length_penalty=0) == [BOS, 5, EOS]


def test_resume_matches_uninterrupted(tmp_path):
    raw = tmp_path / 'rus.txt'
    raw.write_text(''.join(f'Hello number {i}.\tПривет номер {i}.\tcredit\n' for i in range(60)), encoding='utf-8')
    cfg = yaml.safe_load(Path('configs/transformer_cpu.yaml').read_text())
    cfg['paths'].update(raw=str(raw), processed=str(tmp_path / 'processed'))
    cfg['model'].update(d_model=16, d_ff=32, max_length=16, dropout=.1)
    cfg['tokenizer']['vocab_size'] = 80
    cfg['training']['batch_size'] = 16
    cfg['evaluation']['beam_size'] = 1
    def execute(name, epochs, resume=None):
        cfg['training']['epochs'] = epochs
        config = tmp_path / 'config.yaml'
        config.write_text(yaml.safe_dump(cfg))
        return run(config, resume, str(tmp_path / name), str(tmp_path / f'{name}_results'))
    execute('full', 2)
    execute('resumed', 1)
    execute('resumed', 2, str(tmp_path / 'resumed/last.pt'))
    full = torch.load(tmp_path / 'full/last.pt', weights_only=False)
    resumed = torch.load(tmp_path / 'resumed/last.pt', weights_only=False)
    assert full['scheduler_steps'] == resumed['scheduler_steps']
    for key in full['model']:
        torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)
    assert (tmp_path / 'resumed_results/train_log.csv').read_text().count('\n') == 3
    assert json.loads((tmp_path / 'resumed_results/metrics.json').read_text())['test_sources'] > 0
    with pytest.raises(FileExistsError):
        execute('resumed', 2)


def test_multiple_reference_metrics(tmp_path):
    from src.evaluate import evaluate_model
    from unittest.mock import patch
    cfg = yaml.safe_load(Path('configs/transformer_cpu.yaml').read_text())
    rows = [{'en': 'Hello.', 'ru': 'Привет мир сегодня утром.'},
            {'en': 'Hello.', 'ru': 'Здравствуйте друзья сегодня утром.'},
            {'en': 'Goodbye.', 'ru': 'До свидания мои друзья.'}]
    with patch('src.evaluate.decode_sources', return_value=['Здравствуйте друзья сегодня утром.', 'До свидания мои друзья.']):
        metrics = evaluate_model(None, None, cfg, rows, tmp_path, torch.device('cpu'), 0, 'test')
    assert metrics['test_bleu'] == pytest.approx(100.)
    assert metrics['test_chrf'] == pytest.approx(100.)
    assert metrics['test_sources'] == 2
    assert 'nrefs:var' in metrics['bleu_signature']


def test_notebook_valid_and_outputs_cleared():
    import nbformat
    nb = nbformat.read('notebooks/train_transformer.ipynb', as_version=4)
    nbformat.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            assert cell.execution_count is None and cell.outputs == []
            assert i > 0 and nb.cells[i-1].cell_type == 'markdown'


def test_token_batches_budget_coverage_and_epoch_shuffle():
    from src.batching import TokenBatchSampler
    pairs = [([1]*s, [1]*t) for s,t in [(3,4),(4,5),(5,4),(6,7),(9,8),(3,3)]*20]
    sampler = TokenBatchSampler(pairs, 100, 16, 30, seed=42)
    first = list(sampler)
    assert sorted(i for b in first for i in b) == list(range(len(pairs)))
    for batch in first:
        assert len(batch)*(max(len(pairs[i][0]) for i in batch)+max(len(pairs[i][1]) for i in batch)) <= 100
    sampler.set_epoch(1)
    assert list(sampler) != first
    sampler.set_epoch(0)
    assert list(sampler) == first


def test_batched_decoding_matches_individual_and_padding():
    from src.translate import batch_decode_ids
    torch.manual_seed(53)
    model = tiny_model().eval()
    src = torch.tensor([[BOS, 4, EOS, PAD], [BOS, 5, 8, EOS]])
    for width in (1, 3):
        results = batch_decode_ids(model, src, 8, width)
        individual = [decode_ids(model, x[None], 8, width) for x in src]
        # Finished sequences may carry PAD while another batch member keeps decoding.
        def trim(ids):
            return ids[:ids.index(EOS)+1] if EOS in ids else ids
        assert [trim(x) for x in results] == [trim(x) for x in individual]
        assert all(x[0] == BOS and BOS not in x[1:] for x in results)


def test_beam_subset_is_separate_from_full_greedy(tmp_path):
    from src.evaluate import evaluate_model
    from unittest.mock import patch
    cfg = yaml.safe_load(Path('configs/transformer_colab_fast.yaml').read_text())
    cfg['evaluation']['beam_max_sources'] = 1
    rows = [{'en': 'Hello.', 'ru': 'Привет мир сегодня утром.'}, {'en': 'Goodbye.', 'ru': 'До свидания мои друзья.'}]
    with patch('src.evaluate.decode_sources', side_effect=[['Привет мир сегодня утром.', 'До свидания мои друзья.'], ['ошибка']]) as decode:
        metrics = evaluate_model(None, None, cfg, rows, tmp_path, torch.device('cpu'), 0, 'test')
    assert decode.call_args_list[0].args[5] == 1
    assert decode.call_args_list[1].args[5] == 3
    assert metrics['test_decoding'] == 'greedy' and metrics['test_bleu'] == pytest.approx(100)
    assert metrics['beam_subset_sources'] == 1 and metrics['beam_subset_bleu'] < metrics['test_bleu']
    assert metrics['beam_subset_greedy_bleu'] == pytest.approx(100)


def test_fast_resume_uses_token_batches_and_calibration(tmp_path):
    raw = tmp_path/'raw.tsv'
    raw.write_text(''.join(f'Hello number {i}.\tПривет номер {i}.\tcredit\n' for i in range(80)))
    cfg = yaml.safe_load(Path('configs/transformer_fast_cpu.yaml').read_text())
    cfg['paths'].update(raw=str(raw), processed=str(tmp_path/'data'))
    cfg['training'].update(progress=False, num_workers=2)
    cfg['tokenizer']['vocab_size'] = 96
    cfg['model'].update(d_model=16, d_ff=32)
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    def execute(name, resume=None, stop=None, skip=False):
        return run(config, resume, str(tmp_path/name), str(tmp_path/f'{name}_results'), stop, skip)
    execute('full')
    execute('resumed', stop=1, skip=True)
    assert not (tmp_path/'resumed_results/metrics.json').exists()
    estimate = json.loads((tmp_path/'resumed_results/runtime_estimate.json').read_text())
    assert estimate['planned_epochs']==2 and estimate['completed_epoch']==1
    execute('resumed', resume=str(tmp_path/'resumed/last.pt'))
    full = torch.load(tmp_path/'full/last.pt', weights_only=False)
    other = torch.load(tmp_path/'resumed/last.pt', weights_only=False)
    for key in full['model']:
        torch.testing.assert_close(full['model'][key], other['model'][key], rtol=0, atol=0)
    assert full['history'][-1]['logits_dtype']=='torch.float32'
    assert not full['history'][-1]['amp_enabled']


def test_last_position_projection_matches_full_projection():
    model = tiny_model().eval()
    src, tgt = torch.tensor([[BOS,4,EOS]]), torch.tensor([[BOS,5,6]])
    memory, mask = model.encode(src)
    torch.testing.assert_close(model.decode(tgt,memory,mask,last_only=True), model.decode(tgt,memory,mask)[:, -1:])


def test_batch_finished_sentence_stops_and_order_restored():
    from src.translate import batch_decode_ids, translate_many
    class Decoder(nn.Module):
        def encode(self, src):
            return src, None
        def decode(self, tgt, memory, mask, last_only=False):
            logits = torch.full((tgt.size(0),1,8), -100.)
            for i in range(tgt.size(0)):
                token = EOS if memory[i,1]==4 or tgt.size(1)>1 else 6
                logits[i,0,token] = 0
            return logits
    class Tokenizer:
        def encode(self, text, max_length):
            return [BOS,5,7,EOS] if text=='long' else [BOS,4,EOS]
        def decode(self, ids):
            return str(ids[:ids.index(EOS)+1])
    model = Decoder()
    src = torch.tensor([[BOS,4,EOS,PAD],[BOS,5,7,EOS]])
    ids = batch_decode_ids(model,src,8,1)
    assert ids[0]==[BOS,EOS,PAD] and ids[1]==[BOS,6,EOS]
    pred = translate_many(model,Tokenizer(),['long','short'],torch.device('cpu'),8,batch_size=2,progress=False)
    assert pred==[str([BOS,6,EOS]), str([BOS,EOS])]


def test_notebook_streams_stdout_and_stderr_before_exit():
    import contextlib
    import nbformat
    import sys
    import time
    nb = nbformat.read('notebooks/train_transformer.ipynb', as_version=4)
    scope = {}
    exec(nb.cells[2].source.split('REPO_URL =')[0],scope)
    observed = {}
    class Sink:
        def write(self, text):
            for marker in ('early', 'late'):
                if marker in text:
                    observed[marker] = time.perf_counter()
        def flush(self):
            pass
    with contextlib.redirect_stdout(Sink()):
        scope['stream']([sys.executable,'-u','-c',
                         'import sys,time; print("early",flush=True); time.sleep(0.25); print("late",file=sys.stderr,flush=True)'])
    assert observed['late'] - observed['early'] > .15
