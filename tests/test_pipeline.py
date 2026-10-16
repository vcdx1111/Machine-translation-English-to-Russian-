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

    def decode(self, tgt, memory, mask):
        logits = torch.full((1, tgt.size(1), 7), -20.)
        if tgt.size(1) == 1:
            logits[0, -1, 4], logits[0, -1, 5] = 0., -.1
        elif tgt[0, -1] == 5:
            logits[0, -1, EOS] = 10
        else:
            logits[0, -1, EOS], logits[0, -1, 6] = 0., 0.
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
    with patch('src.evaluate.translate', side_effect=['Здравствуйте друзья сегодня утром.', 'До свидания мои друзья.']):
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
