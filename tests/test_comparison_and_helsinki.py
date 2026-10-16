import json
from pathlib import Path
from unittest.mock import patch

import nbformat
import pytest
import torch
import yaml

from src.results import test_identity as heldout_identity
from src.translate import block_repeated_ngrams
from scripts.compare_results import compare, write_comparison


def test_repeated_ngram_mask_on_off_and_suffix():
    seq = torch.tensor([[2,4,5,6,4,5],[2,5,6,4,6,5]])
    logits = torch.zeros(2,10)
    assert torch.equal(block_repeated_ngrams(logits,seq,0),logits)
    out = block_repeated_ngrams(logits,seq,3)
    assert torch.isneginf(out[0,6])
    assert torch.isfinite(out[0,7]) and torch.isfinite(out[1]).all()
    assert not torch.isneginf(logits).any()


def test_large_full_beam_greedy_and_suppression_all_same_sources(tmp_path):
    from src.evaluate import evaluate_model
    cfg=yaml.safe_load(Path('configs/transformer_colab_large.yaml').read_text())
    rows=[{'en':'Hello.','ru':'Привет мир сегодня утром.'},{'en':'Goodbye.','ru':'До свидания мои друзья.'}]
    outputs=['Привет мир сегодня утром.','До свидания мои друзья.']
    with patch('src.evaluate.decode_sources',return_value=outputs) as decode:
        m=evaluate_model(None,None,cfg,rows,tmp_path,torch.device('cpu'),0,'hash')
    assert [c.args[5] for c in decode.call_args_list]==[4,1,4]
    assert decode.call_args_list[-1].kwargs['no_repeat_ngram_size']==3
    assert all(c.args[3]==['Hello.','Goodbye.'] for c in decode.call_args_list)
    assert m['test_beam_bleu']==m['test_greedy_bleu']==pytest.approx(100)
    assert m['repetition_ablation']['bleu']==pytest.approx(100)
    assert (tmp_path/'samples_no_repeat.txt').exists()


def test_test_identity_and_comparison_reject_mismatch(tmp_path):
    a=[{'en':'Hi','ru':'Привет'},{'en':'Hi','ru':'Здравствуйте'}]
    identity=heldout_identity(a,'raw',42)
    assert identity==heldout_identity(list(reversed(a)),'raw',42)
    assert identity['test_split_sha256']!=heldout_identity(a[:1],'raw',42)['test_split_sha256']
    def save(name,meta):
        p=tmp_path/name/'metrics.json'
        p.parent.mkdir()
        p.write_text(json.dumps(dict(identity,test_bleu=20.,test_chrf=40.,test_decoding='beam',test_beam_size=4,
                                    test_greedy_bleu=19.,test_greedy_chrf=39.,training_seconds=1.,evaluation_seconds=2.,**meta)))
        return p
    p1=save('scratch',{})
    p2=save('helsinki',{'stage':'zero_shot'})
    rows=compare([p1,p2])
    assert len(rows)==4 and all(r['identity']=='verified' for r in rows)
    assert 'greedy' in write_comparison(rows,tmp_path/'comparison')
    broken=json.loads(p2.read_text()); broken['test_split_sha256']='different'; p2.write_text(json.dumps(broken))
    with pytest.raises(ValueError,match='reference sets'):
        compare([p1,p2])


def test_both_notebooks_valid_empty_and_default_configs():
    for path,config in [('notebooks/train_transformer.ipynb','configs/transformer_colab_large.yaml'),
                        ('notebooks/finetune_helsinki.ipynb','configs/helsinki_colab.yaml')]:
        nb=nbformat.read(path,as_version=4); nbformat.validate(nb)
        assert config in '\n'.join(c.source for c in nb.cells)
        for i,c in enumerate(nb.cells):
            if c.cell_type=='code':
                assert not c.outputs and c.execution_count is None and nb.cells[i-1].cell_type=='markdown'


@pytest.fixture
def marian_fixture(tmp_path):
    from scripts.make_marian_fixture import make_fixture
    raw=tmp_path/'raw.tsv'
    raw.write_text(''.join(f'Hello number {i}.\tПривет номер {i}.\tcredit\n' for i in range(80)))
    config=make_fixture(tmp_path/'fixture',raw)
    cfg=yaml.safe_load(config.read_text())
    cfg['data'].update(max_train=24,max_val=4,max_test=4)
    cfg['evaluation'].update(max_new_tokens=8)
    config.write_text(yaml.safe_dump(cfg))
    return config


def test_helsinki_local_trainer_resume_and_result_schema(marian_fixture,tmp_path):
    from src.helsinki import run, ParallelDataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    # Same planned schedule: full two epochs vs stop-after-one and resume.
    full_c,full_r=tmp_path/'full',tmp_path/'full_results'
    resumed_c,resumed_r=tmp_path/'resumed',tmp_path/'resumed_results'
    full=run(marian_fixture,checkpoint_dir=str(full_c),results_dir=str(full_r))
    run(marian_fixture,checkpoint_dir=str(resumed_c),results_dir=str(resumed_r),stop_after_epochs=1,skip_test=True)
    assert (resumed_r/'zero_shot/metrics.json').exists()
    assert not (resumed_r/'finetuned/metrics.json').exists()
    resumed=run(marian_fixture,resume='latest',checkpoint_dir=str(resumed_c),results_dir=str(resumed_r))
    m1=AutoModelForSeq2SeqLM.from_pretrained(full_c/'final')
    m2=AutoModelForSeq2SeqLM.from_pretrained(resumed_c/'final')
    for key,value in m1.state_dict().items():
        torch.testing.assert_close(value,m2.state_dict()[key],rtol=0,atol=0)
    assert full['test_split_sha256']==resumed['test_split_sha256']
    assert resumed['purpose']=='cpu_smoke_only' and not resumed['pretrained_model_validated']
    for stage in ('zero_shot','finetuned'):
        assert all((resumed_r/stage/f).is_file() for f in ('metrics.json','train_log.csv','samples.txt'))
    manifest=json.loads((resumed_c/'run_manifest.json').read_text())
    assert manifest['identity']['test_split_sha256']==resumed['test_split_sha256']
    assert len(compare([full_r,resumed_r],include_cpu=True))==8
    cfg=yaml.safe_load(marian_fixture.read_text()); cfg['seed']=99; marian_fixture.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='manifest'):
        run(marian_fixture,resume='latest',checkpoint_dir=str(resumed_c),results_dir=str(resumed_r))


def test_marian_collator_shifts_padded_labels_without_gpu_model(marian_fixture):
    from src.helsinki import MarianCollator, ParallelDataset
    from transformers import AutoTokenizer, AutoConfig
    cfg=yaml.safe_load(marian_fixture.read_text())
    tok=AutoTokenizer.from_pretrained(cfg['model']['name'],use_fast=False)
    mc=AutoConfig.from_pretrained(cfg['model']['name'])
    dataset=ParallelDataset([{'en':'Hello','ru':'Привет'},{'en':'Hello number','ru':'Привет номер два'}],tok,16)
    collator=MarianCollator(tok,mc)
    batch=collator([dataset[0],dataset[1]])
    expected=torch.full_like(batch['labels'],mc.pad_token_id)
    expected[:,0]=mc.decoder_start_token_id
    expected[:,1:]=batch['labels'][:,:-1]
    expected.masked_fill_(expected==-100,mc.pad_token_id)
    assert torch.equal(batch['decoder_input_ids'],expected)
    assert (batch['labels']==-100).any()
    assert not hasattr(collator,'model')


def test_trainer_sets_spawn_on_accelerate_base_loader(marian_fixture,tmp_path):
    from src.helsinki import MarianTrainer, MarianCollator, ParallelDataset
    from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,Seq2SeqTrainingArguments
    cfg=yaml.safe_load(marian_fixture.read_text())
    model=AutoModelForSeq2SeqLM.from_pretrained(cfg['model']['name'])
    tok=AutoTokenizer.from_pretrained(cfg['model']['name'],use_fast=False)
    data=ParallelDataset([{'en':'Hello','ru':'Привет'}]*4,tok,16)
    args=Seq2SeqTrainingArguments(output_dir=str(tmp_path/'out'),use_cpu=True,report_to=[],dataloader_num_workers=2)
    trainer=MarianTrainer(model=model,args=args,train_dataset=data,eval_dataset=data,
                          data_collator=MarianCollator(tok,model.config),processing_class=tok)
    for loader in (trainer.get_train_dataloader(),trainer.get_eval_dataloader()):
        base=getattr(loader,'base_dataloader',loader)
        assert base.multiprocessing_context.get_start_method()=='spawn'
