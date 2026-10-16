import csv
import json
import math
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
import yaml
from src.model import Transformer
from src.overfit import overfit
from src.stability import DivergenceError,StepLog,optimizer_update,validation_diverged
from src.train import Noam,run,run_epoch
from src.checkpoints import prune_to_best
from scripts.compare_results import compare


@pytest.mark.parametrize('name',['fast','large'])
def test_full_config_one_batch_overfits(name):
    result=overfit(f'configs/transformer_colab_{name}.yaml')
    print(json.dumps(result))
    assert result['passed'] and result['exact_greedy']
    assert result['final_loss']<result['initial_loss']*.35
    if name=='large':
        assert result['model']['d_model']==512 and result['model']['layers']==6 and result['vocab_size']==16000


def test_preln_initialization_tying_and_masks():
    torch.manual_seed(42)
    model=Transformer(1024,d_model=64,heads=4,layers=2,d_ff=128,dropout=0,max_length=8,
                      norm_first=True,tie_embeddings=True,initialization='scaled_embedding').eval()
    assert isinstance(model.encoder_norm,torch.nn.LayerNorm) and isinstance(model.decoder_norm,torch.nn.LayerNorm)
    assert model.embedding.weight is model.output.weight
    assert model.scale==math.sqrt(64)
    assert model.embedding.weight[1:].std().item()==pytest.approx(64**-.5,rel=.04)
    assert torch.count_nonzero(model.embedding.weight[0])==0
    assert all(torch.count_nonzero(m.bias)==0 for m in model.modules() if isinstance(m,torch.nn.Linear))
    src=torch.tensor([[2,4,3,0]]);tgt=torch.tensor([[2,8,9]])
    original=model(src,tgt)
    changed=tgt.clone();changed[:,-1]=10
    torch.testing.assert_close(original[:,:2],model(src,changed)[:,:2])
    torch.testing.assert_close(original,model(torch.cat((src,torch.zeros(1,2,dtype=torch.long)),1),tgt))
    legacy=Transformer(1024,d_model=64,heads=4,layers=1,d_ff=128,dropout=0,max_length=8)
    assert legacy.embedding.weight is not legacy.output.weight and isinstance(legacy.encoder_norm,torch.nn.Identity)


@pytest.mark.parametrize('bad_gradient',[float('inf'),1e30])
def test_amp_overflow_or_norm_overflow_never_updates_weights_or_noam(bad_gradient,tmp_path):
    model=torch.nn.Linear(2,2)
    optimizer=torch.optim.Adam(model.parameters(),lr=0)
    schedule=Noam(optimizer,16,4)
    scaler=torch.amp.GradScaler('cpu',enabled=True,init_scale=8.)
    before={k:v.clone() for k,v in model.state_dict().items()}
    scaler.scale(model(torch.ones(1,2)).sum()).backward()
    for p in model.parameters(): p.grad.fill_(bad_gradient)
    result=optimizer_update(model,optimizer,schedule,scaler,1.)
    assert not result['updated'] and result['lr_used'] is None and schedule.steps==0
    assert scaler.get_scale()==4.
    for k,v in model.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    scaler.scale(model(torch.ones(1,2)).sum()).backward()
    good=optimizer_update(model,optimizer,schedule,scaler,1.)
    assert good['updated'] and schedule.steps==1 and good['lr_used']==schedule.rate(1)
    log=StepLog(tmp_path/'steps.csv');log.write(1,1,1.,result,'skipped');log.write(1,2,1.,good);log.close()
    rows=list(csv.DictReader((tmp_path/'steps.csv').open()))
    assert rows[0]['lr_used']=='' and rows[1]['optimizer_step']=='1'


def test_nonfinite_loss_logs_and_stops_before_update(tmp_path):
    model=Transformer(16,d_model=16,heads=2,layers=1,d_ff=32,dropout=0,max_length=8)
    with torch.no_grad(): model.output.bias.fill_(float('inf'))
    opt=torch.optim.Adam(model.parameters(),lr=0);schedule=Noam(opt,16,4)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    log=StepLog(tmp_path/'steps.csv')
    with pytest.raises(DivergenceError,match='nonfinite loss'):
        run_epoch(model,[(torch.tensor([[2,4,3]]),torch.tensor([[2,5,3]]))],
                  torch.nn.CrossEntropyLoss(),torch.device('cpu'),opt,schedule,scaler,step_log=log)
    log.close()
    assert schedule.steps==0 and 'nonfinite_loss' in (tmp_path/'steps.csv').read_text()


@pytest.fixture
def tiny_run(tmp_path):
    raw=tmp_path/'raw.tsv'
    raw.write_text(''.join(f'Hello number {i}.\tПривет номер {i}.\tcredit\n' for i in range(60)))
    cfg=yaml.safe_load(Path('configs/transformer_large_cpu.yaml').read_text())
    cfg['paths'].update(raw=str(raw),processed=str(tmp_path/'processed'),checkpoints=str(tmp_path/'ckpt'),results=str(tmp_path/'results'))
    cfg['model'].update(d_model=16,heads=2,layers=1,d_ff=32)
    cfg['tokenizer']['vocab_size']=80
    cfg['training'].update(epochs=4,patience=5)
    cfg['evaluation'].update(val_greedy_samples=0,beam_size=1,repetition_ablation_ngram=0)
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(cfg))
    return config,cfg


def test_val_blowup_retains_best_checkpoint_and_stops(tiny_run):
    config,cfg=tiny_run;original=run_epoch;counter=0
    def injected(*a,**kw):
        nonlocal counter
        value=original(*a,**kw)
        if len(a)==4:  # loss-only validation call
            counter+=1
            return 1. if counter==1 else 1.6
        return value
    with patch('src.train.run_epoch',side_effect=injected): result=run(config)
    output=Path(cfg['paths']['results']);ckpt=Path(cfg['paths']['checkpoints'])
    assert result['training_status']=='diverged' and result['best_epoch']==1 and counter==2
    assert json.loads((output/'run_status.json').read_text())['status']=='diverged'
    assert torch.load(ckpt/'last.pt',weights_only=False)['epoch']==1
    assert (output/'divergence.json').exists()
    with pytest.raises(ValueError,match='diverged'):
        run(config,resume=str(ckpt/'last.pt'))
    with pytest.raises(ValueError,match='Keep resume'):
        prune_to_best(ckpt,output)
    assert validation_diverged(float('nan'),1.,{}) and not validation_diverged(1.1,1.,{})


def test_step_log_replays_partial_epoch(tmp_path):
    p=tmp_path/'steps.csv';log=StepLog(p)
    for i in range(3): log.write(1,i+1,1.)
    log.close();log=StepLog(p,2);log.write(2,1,.5);log.close()
    rows=list(csv.DictReader(p.open()))
    assert len(rows)==3 and rows[-1]['attempt_step']=='3' and rows[-1]['epoch']=='2'


def test_best_only_cleanup_keeps_evaluation_and_refuses_resume(tiny_run):
    from src.evaluate import load_checkpoint
    config,cfg=tiny_run
    cfg['training']['epochs']=2;config.write_text(yaml.safe_dump(cfg));result=run(config)
    output=Path(cfg['paths']['results']);ckpt=Path(cfg['paths']['checkpoints'])
    assert result['training_status']=='completed'
    original=torch.load(ckpt/'last.pt',weights_only=False)
    torch.save(dict(original,run_id='unrelated'),ckpt/'last.pt')
    with pytest.raises(ValueError,match='different experiment'):
        prune_to_best(ckpt,output)
    assert (ckpt/'last.pt').exists()
    torch.save(original,ckpt/'last.pt')
    size_before=sum(p.stat().st_size for p in ckpt.glob('*.pt'))
    report=prune_to_best(ckpt,output)
    assert list(p.name for p in ckpt.glob('*.pt'))==['best.pt'] and report['bytes_after']<size_before
    model,tok,_,state,_=load_checkpoint(ckpt/'best.pt')
    assert not state['resume_supported'] and 'optimizer' not in state and state['training_seconds']==result['training_seconds']
    assert model.embedding.weight is model.output.weight
    from src.evaluate import checkpoint_training_metadata
    metadata=checkpoint_training_metadata(state,output/'recheck')
    assert metadata['run_id']==result['run_id'] and metadata['training_status']=='completed'
    assert metadata['completed_epochs']==2
    prune_to_best(ckpt,output)  # idempotent, inference still usable
    with pytest.raises(ValueError,match='cannot resume'):
        run(config,resume=str(ckpt/'best.pt'))


def test_new_large_peak_and_four_way_comparison(tmp_path):
    cfg=yaml.safe_load(Path('configs/transformer_colab_large.yaml').read_text())
    tr=cfg['training'];schedule=Noam(None,cfg['model']['d_model'],tr['warmup_steps'],tr['lr_factor'])
    assert tr['max_tokens']==16384 and 2000<=tr['warmup_steps']<=4000
    assert schedule.rate(tr['warmup_steps'])<5e-4
    for name,stage,family in [('scratch_fast_v4','trained','scratch_transformer'),('scratch_large_stable_v4','trained','scratch_transformer'),
                              ('zero','zero_shot','helsinki_marian'),('fine','fine_tuned','helsinki_marian')]:
        p=tmp_path/name/'metrics.json';p.parent.mkdir()
        p.write_text(json.dumps(dict(experiment=name,stage=stage,model_family=family,test_sources=2,test_pairs=2,
            schema_version=2,test_split_sha256='same',data_sha256='raw',split_seed=42,config={},test_bleu=20.,test_chrf=40.,
            training_seconds=1.,evaluation_seconds=1.,test_greedy_bleu=19.,test_greedy_chrf=39.,test_decoding='beam',test_beam_size=4)))
    rows=compare([tmp_path],require_four=True)
    assert len(rows)==8 and {r['stage'] for r in rows}=={'trained','zero_shot','fine_tuned'}
    fast=tmp_path/'scratch_fast_v4/metrics.json'
    original=fast.read_text();bad=json.loads(original);bad['test_beam_size']=3;fast.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='full-test beam4'):
        compare([tmp_path],require_four=True)
    fast.write_text(original)
    (tmp_path/'fine/metrics.json').unlink()
    with pytest.raises(ValueError,match='Missing required'):
        compare([tmp_path],require_four=True)
