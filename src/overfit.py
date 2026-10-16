"""One-batch learnability diagnostic: real model width/depth, configured vocabulary, FP32 CPU only."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
import yaml
from .model import Transformer
from .tokenizer import BOS,EOS,PAD
from .translate import batch_decode_ids


def overfit(config_path, max_steps=80):
    cfg=yaml.safe_load(Path(config_path).read_text())
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model_cfg=dict(cfg['model'],max_length=8)
    vocab=cfg['tokenizer']['vocab_size']
    model=Transformer(vocab,**model_cfg)
    positive=0.9+0.1/vocab
    floor=-positive*math.log(positive)-(vocab-1)*(0.1/vocab)*math.log(0.1/vocab)
    src=torch.tensor([[BOS,4,5,EOS],[BOS,6,7,EOS]])
    tgt=torch.tensor([[BOS,8,9,EOS],[BOS,10,11,EOS]])
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,betas=(.9,.98),eps=1e-9)
    criterion=torch.nn.CrossEntropyLoss(ignore_index=PAD,label_smoothing=.1)
    def loss():
        logits=model(src,tgt[:,:-1]);return criterion(logits.reshape(-1,vocab),tgt[:,1:].reshape(-1))
    model.eval()
    with torch.no_grad(): initial=loss().item()
    begin=time.perf_counter()
    final,decoded=initial,[]
    for step in range(1,max_steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        value=loss()
        if not torch.isfinite(value): raise RuntimeError('Nonfinite toy overfit loss')
        value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step%5==0 or step==max_steps:
            model.eval()
            with torch.no_grad(): final=loss().item()
            decoded=batch_decode_ids(model,src,5,1)
            canonical=[x[:x.index(EOS)+1] if EOS in x else x for x in decoded]
            if final<floor+.3 and final<initial*.35 and canonical==tgt.tolist(): break
    canonical=[x[:x.index(EOS)+1] if EOS in x else x for x in decoded]
    result=dict(config=str(config_path),model=model_cfg,vocab_size=vocab, theoretical_smoothed_loss_floor=floor,steps=step,initial_loss=initial,final_loss=final,
                loss_ratio=final/initial,exact_greedy=(canonical==tgt.tolist()),seconds=time.perf_counter()-begin,
                passed=final<floor+.3 and final<initial*.35 and canonical==tgt.tolist(),
                diagnostic='Full width/depth/vocabulary, dropout from YAML, label smoothing .1; fixed LR 3e-4 for one-batch memorization, NOT production warmup validation')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/transformer_colab_large.yaml')
    p.add_argument('--output',default='results/overfit.json');a=p.parse_args()
    r=overfit(a.config);Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)
    if not r['passed']: raise SystemExit(1)
