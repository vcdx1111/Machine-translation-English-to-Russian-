"""Offline CPU API fixture: random tiny Marian weights, NOT a pretrained Helsinki model."""
import argparse
import json
from pathlib import Path
import shutil
import yaml
import torch
from transformers import MarianConfig, MarianMTModel, MarianTokenizer
from src.data import selected_splits
from src.tokenizer import train_tokenizer


def make_fixture(output, raw='data/raw/rus.txt'):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path('configs/helsinki_colab.yaml').read_text())
    cfg['paths'].update(raw=str(Path(raw).resolve()), processed=str(root/'processed'), checkpoints=str(root/'run'), results=str(root/'results'))
    cfg['data'].update(max_train=48,max_val=8,max_test=40,max_length=16)
    cfg['training'].update(device='cpu',epochs=2,batch_size=8,eval_batch_size=8,gradient_accumulation_steps=1,
                           mixed_precision=False,num_workers=0,pin_memory=False,logging_steps=1,progress=False,patience=5)
    cfg['evaluation'].update(batch_size=16,beam_batch_size=8,beam_size=2,max_new_tokens=16)
    rows = selected_splits(cfg)['train']
    tok = train_tokenizer(rows,root/'sp',256)
    model_dir = root/'model'
    model_dir.mkdir(exist_ok=True)
    for name in ('source.spm','target.spm'):
        shutil.copyfile(root/'sp.model',model_dir/name)
    vocab = {tok.sp.id_to_piece(i):i for i in range(len(tok))}
    (model_dir/'vocab.json').write_text(json.dumps(vocab,ensure_ascii=False))
    tokenizer = MarianTokenizer(vocab=str(model_dir/'vocab.json'),source_spm=str(model_dir/'source.spm'),
                                target_spm=str(model_dir/'target.spm'))
    tokenizer.save_pretrained(model_dir)
    torch.manual_seed(42)
    config = MarianConfig(vocab_size=len(tokenizer),decoder_vocab_size=len(tokenizer),d_model=16,
                          encoder_layers=1,decoder_layers=1,encoder_attention_heads=2,decoder_attention_heads=2,
                          encoder_ffn_dim=32,decoder_ffn_dim=32,max_position_embeddings=64,
                          pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,
                          decoder_start_token_id=tokenizer.pad_token_id,forced_eos_token_id=None)
    model = MarianMTModel(config)
    model.generation_config.forced_eos_token_id = tokenizer.eos_token_id
    model.save_pretrained(model_dir)
    cfg['model'].update(name=str(model_dir.resolve()),revision='main')
    config_path = root/'fixture.yaml'
    config_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    return config_path


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='/workspace/marian-cpu-fixture')
    a=p.parse_args()
    print(make_fixture(a.output))
