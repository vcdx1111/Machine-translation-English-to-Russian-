"""Generate a standalone, output-free notebook from the reviewed pipeline sources.

Source files are read only at build time; the resulting notebook imports no project
modules and never executes repository code. --check detects stale embedded code.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import textwrap

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'notebooks/en_ru_translation_full.ipynb'


def embedded(path, omit=(), rename=None):
    tree = ast.parse((ROOT/path).read_text())
    class Standalone(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.level or (node.module and node.module.split('.')[0] in ('src','scripts')):
                return None
            return node
        def visit_If(self, node):
            if isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__':
                return None
            return self.generic_visit(node)
        def visit_FunctionDef(self, node):
            if node.name in omit:
                return None
            if rename and node.name in rename:
                node.name = rename[node.name]
            return self.generic_visit(node)
    return ast.unparse(ast.fix_missing_locations(Standalone().visit(tree)))


def build():
    cells=[]
    def section(markdown, code):
        cells.extend([nbformat.v4.new_markdown_cell(textwrap.dedent(markdown).strip()),
                      nbformat.v4.new_code_cell(textwrap.dedent(code).strip())])
    section(textwrap.dedent('''# English → Russian translation: the complete experiment

    This notebook embeds all pipeline code, including the custom Transformer and Marian Trainer.
    It has **no project imports**. The repository checkout is used only to read the attributed raw data.
    Training is complete: **all training flags default to False**, so Run all loads existing Drive results.
    Enable flags only for optional reproduction; a T4 GPU is required for full-size model runs.
    Final measured training: fast 7.9 min, large 74.1 min, Helsinki fine-tuning 31.7 min, plus evaluation/setup.

    ## 1. Install the pinned environment
    Run installation before importing torch/transformers. In a reused session, restart the runtime if those
    packages were already imported with different versions. CPU smoke mode skips installation and uses tiny random models;
    it is an execution check, not pretrained Helsinki validation.'''), textwrap.dedent('''
    import os
    import subprocess
    import sys
    from pathlib import Path
    CPU_SMOKE = os.environ.get('EN_RU_NOTEBOOK_SMOKE') == '1'
    def stream(command):
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
            status = process.wait()
        if status:
            raise subprocess.CalledProcessError(status, command)
    '''))
    req=[]
    for file in ('requirements.txt','requirements-helsinki.txt'):
        req.extend(s.strip() for s in (ROOT/file).read_text().splitlines() if s.strip() and not s.startswith(('#','-r ')))
    cells[-1].source += '\nPINNED_PACKAGES = '+repr(list(dict.fromkeys(req)))+"\nif not CPU_SMOKE:\n    stream([sys.executable, '-m', 'pip', 'install', *PINNED_PACKAGES])"
    section(textwrap.dedent('''## 2. Drive, run flags and data location
    Existing runs are read from the paths supplied with the final Colab results. No training runs by default.
    New optional training writes distinct `*_full_notebook` directories and refuses to overwrite an existing run.
    Best-only checkpoints omit optimizer/resume state; an interrupted reproduction starts a fresh run.
    Overwriting/replacing a Drive checkpoint can put previous versions in **Drive Trash**. Keeping one visible
    checkpoint does not clear Trash; inspect and empty it separately when reclaiming space.
    '''), textwrap.dedent('''
    import copy
    import csv
    import hashlib
    import json
    import time
    import torch
    import yaml
    RUN_FAST = False
    RUN_LARGE = False
    RUN_HELSINKI_ZERO_SHOT = False
    RUN_HELSINKI_FT = False
    RUN_EXAMPLE_INFERENCE = False  # saved weights only; never training
    if CPU_SMOKE:
        RUN_FAST = RUN_LARGE = RUN_HELSINKI_ZERO_SHOT = RUN_HELSINKI_FT = True
        ROOT = Path(os.environ['EN_RU_NOTEBOOK_OUTPUT']).resolve()
        WORK = ROOT/'work'
        RAW = Path(os.environ['EN_RU_NOTEBOOK_RAW']).resolve()
    else:
        from google.colab import drive
        drive.mount('/content/drive')
        ROOT = Path('/content/drive/MyDrive/en_ru_translation')
        WORK = Path('/content/en_ru_translation_full')
        RAW = WORK/'repository/data/raw/rus.txt'
        REPO_REF = 'feat/scratch-transformer'
        if not RAW.exists():
            WORK.mkdir(parents=True, exist_ok=True)
            stream(['git', 'clone', '--depth', '1', '--branch', REPO_REF,
                    'https://github.com/vcdx1111/Machine-translation-English-to-Russian-.git', str(WORK/'repository')])
    WORK.mkdir(parents=True, exist_ok=True)
    assert RAW.is_file(), RAW
    if any((RUN_FAST, RUN_LARGE, RUN_HELSINKI_ZERO_SHOT, RUN_HELSINKI_FT)) and not CPU_SMOKE:
        assert torch.cuda.is_available(), 'Select a Colab T4 GPU runtime'
    print('Data:', RAW, 'Device:', 'cuda' if torch.cuda.is_available() and not CPU_SMOKE else 'cpu')
    print('All training is disabled by default; existing Drive artifacts are reused.')
    '''))
    section(textwrap.dedent('''## 3. TSV loading, EDA, cleaning and leakage-resistant splitting
    The raw file has English, Russian and original contributor attribution. Inspect all rows, lengths,
    duplicates, missing values and multi-translation sources. Group normalized English variants before splitting;
    train-only tokenizer fitting occurs later. This is the same preprocessing as the modular pipeline.'''), embedded('src/data.py')+'\n\n'+embedded('src/results.py'))
    configs={name:yaml.safe_load((ROOT/path).read_text()) for name,path in [('fast','configs/transformer_colab_fast.yaml'),('large','configs/transformer_colab_large.yaml'),('helsinki','configs/helsinki_colab.yaml')]}
    summary=json.loads((ROOT/'docs/final_results.json').read_text())
    section(textwrap.dedent('''### Inspect the corpus and fix the split
    Full data is split before applying CPU-only subset limits. The final test identity is checked against
    the published hash; token caps never change the reference set. Raw lines are never silently sampled for EDA.'''),
    'ORIGINAL_CONFIGS = '+repr(configs)+'\nFINAL_REPORT = '+repr(summary)+textwrap.dedent('''\n
    FULL_ROWS = prepare(RAW, WORK/'processed', 42)
    AUDIT = json.loads((WORK/'processed/audit.json').read_text())
    print(json.dumps(AUDIT, ensure_ascii=False, indent=2))
    FULL_IDENTITY = test_identity(FULL_ROWS['test'], AUDIT['sha256'], 42)
    assert FULL_IDENTITY['test_split_sha256'] == FINAL_REPORT['test_split_sha256']
    print('Shared held-out identity:', FULL_IDENTITY)
    for name in ('train','val','test'):
        print(name, 'pairs:',len(FULL_ROWS[name]), 'first row:', FULL_ROWS[name][0])
    assert not (set(group_key(r['en']) for r in FULL_ROWS['train']) & set(group_key(r['en']) for r in FULL_ROWS['test']))
    '''))
    section(textwrap.dedent('''## 4. Train-only joint SentencePiece BPE and token-budget batches
    Fast uses an 8k vocabulary and a 24-token cap; large uses 16k and 64. BOS/EOS count toward the cap.
    Length buckets limit padded source+target tokens to 16,384. Embedded notebook callables are not importable
    by spawned worker processes, so notebook DataLoaders use `num_workers=0`; CLI configs retain two workers.
    All model, schedule and batch-budget hyperparameters otherwise match the published configs.'''),embedded('src/tokenizer.py')+'\n\n'+embedded('src/batching.py'))
    section(textwrap.dedent('''## 5. The Transformer, implemented directly in PyTorch
    Explicit multi-head scaled attention, sinusoidal positions, source padding and target causal masks,
    encoder/decoder residual blocks, Post-LN and Pre-LN. Large v4 ties input/output weights and normalizes
    embedding initialization before sqrt(d_model) scaling. No `nn.Transformer` is used.'''),embedded('src/model.py'))
    section(textwrap.dedent('''## 6. Batched greedy/beam decoding and shared multi-reference metrics
    Beam size 4 and length penalty 0.6 are used on the **whole** test set. Greedy is recorded separately.
    The large no-repeat-3 ablation remains separate from the primary score. sacreBLEU 13a and chrF2 share
    exactly the same references with Marian. CPU scores are labelled and must not be read as model quality.'''),embedded('src/translate.py')+'\n\n'+embedded('src/evaluate.py'))
    section(textwrap.dedent('''## 7. Stable optimization, live progress and best-only saving
    Label smoothing, Adam/Noam, clipping, FP16 GradScaler, per-step LR/gradient diagnostics, epoch timing
    and ETA are embedded below. Nonfinite losses/gradients and sharp validation regressions stop a run.
    The original learning loop is preserved; the notebook saver writes **only** compact `best.pt`, never `last.pt`.
    It saves selected weights and tokenizer/config metadata, without optimizer/RNG state. CLI resume is intentionally unavailable here.'''),embedded('src/stability.py')+'\n\n'+embedded('src/train.py',rename={'run':'scratch_run'})+textwrap.dedent('''\n
    def save_checkpoint(path, state):
        path = Path(path)
        if path.name != 'best.pt':
            return  # no last checkpoint, optimizer duplicates or resume state on Drive
        weights = dict(state['best_model'])
        if state['config']['model'].get('tie_embeddings'):
            weights['output.weight'] = weights['embedding.weight']
        compact = {k:state[k] for k in ('run_id','config','best_epoch','history','training_seconds','tokenizer_model','data_sha256')}
        compact.update(best_model=weights, artifact_type='inference_best', resume_supported=False, pipeline_version=3)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary = path.with_suffix('.tmp')
        torch.save(compact, temporary)
        temporary.replace(path)

    def notebook_config(name):
        cfg=copy.deepcopy(ORIGINAL_CONFIGS[name])
        run_name={'fast':'scratch_fast_full_notebook','large':'scratch_large_full_notebook','helsinki':'helsinki_full_notebook'}[name]
        base=ROOT/run_name
        cfg['paths'].update(raw=str(RAW),processed=str(WORK/'processed'),checkpoints=str(base/'checkpoints'),results=str(base/'results'))
        cfg['training']['num_workers']=0
        if CPU_SMOKE:
            cfg['data'].update(max_train=48,max_val=8,max_test=40)
            cfg['training'].update(device='cpu',epochs=1,patience=3,mixed_precision=False,pin_memory=False,progress=False)
            cfg['evaluation'].update(batch_size=16,beam_batch_size=8)
            if name in ('fast','large'):
                # Preserve Post-LN vs Pre-LN/tied behavior, shrink dimensions only for smoke.
                cfg['model'].update(d_model=32,heads=4,layers=1,d_ff=64,max_length=16)
                cfg['tokenizer']['vocab_size']=256
                cfg['training'].update(max_tokens=256,max_sentences=8,bucket_size=32,warmup_steps=10,lr_factor=.2)
                cfg['evaluation']['val_greedy_samples']=4
            else:
                cfg['data']['max_length']=16
                cfg['training'].update(batch_size=8,eval_batch_size=8,gradient_accumulation_steps=1,logging_steps=1)
                cfg['evaluation']['max_new_tokens']=16
        return cfg

    def train_scratch_notebook(name):
        cfg=notebook_config(name)
        path=WORK/(name+'.yaml'); path.write_text(yaml.safe_dump(cfg))
        metrics=scratch_run(path)
        if metrics.get('training_status')=='diverged':
            raise DivergenceError('Inspect step_log.csv and divergence.json; do not compare failed weights')
        return Path(cfg['paths']['results'])
    '''))
    section(textwrap.dedent('''## 8. Fast v4: optional training and full-test evaluation
    Default: reuse the completed fast v4 artifacts. Opt-in: 12 epochs, Post-LN d256/3 layers and the exact
    Noam/batch settings above. A new run name protects the final experiment. Progress streams directly in the cell.'''), textwrap.dedent('''
    RESULT_PATHS={name:ROOT/run['path'] for name,run in zip(('fast','large','zero_shot','finetuned'),FINAL_REPORT['runs'])}
    if RUN_FAST:
        RESULT_PATHS['fast']=train_scratch_notebook('fast')
    else:
        print('Fast: reusing',RESULT_PATHS['fast'])
    '''))
    section(textwrap.dedent('''## 9. Stable large v4: optional training and full-test evaluation
    Default: reuse Drive results. Opt-in: Pre-LN d512/6 layers, tied embeddings, warmup 3,000, factor 0.4,
    16,384 tokens/batch, 30 epochs, patience 5. Large v3 collapsed at an excessive early LR peak; v4 converged.
    no-repeat-3 had no meaningful gain (59.88 → 59.87 BLEU); both policies are recorded when reproducing.'''),textwrap.dedent('''
    if RUN_LARGE:
        RESULT_PATHS['large']=train_scratch_notebook('large')
    else:
        print('Large: reusing',RESULT_PATHS['large'])
    '''))
    section(textwrap.dedent('''## 10. Marian datasets, cached generation and shared evaluation
    Helsinki uses its pretrained tokenizer, **the identical split and multi-reference scorer**, cached batched
    generation and full-test beam 4/greedy. OPUS may contain Tatoeba; pretraining contamination is unresolved.
    Recovered Helsinki metrics and samples are valid completed artifacts even though their original CSV logs are missing.'''),embedded('src/helsinki.py',omit={'run'},rename={'evaluate':'evaluate_helsinki'}))
    # All runtime APIs are embedded. This best-only callback changes storage, not optimization.
    section(textwrap.dedent('''### Best-only Marian storage and an offline CPU fixture
    `Seq2SeqTrainer` retains the same AdamW schedule, smoothing, batching, seeds and loss-based best selection.
    A callback saves only improvements into one `best/` directory (safetensors + tokenizer); it never writes
    epoch checkpoint directories or optimizer state. This trades training resume for Drive capacity.
    CPU validation builds tiny **random** Marian weights locally without downloading Helsinki; its results are explicitly labelled.'''), textwrap.dedent('''
    from transformers import MarianConfig, MarianMTModel, MarianTokenizer
    import shutil

    class NotebookBest(TrainerCallback):
        def __init__(self,path,tokenizer,patience):
            self.path,self.tokenizer,self.patience=Path(path),tokenizer,patience
            self.best=float('inf'); self.bad=0
        def on_evaluate(self,args,state,control,metrics=None,model=None,**kwargs):
            value=metrics['eval_loss']
            if not math.isfinite(value):
                raise DivergenceError('Marian validation loss is nonfinite')
            if value < self.best:
                self.best=value; self.bad=0
                model.save_pretrained(self.path, safe_serialization=True)
                self.tokenizer.save_pretrained(self.path)
                (self.path/'best_selection.json').write_text(json.dumps(dict(epoch=state.epoch,val_loss=value)))
            else:
                self.bad+=1
                if self.bad >= self.patience:
                    control.should_training_stop=True
            return control

    def cpu_marian_fixture(cfg):
        root=WORK/'random_marian_fixture';root.mkdir(exist_ok=True)
        rows=selected_splits(cfg)['train']
        tok=train_tokenizer(rows,root/'sp',256)
        for name in ('source.spm','target.spm'):
            shutil.copyfile(root/'sp.model',root/name)
        (root/'vocab.json').write_text(json.dumps({tok.sp.id_to_piece(i):i for i in range(len(tok))},ensure_ascii=False))
        tokenizer=MarianTokenizer(vocab=str(root/'vocab.json'),source_spm=str(root/'source.spm'),target_spm=str(root/'target.spm'))
        tokenizer.save_pretrained(root)
        torch.manual_seed(42)
        config=MarianConfig(vocab_size=len(tokenizer),decoder_vocab_size=len(tokenizer),d_model=16,
            encoder_layers=1,decoder_layers=1,encoder_attention_heads=2,decoder_attention_heads=2,
            encoder_ffn_dim=32,decoder_ffn_dim=32,max_position_embeddings=64,pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,decoder_start_token_id=tokenizer.pad_token_id,forced_eos_token_id=None)
        model=MarianMTModel(config);model.generation_config.forced_eos_token_id=tokenizer.eos_token_id
        model.save_pretrained(root)
        cfg['model'].update(name=str(root),revision='main')
        return cfg

    def setup_marian():
        cfg=notebook_config('helsinki')
        set_seed(cfg['seed']);torch.set_num_threads(cfg['training']['cpu_threads'])
        if CPU_SMOKE:
            cfg=cpu_marian_fixture(cfg)
        rows=selected_splits(cfg)
        device=torch.device('cpu' if CPU_SMOKE else 'cuda')
        model=AutoModelForSeq2SeqLM.from_pretrained(cfg['model']['name'],revision=cfg['model']['revision']).to(device)
        revision=getattr(model.config,'_commit_hash',None)
        if CPU_SMOKE:
            revision='local:'+hashlib.sha256((Path(cfg['model']['name'])/'model.safetensors').read_bytes()).hexdigest()
        elif not revision:
            raise RuntimeError('Cannot resolve pretrained model snapshot')
        tokenizer=AutoTokenizer.from_pretrained(cfg['model']['name'],revision=revision if not CPU_SMOKE else 'main',use_fast=False)
        identity=test_identity(rows['test'],hashlib.sha256(RAW.read_bytes()).hexdigest(),cfg['seed'])
        root=Path(cfg['paths']['checkpoints']); root.mkdir(parents=True,exist_ok=True)
        if (root/'best').exists():
            raise FileExistsError('Choose a new Helsinki reproduction run name before repeating')
        (root/'run_manifest.json').write_text(json.dumps(dict(config=cfg,identity=identity,resolved_revision=revision),indent=2))
        return cfg,rows,device,model,tokenizer,identity,revision
    '''))
    section(textwrap.dedent('''## 11. Helsinki zero-shot BEFORE any fine-tuning
    Default: load the reported zero-shot run. Enabling fine-tuning also evaluates its fresh base model first,
    even if the standalone zero-shot flag is off. Full-test greedy and beam4 are always recorded.'''),textwrap.dedent('''
    MARIAN_CONTEXT=None
    if RUN_HELSINKI_ZERO_SHOT or RUN_HELSINKI_FT:
        MARIAN_CONTEXT=setup_marian()
        hcfg,hrows,hdevice,hmodel,htok,hidentity,hrevision=MARIAN_CONTEXT
        output=Path(hcfg['paths']['results'])/'zero_shot'
        if (output/'metrics.json').exists():
            raise FileExistsError('Choose a fresh Helsinki reproduction results directory')
        evaluate_helsinki(hmodel,htok,hcfg,hrows['test'],output,'zero_shot',hidentity,hrevision,0.)
        RESULT_PATHS['zero_shot']=output
    else:
        print('Zero-shot: reusing', RESULT_PATHS['zero_shot'])
    '''))
    section(textwrap.dedent('''## 12. Optional Helsinki fine-tuning with Seq2SeqTrainer
    Default: reuse the completed 3-epoch run. The reproduction uses loss-only validation, FP16 on T4,
    batch 64 × accumulation 2, LR 3e−5, 5% warmup and patience 2. Model generation is performed after
    restoring the validation-best model. Live Trainer progress and epoch CSV/ETA stay visible in this cell.
    No recovered historical training log is synthesized when this stage is skipped.'''),textwrap.dedent('''
    if RUN_HELSINKI_FT:
        hcfg,hrows,hdevice,hmodel,htok,hidentity,hrevision=MARIAN_CONTEXT
        tr=hcfg['training']; output=Path(hcfg['paths']['results'])/'finetuned';output.mkdir(parents=True,exist_ok=True)
        root=Path(hcfg['paths']['checkpoints'])
        epoch_log=EpochLog(output,tr['epochs'])
        best=NotebookBest(root/'best',htok,tr['patience'])
        args=Seq2SeqTrainingArguments(output_dir=str(WORK/'trainer_scratch'),num_train_epochs=tr['epochs'],
            per_device_train_batch_size=tr['batch_size'],per_device_eval_batch_size=tr['eval_batch_size'],
            gradient_accumulation_steps=tr['gradient_accumulation_steps'],learning_rate=tr['learning_rate'],
            warmup_ratio=tr['warmup_ratio'],weight_decay=tr['weight_decay'],label_smoothing_factor=tr['label_smoothing'],
            max_grad_norm=tr['gradient_clip'],fp16=hdevice.type=='cuda' and tr['mixed_precision'],use_cpu=hdevice.type=='cpu',
            eval_strategy='epoch',save_strategy='no',logging_steps=tr['logging_steps'],logging_first_step=True,
            load_best_model_at_end=False,predict_with_generate=False,group_by_length=True,dataloader_num_workers=0,
            dataloader_pin_memory=tr['pin_memory'] and hdevice.type=='cuda',dataloader_persistent_workers=False,
            report_to=[],push_to_hub=False,seed=hcfg['seed'],data_seed=hcfg['seed'],disable_tqdm=not tr['progress'])
        trainer=Seq2SeqTrainer(model=hmodel,args=args,
            train_dataset=ParallelDataset(hrows['train'],htok,hcfg['data']['max_length']),
            eval_dataset=ParallelDataset(hrows['val'],htok,hcfg['data']['max_length']),processing_class=htok,
            data_collator=MarianCollator(htok,hmodel.config,8 if hdevice.type=='cuda' else None),callbacks=[epoch_log,best])
        trainer.train()
        hmodel=AutoModelForSeq2SeqLM.from_pretrained(root/'best').to(hdevice)
        elapsed=sum(r['epoch_seconds'] for r in epoch_log.rows)
        evaluate_helsinki(hmodel,htok,hcfg,hrows['test'],output,'fine_tuned',hidentity,hrevision,elapsed)
        RESULT_PATHS['finetuned']=output
        del trainer,hmodel
        MARIAN_CONTEXT=None
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print('Fine-tuned Helsinki: reusing',RESULT_PATHS['finetuned'])
    '''))
    section(textwrap.dedent('''## 13. Exact-identity comparison and verbatim examples
    Missing metrics are reported rather than replaced with fabricated run artifacts. The separately labelled
    user-reported table remains available. Comparison refuses mismatched tests, metric signatures or source counts.
    It does not require Helsinki training CSVs. Standard sample files contain just 30 examples, so special error
    cases may be unavailable; an optional inference section follows.'''),embedded('scripts/compare_results.py')+'\n\n'+embedded('scripts/build_example_table.py')+textwrap.dedent('''\n
    available=[p for p in RESULT_PATHS.values() if (p/'metrics.json').exists()]
    print('User-reported final aggregate scores (not recomputed here):')
    for report in FINAL_REPORT['runs']:
        print(report['name'], 'greedy:',report['greedy_bleu'],report['greedy_chrf'],
              'beam4:',report['beam_bleu'],report['beam_chrf'])
    if available:
        table=compare(available,include_cpu=CPU_SMOKE,require_four=len(available)==4)
        print(write_comparison(table,ROOT/'full_notebook_comparison'))
    for name,path in RESULT_PATHS.items():
        if not (path/'metrics.json').exists():
            print('Missing result:',name,path,'— check the Drive location; no training was started.')
        elif name=='finetuned' and not (path/'train_log.csv').exists():
            print('Helsinki recovered artifacts: completed; train_log.csv unavailable.')
    sample_maps=[read_samples(RESULT_PATHS[name]/'samples.txt') for name in ('fast','large','zero_shot','finetuned')]
    common=set.intersection(*(set(m) for m in sample_maps))
    example_lines=['| English | Fast | Large v4 | Helsinki zero-shot | Helsinki fine-tuned |','| --- | --- | --- | --- | --- |']
    for source in SOURCES+sorted(common-set(SOURCES))[:10]:
        values=[source]+[m.get(source,'Exact output unavailable') for m in sample_maps]
        example_lines.append('| '+' | '.join(v.replace('|',chr(92)+'|') for v in values)+' |')
    examples='\\n'.join(example_lines)+'\\n'
    (ROOT/'full_notebook_examples.md').write_text(examples,encoding='utf-8')
    print(examples)
    '''))
    section(textwrap.dedent('''## 14. Optional exact four-model inference, with NO training
    Leave `RUN_EXAMPLE_INFERENCE=False` to avoid model downloads/loading. Enable only to fill missing examples.
    The paths below follow the original modular notebooks: scratch `checkpoints/best.pt`, Helsinki `checkpoints/final`.
    Adjust them if you moved saved weights; the inline reproduction stores Marian in `checkpoints/best` instead.
    Zero-shot must use the **original resolved base revision** recorded in its metrics; it is not inferred from current `main`.
    Only trusted local checkpoints should be loaded. Each model is freed before the next one to fit T4 memory.'''),textwrap.dedent('''
    if RUN_EXAMPLE_INFERENCE:
        exact={}
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for name,run_name in [('fast','scratch_fast_v4_fullbeam'),('large','scratch_large_stable_v4')]:
            model,tok,cfg,state,_=load_checkpoint(ROOT/run_name/'checkpoints/best.pt')
            cfg['training']['progress']=True
            exact[name]=decode_sources(model,tok,cfg,SOURCES,next(model.parameters()).device,4)
            del model,state
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        base_metrics=json.loads((RESULT_PATHS['zero_shot']/'metrics.json').read_text())
        hcfg=base_metrics['config']; revision=base_metrics['base_revision']
        for name,location in [('zero_shot',hcfg['model']['name']),('finetuned',str(ROOT/'helsinki_v1_seed42/checkpoints/final'))]:
            kwargs={'revision':revision} if name=='zero_shot' else {}
            model=AutoModelForSeq2SeqLM.from_pretrained(location,**kwargs).to(device)
            tok=AutoTokenizer.from_pretrained(location,use_fast=False,**kwargs)
            exact[name]=generate_texts(model,tok,hcfg,SOURCES,device,4)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        lines=['| English | Fast | Large v4 | Helsinki zero-shot | Helsinki fine-tuned |','| --- | --- | --- | --- | --- |']
        for i,source in enumerate(SOURCES):
            lines.append('| '+' | '.join(v.replace('|',chr(92)+'|') for v in [source]+[exact[k][i] for k in ('fast','large','zero_shot','finetuned')])+' |')
        (ROOT/'exact_error_examples.md').write_text('\\n'.join(lines)+'\\n',encoding='utf-8')
        print('\\n'.join(lines))
    else:
        print('Saved-weight example inference skipped. No training or model download is needed for the existing results table.')
    '''))
    section(textwrap.dedent('''## 15. Export plots from REAL artifacts
    Large's curve requires its real epoch log. Missing Helsinki logs are expected and do not prevent plotting.
    Images go to Drive `full_notebook_figures`; copy them to repository `docs/figures/` and commit.
    The score chart is generated only when all four actual metrics are present. CPU fixtures are clearly titled.
    All notebook outputs must be cleared before committing.'''),embedded('scripts/plot_results.py')+textwrap.dedent('''\n
    summary=copy.deepcopy(FINAL_REPORT)
    for report,key in zip(summary['runs'],('fast','large','zero_shot','finetuned')):
        report['path']=str(RESULT_PATHS[key])
    if CPU_SMOKE and available:
        m=json.loads((available[0]/'metrics.json').read_text())
        summary.update(test_sources=m['test_sources'],test_split_sha256_prefix=m['test_split_sha256'],test_split_sha256=m['test_split_sha256'])
        for report in summary['runs']:report['name']='CPU fixture: '+report['name']
    summary_path=WORK/'figure_summary.json';summary_path.write_text(json.dumps(summary,ensure_ascii=False))
    log=RESULT_PATHS['large']/'train_log.csv'
    if len(available)==4:
        print(plot_results(summary_path,log if log.exists() else None,ROOT/'full_notebook_figures',ROOT))
    else:
        print('Figures skipped: full original artifacts unavailable. No epoch curve was fabricated.')
    '''))
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':DEST.name,'provenance':[]}})
    # Stable IDs make regeneration reviewable.
    for i,c in enumerate(nb.cells):
        c['id']=f'en-ru-{i:03d}'
    nbformat.validate(nb)
    return nb


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    serialized=nbformat.writes(build())+'\n'
    if a.check:
        if not DEST.exists() or DEST.read_text()!=serialized:
            raise SystemExit('Standalone notebook is stale: run python scripts/build_full_notebook.py')
        print('Standalone notebook matches embedded pipeline/config sources.')
    else:
        DEST.write_text(serialized);print(DEST)


if __name__=='__main__':main()
