"""Execute the standalone notebook's Python cells with tiny offline CPU fixtures.

No repository modules may be imported. The output directory must be new. GPU-sized
learning is disabled; this check builds a tiny random Marian fixture locally.
"""
import argparse
import contextlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import time

import nbformat

ROOT=Path(__file__).resolve().parents[1]


class NoProjectImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('src','scripts'):
            raise ImportError(f'Standalone notebook tried to import project module {fullname}')
        return None


def validate(output):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    notebook=nbformat.read(ROOT/'notebooks/en_ru_translation_full.ipynb',as_version=4)
    nbformat.validate(notebook)
    os.environ.update(EN_RU_NOTEBOOK_SMOKE='1',EN_RU_NOTEBOOK_OUTPUT=str(output),
                      EN_RU_NOTEBOOK_RAW=str(ROOT/'data/raw/rus.txt'),MPLCONFIGDIR=str(output/'matplotlib'))
    guard=NoProjectImports();sys.meta_path.insert(0,guard)
    namespace={'__name__':'standalone_notebook','__file__':str(output/'standalone.py')}
    durations=[];previous=Path.cwd();os.chdir(output)
    try:
        with (output/'execution.log').open('w',encoding='utf-8',buffering=1) as log:
            class Tee:
                def __init__(self,stream):self.stream=stream
                def write(self,text):log.write(text);return self.stream.write(text)
                def flush(self):log.flush();self.stream.flush()
            with contextlib.redirect_stdout(Tee(sys.stdout)),contextlib.redirect_stderr(Tee(sys.stderr)):
                for i,cell in enumerate(notebook.cells):
                    if cell.cell_type!='code':continue
                    assert notebook.cells[i-1].cell_type=='markdown'
                    assert cell.execution_count is None and not cell.outputs
                    begin=time.perf_counter()
                    print(f'Executing standalone cell {i}',flush=True)
                    exec(compile(cell.source,f'<notebook-cell-{i}>','exec'),namespace)
                    durations.append(dict(cell=i,seconds=time.perf_counter()-begin))
        results={}
        for name,path in namespace['RESULT_PATHS'].items():
            metrics=json.loads((path/'metrics.json').read_text())
            assert metrics['schema_version']==2 and metrics['purpose']=='cpu_smoke_only'
            assert metrics['test_beam_size']==4 and metrics['test_decoding']=='beam'
            assert all(k in metrics for k in ('test_greedy_bleu','test_greedy_chrf','test_split_sha256'))
            assert (path/'samples.txt').read_text().count('EN: ')==30
            assert (path/'train_log.csv').exists()
            results[name]=dict(path=str(path),family=metrics['model_family'],test_sources=metrics['test_sources'],
                              test_split_sha256=metrics['test_split_sha256'],training_status=metrics['training_status'])
        assert len({r['test_split_sha256'] for r in results.values()})==1
        weights=list(output.rglob('*.pt'))
        assert len(weights)==2 and all(p.name=='best.pt' for p in weights)
        assert not list(output.rglob('checkpoint-*'))
        assert len(list((output/'helsinki_full_notebook/checkpoints/best').glob('*.safetensors')))==1
        assert (output/'full_notebook_figures/model_scores.png').exists()
        assert (output/'full_notebook_figures/large_v4_loss.png').exists()
        report=dict(purpose='CPU execution only; random Marian fixture, not pretrained quality',
                    code_cells=len(durations),execution_seconds=sum(r['seconds'] for r in durations),
                    cells=durations,results=results,scratch_checkpoint_files=[str(p.relative_to(output)) for p in weights])
        (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        return report
    finally:
        sys.meta_path.remove(guard);os.chdir(previous)
        for k in ('EN_RU_NOTEBOOK_SMOKE','EN_RU_NOTEBOOK_OUTPUT','EN_RU_NOTEBOOK_RAW'):
            os.environ.pop(k,None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='/tmp/en-ru-notebook-smoke')
    validate(p.parse_args().output)
