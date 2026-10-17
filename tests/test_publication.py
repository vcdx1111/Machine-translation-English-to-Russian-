import ast
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import pytest

from scripts.build_example_table import example_table
from scripts.compare_results import compare, result_status
from scripts.plot_results import plot_results


def test_recovered_helsinki_completion_does_not_need_training_log(tmp_path):
    metrics=dict(model_family='helsinki_marian',stage='fine_tuned',schema_version=2,
                 test_sources=12556,test_split_sha256='same',test_bleu=62.68,test_chrf=76.77,
                 test_decoding='beam',test_beam_size=4,test_greedy_bleu=56.41,test_greedy_chrf=74.52,
                 training_seconds=1901)
    path=tmp_path/'metrics.json';path.write_text(json.dumps(metrics))
    assert result_status(metrics)=='completed'
    assert all(r['status']=='completed' for r in compare([path]))
    assert not (tmp_path/'train_log.csv').exists()
    assert result_status(dict(metrics,training_status='diverged'))=='diverged'
    assert result_status(dict(metrics,model_family='scratch_transformer'))=='unreported'


def test_join_examples_by_source_without_fabricating_missing_outputs(tmp_path):
    summary={'runs':[{'name':'Fast','path':'fast'},{'name':'Large','path':'large'}]}
    s=tmp_path/'summary.json';s.write_text(json.dumps(summary))
    for run,text in [('fast','EN: Hi.\nREF: Привет.\nMODEL: Привет!\n'),
                     ('large','EN: Other.\nREF: Другой.\nMODEL: Другой!\n\nEN: Hi.\nREF: Привет.\nMODEL: Здравствуйте!\n')]:
        p=tmp_path/run/'samples.txt';p.parent.mkdir();p.write_text(text)
    table=example_table(tmp_path,s,['Hi.','Missing.'],include_common=False)
    assert '| Hi. | Привет! | Здравствуйте! |' in table
    assert '| Missing. | Exact output unavailable' in table


def test_plot_requires_real_loss_rows_and_records_provenance(tmp_path):
    csv=tmp_path/'actual.csv';csv.write_text('epoch,train_loss,val_loss\n1,6.64,5.28\n2,4.2,3.8\n')
    output=plot_results('docs/final_results.json',csv,tmp_path/'figures')
    for name in ('model_scores.png','large_v4_loss.png'):
        assert (output/name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert json.loads((output/'provenance.json').read_text())['loss_source']==str(csv)
    csv.write_text('epoch,train_loss,val_loss\n')
    with pytest.raises(ValueError,match='Empty loss CSV'):
        plot_results('docs/final_results.json',csv,tmp_path/'empty')


def test_full_notebook_embeds_current_sources_without_project_imports_or_outputs():
    from scripts.build_full_notebook import build, DEST
    n=nbformat.read(DEST,as_version=4);nbformat.validate(n)
    assert nbformat.writes(n)+'\n'==nbformat.writes(build())+'\n'
    flags={};cell_count=0
    for i,c in enumerate(n.cells):
        if c.cell_type!='code':continue
        cell_count+=1
        assert not c.outputs and c.execution_count is None and n.cells[i-1].cell_type=='markdown'
        tree=ast.parse(c.source)
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                assert not node.level
                assert not node.module or node.module.split('.')[0] not in ('src','scripts')
            if isinstance(node,ast.Import):
                assert all(a.name.split('.')[0] not in ('src','scripts') for a in node.names)
        for node in tree.body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                if node.targets[0].id.startswith('RUN_'):
                    flags[node.targets[0].id]=ast.literal_eval(node.value)
    assert cell_count>=15
    assert flags and not any(flags.values())


def test_full_notebook_cpu_pipeline(tmp_path):
    output=tmp_path/'standalone'
    proc=subprocess.run([sys.executable,'scripts/validate_full_notebook.py','--output',str(output)],
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300)
    assert proc.returncode==0,proc.stdout[-9000:]
    report=json.loads((output/'validation.json').read_text())
    assert set(report['results'])=={'fast','large','zero_shot','finetuned'}
    assert report['results']['finetuned']['family']=='marian_cpu_fixture'
    assert len(report['scratch_checkpoint_files'])==2
