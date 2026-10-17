"""Audit reachable Git blobs for checkpoint paths, large files and common secret formats.

This heuristic scan is not a guarantee that no secret exists. Only categories and
paths are printed, never matching credential contents. It does not rewrite history.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PATTERNS={
    'private_key':re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'github_token':re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{40,255})\b'),
    'huggingface_token':re.compile(rb'\bhf_[A-Za-z0-9]{30,255}\b'),
    'aws_access_key':re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'google_api_key':re.compile(rb'\bAIza[A-Za-z0-9_-]{35}\b'),
    'openai_key':re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{40,255}\b'),
    'credential_url':re.compile(rb'https?://[^\s/:@]{1,80}:[^\s/@]{8,200}@'),
}


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def audit():
    objects=git('rev-list','--objects','--all').decode().splitlines()
    names={line.split(' ',1)[0]:line.split(' ',1)[1] if ' ' in line else '' for line in objects}
    batch=subprocess.check_output(['git','cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)'],
                                  input=('\n'.join(names)+'\n').encode(),cwd=ROOT).decode().splitlines()
    large=[];suspicious=[];secrets=[];blobs=0;total=0
    for line in batch:
        sha,kind,size=line.split();size=int(size)
        if kind!='blob':continue
        blobs+=1;total+=size;name=names[sha];path=Path(name)
        if size>10*1024**2:large.append(dict(path=name,bytes=size,sha=sha))
        if path.suffix in ('.pt','.pth','.safetensors','.ckpt') or path.name=='.env' or path.name.startswith('.env.'):
            suspicious.append(dict(path=name,sha=sha))
        # The attributed corpus is scanned too; only exact high-confidence formats are matched.
        content=git('cat-file','blob',sha)
        for category,pattern in PATTERNS.items():
            if pattern.search(content):secrets.append(dict(path=name,sha=sha,category=category))
    unexpected_large=[x for x in large if x['path']!='data/raw/rus.txt']
    report=dict(scope='All reachable local Git refs (git rev-list --objects --all); regex formats only',
                blobs_scanned=blobs,total_blob_bytes=total,large_blobs_over_10_mib=large,
                unexpected_large_blobs=unexpected_large,checkpoint_or_environment_paths=suspicious,
                secret_pattern_findings=secrets,
                passed=not (unexpected_large or suspicious or secrets),
                limitations='Heuristics cannot prove absence of all secrets. Unreachable server objects and untracked local files are outside this audit.')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');a=p.parse_args();report=audit()
    serialized=json.dumps(report,indent=2)+'\n'
    if a.output:Path(a.output).write_text(serialized)
    print(serialized)
    raise SystemExit(0 if report['passed'] else 1)
