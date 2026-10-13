import csv
import json
import subprocess
import sys
from pathlib import Path
import torch
from main import run_dummy_test


def test_complete_reproducible_run(tmp_path,config):
    config['experiment']['output_dir']=str(tmp_path)
    first,metrics=run_dummy_test(config)
    second,repeat=run_dummy_test(config)
    assert first!=second and metrics==repeat
    expected={'config.yaml','metadata.json','round_metrics.csv','client_metrics.csv','aggregation_weights.csv','final_metrics.json','checkpoint.pt'}
    assert {p.name for p in first.iterdir()}==expected
    metadata=json.loads((first/'metadata.json').read_text())
    assert metadata['testing_only'] is True
    assert metadata['seed']==42
    assert metadata['git_commit']
    weights=list(csv.DictReader((first/'aggregation_weights.csv').open()))
    assert len(weights)==9
    for round_id in ('1','2','3'):
        rows=[r for r in weights if r['round_id']==round_id]
        assert abs(sum(float(r['weight']) for r in rows)-1)<1e-12
        for row in rows:
            assert abs(float(row['weight'])-int(row['sample_count'])/72)<1e-12
    a=torch.load(first/'checkpoint.pt',weights_only=True)
    b=torch.load(second/'checkpoint.pt',weights_only=True)
    assert a['round_id']==3 and a['resume_supported'] is False
    assert all(torch.equal(a['model_state'][k],b['model_state'][k]) for k in a['model_state'])
    assert sum(len(v) for v in a['history'].values())==9


def test_cli_requires_testing_opt_in():
    root=Path(__file__).resolve().parents[2]
    result=subprocess.run([sys.executable,str(root/'main.py')],capture_output=True,text=True)
    assert result.returncode!=0
    assert '--dummy-test' in result.stderr
