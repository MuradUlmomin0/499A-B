import csv
import json
import pytest
import torch
from support.utils.experiment_logger import ExperimentLogger
from support.federated.types import ClientResult


def test_unique_run_and_finalization(tmp_path,config):
    a=ExperimentLogger(tmp_path,'test',config,{'testing_only':True})
    b=ExperimentLogger(tmp_path,'test',config,{'testing_only':True})
    assert a.path!=b.path
    a.log_round({'round_id':1,'selected_clients':['0']},[ClientResult('0',1,2,{}, {},{'train_loss':1.})],{'0':1.})
    a.finish({'accuracy':0.5},{'model_state':{'w':torch.tensor([1.])}})
    final=json.loads((a.path/'final_metrics.json').read_text())
    assert final['testing_only'] is True
    assert len(list(csv.DictReader((a.path/'aggregation_weights.csv').open())))==1
    saved=torch.load(a.path/'checkpoint.pt',weights_only=True)
    assert saved['testing_only'] is True
    assert saved['model_state']['w'].item()==1
    with pytest.raises(ValueError):
        a.finish({}, {})
