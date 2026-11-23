"""Artificial adapter fixtures; their scores are software checks only."""
import csv
import gc
import io
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import weakref

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from Support.data.contract import ModelDataContract
from Support.models.factory import create_model
from Support.models.artifact import save_model_artifact,load_model_artifact
from Support.data.prepared import load_prepared_data
from Support.evaluation.predict import predict_prepared_data
from Support.frontend import service

NAMES={'fedavg':'cp2_iid_seed42','fedmade':'cp3_iid_seed42','temporal':'cp4_iid_seed42','fusion':'cp6_fused_clean_iid_seed42'}


@pytest.fixture
def bundle(tmp_path):
    root=tmp_path/'bundle';configs=root/'Support/configs';configs.mkdir(parents=True)
    data=root/'data/processed/demo_predict';data.mkdir(parents=True)
    contract=ModelDataContract('mlp',2,2,(2,),('a','b'),('BenignTraffic','Attack'),None)
    x=np.tile(np.array([[2.,0.],[0.,2.]],dtype=np.float32),(10,1))
    np.savez(data/'features.npz',features=x,feature_names=np.array(contract.feature_names),class_names=np.array(contract.class_names))
    (data/'manifest.json').write_text(json.dumps(dict(schema_version=1,dataset_name='synthetic-ui-fixture',partition_name='predict',role='predict',client_id=None,data_format='npz',data_path='features.npz',preprocessing_path=None,contract=contract.to_dict(),testing_only=True)))
    paths={}
    for key,name in NAMES.items():
        model=create_model(contract.model_config())
        with torch.no_grad():
            model.network[0].weight.copy_(torch.eye(2));model.network[0].bias.zero_()
            model.network[2].weight.copy_(torch.eye(2)*2);model.network[2].bias.zero_()
        path=root/'Support/results'/key/'model_artifact.pt';path.parent.mkdir(parents=True)
        save_model_artifact(model,path,contract=contract,model_name=name,training_completed=False,testing_only=True);paths[key]=path
        config=dict(schema_version=1,mode='evaluate',experiment=dict(name='fixture',seed=42,device='cpu',deterministic=True,output_dir='../results'),model=dict(artifact_path=str(path)),dataset=dict(manifest_path=str(data/'manifest.json'),expected_role='predict'),inference=dict(batch_size=4))
        (configs/f'demo_{key}.yaml').write_text(yaml.safe_dump(config))
        if key=='fedavg':
            config['mode']='predict';(configs/'demo_predict.yaml').write_text(yaml.safe_dump(config))
        with (path.parent/'round_metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['round_id','evaluation_role','accuracy','macro_f1']);w.writeheader()
            for r in range(1,21):w.writerow(dict(round_id=r,evaluation_role='validation',accuracy=.6+r/100,macro_f1=.4+r/100))
    comparison=root/'Support/results/cp7_v1/analysis/comparison.json';comparison.parent.mkdir(parents=True)
    rows=[dict(method=m,condition=c,n_seeds=3,clean_accuracy_mean=.8,attacked_accuracy_mean=.79,clean_macro_f1_mean=.5,attacked_macro_f1_mean=.48,recall_degradation_mean=.02) for c in ['iid','dirichlet'] for m in NAMES]
    comparison.write_text(json.dumps(dict(evaluation_role='test',test_scored=True,training_performed=False,case_count=48,summary=rows)))
    return root,paths,contract


def test_catalog_and_summary_derive_real_contract(bundle):
    root,_,_=bundle;api=service.PredictionService(root)
    assert {c.key for c in api.catalog() if c.available}==set(NAMES)
    assert api.demo_summary()['records']==20
    assert api.demo_summary()['features']==2 and api.demo_summary()['classes']==2


@pytest.mark.parametrize('key',list(NAMES))
def test_predictions_match_existing_api_and_never_train(bundle,key,monkeypatch):
    root,paths,_=bundle
    def forbidden(*a,**k):raise AssertionError('Frontend must not train')
    monkeypatch.setattr(torch.optim.Optimizer,'__init__',forbidden);monkeypatch.setattr(torch.Tensor,'backward',forbidden)
    view=service.PredictionService(root).predict(key)
    artifact=load_model_artifact(paths[key]);data=load_prepared_data(root/'data/processed/demo_predict/manifest.json',expected_role='predict',expected_contract=artifact.contract)
    expected=predict_prepared_data(artifact.model,data,model_contract=artifact.contract,batch_size=4)
    assert view.frame['class_id'].tolist()==expected.class_ids.tolist()
    assert view.frame['class_name'].tolist()==list(expected.predicted_labels)
    assert view.model_key==key and view.model_name==NAMES[key] and view.testing_only
    rows=list(csv.DictReader(io.StringIO(view.csv_bytes().decode())))
    assert len(rows)==20 and all(row['model_method']==key for row in rows)
    assert {'probability_0','probability_1'}<=set(rows[0])
    assert 'accuracy' not in view.frame and 'confidence' not in view.frame


def test_only_one_cached_model_survives_switch(bundle,monkeypatch):
    root,_,_=bundle;loaded=[];original=service.load_model_artifact
    def capture(path):
        result=original(path);loaded.append(weakref.ref(result.model));return result
    monkeypatch.setattr(service,'load_model_artifact',capture)
    api=service.PredictionService(root);first=api.predict('fedavg');repeat=api.predict('fedavg')
    assert len(loaded)==1;pd.testing.assert_frame_equal(first.frame,repeat.frame)
    api.predict('fusion');gc.collect()
    assert len(loaded)==2 and loaded[0]() is None and loaded[1]() is not None


def test_concurrent_requests_keep_their_model_identity(bundle):
    root,_,_=bundle;api=service.PredictionService(root);keys=['fedavg','fusion']*3
    with ThreadPoolExecutor(max_workers=2) as executor:views=list(executor.map(api.predict,keys))
    assert [v.model_key for v in views]==keys
    assert [v.model_name for v in views]==[NAMES[k] for k in keys]


def test_minimal_bundle_fedavg_fallback_and_missing_choices(bundle):
    root,paths,_=bundle
    for key in NAMES:
        (root/f'Support/configs/demo_{key}.yaml').unlink()
    for key in ['fedmade','temporal','fusion']:paths[key].unlink()
    api=service.PredictionService(root)
    assert [c.key for c in api.catalog() if c.available]==['fedavg']
    assert len(api.predict('fedavg').frame)==20
    with pytest.raises(service.UnavailableError):api.predict('fusion')


@pytest.mark.parametrize('damage',['missing','corrupt','wrong_contract','wrong_identity'])
def test_bad_model_cannot_produce_mislabelled_output(bundle,damage):
    root,paths,_=bundle;p=paths['fedavg']
    if damage=='missing':p.unlink()
    elif damage=='corrupt':p.write_bytes(b'broken')
    else:
        value=torch.load(p,weights_only=True)
        if damage=='wrong_contract':value['contract']['class_names'].reverse()
        else:value['model_name']='a_different_experiment'
        torch.save(value,p)
    with pytest.raises(service.FrontendError):service.PredictionService(root).predict('fedavg')


def test_demo_cannot_switch_to_full_validation(bundle):
    root,_,_=bundle;p=root/'Support/configs/demo_predict.yaml';v=yaml.safe_load(p.read_text());v['dataset']['expected_role']='validation';p.write_text(yaml.safe_dump(v))
    with pytest.raises(service.FrontendError):service.PredictionService(root).predict('fedavg')


def test_more_than_twenty_rows_is_rejected(bundle):
    root,_,contract=bundle;p=root/'data/processed/demo_predict/features.npz'
    np.savez(p,features=np.zeros((21,2),np.float32),feature_names=np.array(contract.feature_names),class_names=np.array(contract.class_names))
    with pytest.raises(service.FrontendError):service.PredictionService(root).predict('fedavg')


def test_saved_results_use_test_role_and_correct_units(bundle):
    root,_,_=bundle;frame=service.load_research_results(root,'iid')
    assert len(frame)==4
    assert frame.iloc[0]['Clean accuracy (%)']==80
    assert frame.iloc[0]['Clean macro F1']==.5
    assert frame.iloc[0]['Target recall drop (pp)']==2


def test_negative_recall_drop_is_preserved(bundle):
    root,_,_=bundle;p=root/'Support/results/cp7_v1/analysis/comparison.json';v=json.loads(p.read_text());v['summary'][0]['recall_degradation_mean']=-.03;p.write_text(json.dumps(v))
    assert service.load_research_results(root,'iid').iloc[0]['Target recall drop (pp)']==-3


@pytest.mark.parametrize('damage',['role','seeds','duplicate','nonfinite'])
def test_invalid_research_summary_is_not_presented_as_results(bundle,damage):
    root,_,_=bundle;p=root/'Support/results/cp7_v1/analysis/comparison.json';v=json.loads(p.read_text())
    if damage=='role':v['evaluation_role']='validation'
    if damage=='seeds':v['summary'][0]['n_seeds']=1
    if damage=='duplicate':v['summary'][1]=v['summary'][0]
    if damage=='nonfinite':v['summary'][0]['clean_accuracy_mean']=float('nan')
    p.write_text(json.dumps(v))
    with pytest.raises(service.FrontendError):service.load_research_results(root,'iid')


def test_optional_results_and_history_can_be_missing(bundle):
    root,paths,_=bundle;(root/'Support/results/cp7_v1/analysis/comparison.json').unlink();(paths['fedavg'].parent/'round_metrics.csv').unlink()
    with pytest.raises(service.UnavailableError):service.load_research_results(root)
    with pytest.raises(service.UnavailableError):service.load_training_history('fedavg',root)
    assert len(service.PredictionService(root).predict('fedavg').frame)==20


def test_history_keeps_validation_values_and_twenty_rounds(bundle):
    root,_,_=bundle;frame=service.load_training_history('fedavg',root)
    assert frame['round_id'].tolist()==list(range(1,21))
    assert frame.iloc[0]['accuracy']==.61 and frame.iloc[0]['macro_f1']==pytest.approx(.41)


@pytest.mark.parametrize('damage',['role','missing_round','invalid_metric'])
def test_invalid_history_is_rejected(bundle,damage):
    root,paths,_=bundle;p=paths['fedavg'].parent/'round_metrics.csv';frame=pd.read_csv(p)
    if damage=='role':frame.loc[0,'evaluation_role']='test'
    if damage=='missing_round':frame=frame.iloc[1:]
    if damage=='invalid_metric':frame.loc[0,'accuracy']=2
    frame.to_csv(p,index=False)
    with pytest.raises(service.FrontendError):service.load_training_history('fedavg',root)
