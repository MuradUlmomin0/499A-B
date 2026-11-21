"""Verify FedMADE's saved CPM/state equations and paired clean experiment."""
from collections import OrderedDict
import csv
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from Support.data.latest_audit import source_hash
from Support.data.latest_pilot import _json
from Support.data.prepared import load_prepared_data
from Support.evaluation.metrics import evaluate_model
from Support.federated.cpm import class_probability_matrix
from Support.federated.fedmade import cluster_cpms,solve_cluster_weights,client_weights
from Support.federated_runner import _state_sha256
from Support.models.artifact import load_model_artifact
from Support.models.factory import create_model
from Support.Utils.fedmade_config import load_fedmade_config
from Support.Utils.parameters import clone_state,extract_update


def verify_run(path, *, started=None):
    path=Path(path);started=perf_counter() if started is None else started
    config=load_fedmade_config(path/'config.yaml');metadata=json.loads((path/'metadata.json').read_text())
    summary=json.loads((path/'run_summary.json').read_text());rounds=config['federated']['rounds']
    artifact=load_model_artifact(path/'model_artifact.pt');checkpoint=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
    metrics=json.loads((path/'validation_metrics.json').read_text())['metrics'];m=config['fedmade']
    expected={p['client_id']:p['sample_count'] for p in metadata['federated_inputs']['clients']}
    if summary['selected_round']!=rounds or checkpoint['round_id']!=rounds:raise ValueError('Incomplete rounds')
    if set(checkpoint['history'])!=set(expected) or any(set(h)!=set(range(1,rounds+1)) for h in checkpoint['history'].values()):raise ValueError('Incomplete update histories')
    final_hash=_state_sha256(artifact.model.state_dict())
    if final_hash!=_state_sha256(checkpoint['model_state']):raise ValueError('Artifact/checkpoint state mismatch')
    if final_hash==metadata['initialization']['model_state_sha256']:raise ValueError('Model did not change')
    auxiliary=load_prepared_data(m['auxiliary_manifest_path'],expected_role='auxiliary',expected_contract=artifact.contract)
    if source_hash(auxiliary.manifest.data_path)!=metadata['auxiliary']['data_sha256'] or source_hash(m['auxiliary_evidence_path'])!=metadata['auxiliary']['evidence_sha256']:
        raise ValueError('Auxiliary inputs changed during training')
    loader=DataLoader(auxiliary.dataset,batch_size=config['training']['batch_size'],shuffle=False,num_workers=0,
                      generator=torch.Generator().manual_seed(0))
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(config['experiment']['seed']).get_state())
        model=create_model(artifact.contract.model_config())
    previous_global=clone_state(model.state_dict())
    if _state_sha256(previous_global)!=metadata['initialization']['model_state_sha256']:
        raise ValueError('Initial state does not match the declared random seed/model')
    with (path/'aggregation_weights.csv').open() as stream:weight_rows=list(csv.DictReader(stream))
    with (path/'round_metrics.csv').open() as stream:round_rows=list(csv.DictReader(stream))
    if len(round_rows)!=rounds or any(r['evaluation_role']!='validation' for r in round_rows):raise ValueError('Wrong round evaluation')
    cluster_counts=[];zero_weights=[]
    for rnd in range(1,rounds+1):
        prefix=path/'fedmade_diagnostics'/f'round_{rnd:03d}'
        diagnostics=json.loads(prefix.with_suffix('.json').read_text())
        with np.load(prefix.with_suffix('.npz'),allow_pickle=False) as data:
            cpms=data['cpms'];ccpms=data['ccpms'];labels=data['labels'];alpha=data['alpha'];beta=data['beta'];ids=data['client_ids'].tolist()
            actual_counts=torch.bincount(auxiliary.dataset.tensors[1],minlength=artifact.contract.num_classes).numpy()
            if not np.array_equal(data['auxiliary_counts'],actual_counts):raise ValueError('Auxiliary class counts differ')
        states=torch.load(prefix.with_suffix('.pt'),map_location='cpu',weights_only=True)
        if ids!=sorted(expected) or set(states['client_states'])!=set(ids):raise ValueError('Missing diagnostic client')
        actual=[]
        for client in ids:
            update=extract_update(states['client_states'][client],previous_global)
            if _state_sha256(update)!=_state_sha256(checkpoint['history'][client][rnd]):
                raise ValueError('History update differs from local minus preceding global state')
            model.load_state_dict(states['client_states'][client],strict=True)
            actual.append(class_probability_matrix(model,loader,artifact.contract.num_classes))
        if not np.array_equal(np.stack(actual),cpms):raise ValueError('Saved CPM does not match local model probabilities')
        l,c=cluster_cpms(cpms,m['eps']);a,residual,kkt=solve_cluster_weights(c,m['maxiter']);b=client_weights(l,a)
        if (not np.array_equal(l,labels) or not np.allclose(c,ccpms,atol=1e-12,rtol=0)
                or not np.allclose(a,alpha,atol=1e-10,rtol=1e-10) or not np.allclose(b,beta,atol=1e-12,rtol=0)):
            raise ValueError('Clustering/CCPM/NNLS diagnostics differ')
        weights=dict(zip(ids,b.tolist()));rows=[r for r in weight_rows if int(r['round_id'])==rnd]
        if (diagnostics['client_ids']!=ids or diagnostics['cluster_labels']!=l.tolist()
                or diagnostics['cluster_count']!=len(a) or diagnostics['cluster_sizes']!=np.bincount(l).tolist()
                or diagnostics['auxiliary_class_counts']!=actual_counts.tolist()
                or not np.allclose(diagnostics['raw_alpha'],a,atol=1e-10,rtol=1e-10)
                or not np.allclose(diagnostics['normalized_cluster_mass'],a/a.sum(),atol=1e-12,rtol=0)
                or set(diagnostics['client_weights'])!=set(ids)
                or any(abs(diagnostics['client_weights'][c]-weights[c])>1e-12 for c in ids)
                or abs(diagnostics['projected_kkt_residual']-kkt)>1e-10):
            raise ValueError('JSON method diagnostics differ from actual CPM/NNLS equations')
        selected=json.loads(round_rows[rnd-1]['selected_clients'])
        if (len(rows)!=len(ids) or {r['client_id'] for r in rows}!=set(ids) or set(selected)!=set(ids)
                or states['client_order']!=selected or abs(sum(weights.values())-1)>1e-12):raise ValueError('Invalid selected client/weight coverage')
        for r in rows:
            client=r['client_id']
            if int(r['sample_count'])!=expected[client] or abs(float(r['weight'])-weights[client])>1e-12:raise ValueError('Logged method weight differs')
        weighted=OrderedDict()
        reference=states['aggregated_state']
        for key,value in reference.items():
            accumulator=torch.zeros_like(value,dtype=torch.float64)
            for client in states['client_order']:accumulator.add_(states['client_states'][client][key].to(torch.float64),alpha=weights[client])
            weighted[key]=accumulator.to(value.dtype)
        if _state_sha256(weighted)!=_state_sha256(reference):raise ValueError('Weighted-state equation differs')
        if rnd==rounds and _state_sha256(reference)!=final_hash:raise ValueError('Last aggregated state differs from artifact')
        previous_global=reference
        if diagnostics['round_id']!=rnd or abs(diagnostics['objective_frobenius']-residual)>1e-10:raise ValueError('Diagnostic objective/round differs')
        cluster_counts.append(len(alpha));zero_weights.append(int(np.count_nonzero(beta==0)))
    bundle_path=Path(config['dataset']['bundle_path']);bundle=json.loads(bundle_path.read_text())
    validation=load_prepared_data((bundle_path.parent/bundle['validation']['manifest_path']).resolve(),expected_role='validation',expected_contract=artifact.contract)
    reloaded=evaluate_model(artifact.model,DataLoader(validation.dataset,batch_size=config['training']['batch_size'],shuffle=False),'cpu',artifact.contract.num_classes)
    if reloaded!=metrics:raise ValueError('Saved-model validation does not reproduce logged metrics')
    paired=m['reference_run_path']
    if paired is not None:
        reference=json.loads((Path(paired)/'verification.json').read_text())
        if reference['initial_state_sha256']!=metadata['initialization']['model_state_sha256']:raise ValueError('Unmatched initial states')
        with (Path(paired)/'round_metrics.csv').open() as stream:baseline_rows=list(csv.DictReader(stream))
        if [(r['round_id'],r['selected_clients']) for r in round_rows]!=[(r['round_id'],r['selected_clients']) for r in baseline_rows]:raise ValueError('Unmatched client-selection schedule')
    verification={'initial_state_sha256':metadata['initialization']['model_state_sha256'],'final_state_sha256':final_hash,
        'model_artifact_sha256':artifact.artifact_sha256,'rounds':rounds,'train_rows':sum(expected.values()),'client_rows':expected,
        'validation_rows':len(validation.dataset),'auxiliary_rows':len(auxiliary.dataset),'history_verified':True,
        'cpm_and_equations_verified':True,'aggregation_verified':True,'reloaded_validation_exact':True,
        'history_update_reference_verified':True,
        'artifact_matches_checkpoint':True,'model_changed':True,'paired_controls_verified':paired is not None,
        'cluster_counts_by_round':cluster_counts,'exact_zero_client_weights_by_round':zero_weights,
        'testing_only':artifact.testing_only,'test_scored':False,'adapted_not_exact_reproduction':True,
        'end_to_end_with_verification_seconds':perf_counter()-started,'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    _json(path/'verification.json',verification);_json(path/'reloaded_validation_metrics.json',reloaded)
    with (path/'validation_per_class.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['class_id','class_name','support','precision','recall','f1'])
        for row in metrics['per_class']:writer.writerow([row['class_id'],artifact.contract.class_names[row['class_id']],*[row[k] for k in ['support','precision','recall','f1']]])
    return verification
