"""Independently recompute both CP6 components, original-state blend and labels."""
from Support.Utils.runtime_paths import load_json as runtime_json, load_json_file as runtime_json_file, load_yaml as runtime_yaml, safe_load_yaml as runtime_safe_yaml
from collections import OrderedDict
import csv
import json
from pathlib import Path
import random
import resource
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits

from Support.data.latest_audit import source_hash
from Support.data.latest_pilot import _json
from Support.evaluation.metrics import evaluate_model
from Support.federated.cpm import class_probability_matrix
from Support.federated.fedmade import cluster_cpms,solve_cluster_weights,client_weights
from Support.federated_runner import _state_sha256
from Support.models.artifact import load_model_artifact
from Support.models.factory import create_model
from Support.temporal_verification import audit_scores,_equal_numbers
from Support.Utils.parameters import clone_state,extract_update,flatten_update
from Support.federated.types import ClientResult
from Support.Utils.fusion_config import load_fusion_config


def csv_rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def verify_label_events(path,config,bundle,selected,round_rows):
    """Recompute expected labels from authenticated originals, not wrapper counters."""
    path=Path(path);a=config['attack'];base=config['_baseline'];epochs=base['training']['local_epochs']
    source={p.client_id:p.data.dataset.tensors[1].numpy() for p in bundle.clients}
    events=runtime_json((path/'attack_ground_truth.json').read_text())
    if events['selected_malicious']!=selected or events['attack']!=a:raise ValueError('Attack ground truth differs from frozen selection/settings')
    expected_keys={(int(r['round_id']),c) for r in round_rows for c in runtime_json(r['selected_clients'])}
    event_keys=[(e['round_id'],e['client_id']) for e in events['events']]
    if len(set(event_keys))!=len(event_keys) or set(event_keys)!=expected_keys:raise ValueError('Missing/duplicate attack events')
    audit_files=set();total_observations=0
    for event in events['events']:
        rnd,c=event['round_id'],event['client_id'];labels=source[c]
        active=bool(a['enabled'] and c in selected and a['start_round']<=rnd<=a['end_round'])
        count=int(np.count_nonzero(labels==a['target_class']));file=f'round_{rnd:03d}_{c}.npz' if active else None
        expected=dict(round_id=rnd,client_id=c,selected_malicious=c in selected,active_attack=active,sample_count=len(labels),local_epochs=epochs,
            target_rows=count,flipped_unique_rows=count if active else 0,flipped_observations=count*epochs if active else 0,audit_file=file)
        if event!=expected:raise ValueError('Label-change event differs from actual source/support/schedule')
        if active:
            if Path(file).name!=file:raise ValueError('Unsafe audit filename')
            audit_files.add(file)
            with np.load(path/'label_audits'/file,allow_pickle=False) as record:
                changed=labels.copy();changed[labels==a['target_class']]=a['destination_class']
                if (set(record.files)!={'observed_labels','access_counts'} or record['observed_labels'].dtype!=np.int64
                        or record['access_counts'].dtype!=np.int32 or record['access_counts'].shape!=labels.shape
                        or not np.array_equal(record['observed_labels'],changed) or not np.all(record['access_counts']==epochs)):
                    raise ValueError('Delivered labels/record visits differ from approved poisoning')
            total_observations+=count*epochs
    if {p.name for p in (path/'label_audits').glob('*.npz')}!=audit_files:raise ValueError('Extra or missing label audits')
    return events,total_observations


def verify_fedmade_round(path,rnd,results,auxiliary,model,base):
    prefix=path/'fedmade_diagnostics'/f'round_{rnd:03d}';ids=sorted(r.client_id for r in results)
    rows=sorted(results,key=lambda r:r.client_id);loader=DataLoader(auxiliary.dataset,batch_size=base['training']['batch_size'],shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(0))
    actual=[]
    for r in rows:
        model.load_state_dict(r.state,strict=True);actual.append(class_probability_matrix(model,loader,model.num_classes))
    cpms=np.stack(actual);labels,ccpms=cluster_cpms(cpms,base['fedmade']['eps']);alpha,residual,kkt=solve_cluster_weights(ccpms,base['fedmade']['maxiter']);beta=client_weights(labels,alpha)
    with np.load(prefix.with_suffix('.npz'),allow_pickle=False) as arrays:
        checks=dict(client_ids=np.array(ids),cpms=cpms,ccpms=ccpms,labels=labels,alpha=alpha,beta=beta,
            auxiliary_counts=torch.bincount(auxiliary.dataset.tensors[1],minlength=model.num_classes).numpy())
        if set(arrays.files)!=set(checks) or any(not np.array_equal(arrays[k],v) for k,v in checks.items()):raise ValueError('FedMADE CPM/cluster/NNLS evidence differs')
    d=runtime_json(prefix.with_suffix('.json').read_text());weights=dict(zip(ids,beta.tolist()))
    if (d['round_id']!=rnd or d['client_ids']!=ids or d['cluster_labels']!=labels.tolist() or d['cluster_count']!=len(alpha)
            or d['cluster_sizes']!=np.bincount(labels).tolist() or d['auxiliary_class_counts']!=checks['auxiliary_counts'].tolist()
            or not np.array_equal(d['raw_alpha'],alpha) or not np.array_equal(d['normalized_cluster_mass'],alpha/alpha.sum())
            or d['client_weights']!=weights or d['objective_frobenius']!=residual or d['projected_kkt_residual']!=kkt):raise ValueError('FedMADE JSON equations differ')
    return weights


@threadpool_limits.wrap(limits=1)
def verify_run(path,*,started=None):
    torch.set_num_threads(1);path=Path(path);started=perf_counter() if started is None else started
    config=load_fusion_config(path/'config.yaml');base=config['_baseline'];method=config['method'];a=config['attack']
    from Support.fusion_runner import load_inputs,paired_reference
    bundle,auxiliary=load_inputs(config);reference=paired_reference(config,bundle)
    meta=runtime_json((path/'metadata.json').read_text());summary=runtime_json((path/'run_summary.json').read_text())
    if (meta['workflow']!='prepared_fusion' or meta['method']!=method or meta['federated_inputs']!=bundle.source_metadata()
            or meta['lambda_value']!=config['lambda_value'] or meta['test_scored'] or summary['test_scored'] or meta['baseline_config_sha256']!=source_hash(config['baseline_config_path'])):raise ValueError('Changed source/configuration or invalid workflow scope')
    if reference is not None and meta['paired_clean_config_sha256']!=source_hash(Path(config['reference_clean_run_path'])/'config.yaml'):raise ValueError('Changed clean reference configuration')
    if meta['temporal_config_sha256']!=source_hash(config['temporal_config_path']):raise ValueError('Temporal configuration changed')
    if auxiliary is not None:
        for field in ['manifest','data','evidence']:
            if source_hash(meta['auxiliary'][field+'_path'])!=meta['auxiliary'][field+'_sha256']:raise ValueError('Auxiliary data changed')
    artifact=load_model_artifact(path/'model_artifact.pt');bundle.contract.assert_compatible(artifact.contract)
    if artifact.testing_only!=bundle.testing_only:raise ValueError('Testing-only status mismatch')
    checkpoint=torch.load(path/'checkpoint.pt',weights_only=True,map_location='cpu');rounds=base['federated']['rounds']
    if checkpoint['round_id']!=rounds or summary['selected_round']!=rounds:raise ValueError('Incomplete model training')
    if source_hash(path/'model_artifact.pt')!=summary['model_artifact_sha256']:raise ValueError('Exported artifact hash differs')
    expected_diagnostics={f'round_{r:03d}' for r in range(1,rounds+1)}
    for directory in ['fusion_diagnostics','fedmade_diagnostics','temporal_diagnostics']:
        if {p.stem for p in (path/directory).glob('*.json')}!=expected_diagnostics:raise ValueError('Incomplete/extra component round diagnostics')
    final_hash=_state_sha256(artifact.model.state_dict())
    if final_hash!=_state_sha256(checkpoint['model_state']):raise ValueError('Artifact and inspection model differ')
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(base['experiment']['seed']).get_state());scratch=create_model(bundle.contract.model_config())
    previous=clone_state(scratch.state_dict());initial_hash=_state_sha256(previous)
    if initial_hash!=meta['initialization']['model_state_sha256'] or (reference is not None and initial_hash!=reference['initialization']['model_state_sha256']):raise ValueError('Unmatched initialization')
    ids=sorted(p.client_id for p in bundle.clients);expected={p.client_id:len(p.data.dataset) for p in bundle.clients}
    selected=random.Random(base['experiment']['seed']+5005).sample(ids,a['malicious_clients'])
    round_rows=csv_rows(path/'round_metrics.csv');weight_rows=csv_rows(path/'aggregation_weights.csv')
    if len(round_rows)!=rounds or [int(r['round_id']) for r in round_rows]!=list(range(1,rounds+1)) or any(r['evaluation_role']!='validation' for r in round_rows):raise ValueError('Invalid round evaluation coverage')
    events,total_flips=verify_label_events(path,config,bundle,selected,round_rows)
    projection=np.random.default_rng(1729).normal(0,1/np.sqrt(5),size=(sum(v.numel() for v in previous.values()),5)) if method=='fusion' else None
    mixture=None;observations=[];score_history=[];prior_md={};observed={c:set() for c in ids};logged_weights=[]
    for rnd,row in enumerate(round_rows,1):
        order=runtime_json(row['selected_clients']);expected_order=random.Random(base['experiment']['seed']+rnd).sample(ids,max(1,__import__('math').ceil(base['federated']['participation_rate']*len(ids))))
        if order!=expected_order:raise ValueError('Unmatched client-selection schedule')
        states=torch.load(path/'aggregation_evidence'/f'round_{rnd:03d}.pt',weights_only=True,map_location='cpu')
        if states['client_order']!=order or set(states['client_states'])!=set(order):raise ValueError('Invalid local model coverage')
        results=[]
        for c in order:
            update=extract_update(states['client_states'][c],previous)
            if c not in checkpoint['history'] or rnd not in checkpoint['history'][c] or _state_sha256(update)!=_state_sha256(checkpoint['history'][c][rnd]):raise ValueError('Update history differs from actual local-minus-global state')
            observed[c].add(rnd);results.append(ClientResult(c,rnd,expected[c],states['client_states'][c],update,{}))
        utility=verify_fedmade_round(path,rnd,results,auxiliary,scratch,base)
        if method=='fusion':
            calculated,mixture,z,prior_md=audit_scores(results,rnd,projection,mixture,observations,score_history,prior_md)
            prefix=path/'temporal_diagnostics'/f'round_{rnd:03d}'
            if not _equal_numbers(runtime_json(prefix.with_suffix('.json').read_text()),calculated):raise ValueError('Temporal scoring equations differ')
            if not np.array_equal(np.load(path/'temporal_diagnostics/projection.npy',allow_pickle=False),projection):raise ValueError('Temporal projection differs')
            with np.load(prefix.with_suffix('.npz'),allow_pickle=False) as arrays:
                checks={'projected':z,**(mixture or {})}
                if set(arrays.files)!=set(checks) or any(arrays[k].shape!=v.shape or not np.allclose(arrays[k],v,atol=1e-12,rtol=1e-12) for k,v in checks.items()):raise ValueError('Temporal mixture/projection evidence differs')
            observations=[(t,v) for t,v in observations if t>=rnd-4]+[(rnd,z)];score_history=[(t,v) for t,v in score_history if t>=rnd-5]+[(rnd,calculated['clients'])]
            temporal={r['client_id']:r['weight'] for r in calculated['clients']}
            lam=config['lambda_value'];weights={c:lam*temporal[c]+(1-lam)*utility[c] for c in order}
            fusion=runtime_json((path/'fusion_diagnostics'/f'round_{rnd:03d}.json').read_text())
            expected_fusion=dict(round_id=rnd,lambda_value=lam,temporal_weights=temporal,class_utility_weights=utility,weights=weights)
            if not _equal_numbers(fusion,expected_fusion):raise ValueError('Fusion equations differ')
        if set(states['weights'])!=set(weights) or any(not np.isfinite(states['weights'][c]) or abs(states['weights'][c]-weights[c])>1e-12 for c in weights):raise ValueError('Saved strategy weights differ from equations')
        rows=[r for r in weight_rows if int(r['round_id'])==rnd]
        if len(rows)!=len(order) or {r['client_id'] for r in rows}!=set(order):raise ValueError('Missing client weight rows')
        for r in rows:
            c=r['client_id']
            if int(r['sample_count'])!=expected[c] or not np.isfinite(float(r['weight'])) or abs(float(r['weight'])-weights[c])>1e-12:raise ValueError('Logged weight differs')
        if not np.isclose(sum(weights.values()),1,rtol=0,atol=1e-12) or any(w<0 for w in weights.values()):raise ValueError('Invalid normalized weights')
        weighted=OrderedDict()
        for key,value in previous.items():
            acc=torch.zeros_like(value,dtype=torch.float64)
            for c in order:acc.add_(states['client_states'][c][key].to(torch.float64),alpha=weights[c])
            weighted[key]=acc.to(value.dtype)
        if _state_sha256(weighted)!=_state_sha256(states['aggregated_state']):raise ValueError('Weighted state equation differs')
        for directory,component_weights in [('fedmade_diagnostics',utility),('temporal_diagnostics',temporal)]:
            duplicate=torch.load(path/directory/f'round_{rnd:03d}.pt',weights_only=True,map_location='cpu')
            component_state=OrderedDict()
            for key,value in previous.items():
                acc=torch.zeros_like(value,dtype=torch.float64)
                for c in order:acc.add_(states['client_states'][c][key].to(torch.float64),alpha=component_weights[c])
                component_state[key]=acc.to(value.dtype)
            if (duplicate['client_order']!=order or _state_sha256(duplicate['aggregated_state'])!=_state_sha256(component_state)
                    or set(duplicate['client_states'])!=set(order) or any(_state_sha256(duplicate['client_states'][c])!=_state_sha256(states['client_states'][c]) for c in order)):raise ValueError('Companion strategy state evidence differs')
        previous=weighted;logged_weights.extend(rows)
    if len(weight_rows)!=len(logged_weights) or set(checkpoint['history'])!={c for c in ids if observed[c]} or any(set(checkpoint['history'][c])!=observed[c] for c in checkpoint['history']):raise ValueError('Extra histories/weights')
    if _state_sha256(previous)!=final_hash or initial_hash==final_hash:raise ValueError('Final model state is incorrect or unchanged')
    validation=bundle.validation.data.dataset;metrics=runtime_json((path/'validation_metrics.json').read_text())['metrics']
    reloaded=evaluate_model(artifact.model,DataLoader(validation,batch_size=base['training']['batch_size'],shuffle=False),'cpu',bundle.contract.num_classes)
    if reloaded!=metrics:raise ValueError('Saved-model validation differs')
    zero_match=None
    if reference is not None:
        clean=Path(config['reference_clean_run_path']);reference_rounds=csv_rows(clean/'round_metrics.csv')
        if [(r['round_id'],r['selected_clients']) for r in round_rows]!=[(r['round_id'],r['selected_clients']) for r in reference_rounds]:raise ValueError('Paired selection changed')
    endpoint=config['endpoint_reference_run_path']
    if endpoint is not None:
        endpoint=Path(endpoint);old=runtime_json((endpoint/'verification.json').read_text())
        old_metrics=runtime_json((endpoint/'validation_metrics.json').read_text())['metrics']
        if final_hash!=old['final_state_sha256'] or metrics!=old_metrics:raise ValueError('Fusion endpoint differs from frozen reference state/metrics')
        left={(r['round_id'],r['client_id']):(r['sample_count'],float(r['weight'])) for r in weight_rows}
        right={(r['round_id'],r['client_id']):(r['sample_count'],float(r['weight'])) for r in csv_rows(endpoint/'aggregation_weights.csv')}
        if left!=right:raise ValueError('Fusion endpoint weights differ from frozen reference')
        zero_match=True
    counts={p.client_id:torch.bincount(p.data.dataset.tensors[1],minlength=bundle.contract.num_classes).tolist() for p in bundle.clients}
    _json(path/'client_class_counts.json',counts)
    with (path/'validation_per_class.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['class_id','class_name','support','precision','recall','f1'])
        for r in metrics['per_class']:w.writerow([r['class_id'],bundle.contract.class_names[r['class_id']],*[r[k] for k in ['support','precision','recall','f1']]])
    target=a['target_class'];destination=a['destination_class'];support=metrics['per_class'][target]['support']
    effect=dict(target_class=target,target_class_name=bundle.contract.class_names[target],destination_class=destination,
        target_support=support,target_recall=metrics['per_class'][target]['recall'],target_f1=metrics['per_class'][target]['f1'],
        target_to_benign_count=metrics['confusion_matrix'][target][destination],target_to_benign_rate=metrics['confusion_matrix'][target][destination]/support if support else None)
    _json(path/'target_metrics.json',effect)
    verification=dict(method=method,both_components_verified=True,lambda_value=config['lambda_value'],endpoint_matches_reference=zero_match,initial_state_sha256=initial_hash,final_state_sha256=final_hash,rounds=rounds,
        train_rows=sum(expected.values()),client_rows=expected,validation_rows=len(validation),testing_only=bundle.testing_only,test_scored=False,
        delivered_labels_verified=True,original_sources_verified=True,aggregation_verified=True,history_verified=True,
        reloaded_validation_exact=True,paired_controls_verified=reference is not None,zero_attack_matches_clean=zero_match,
        selected_malicious=selected,flipped_unique_rows_per_active_round=sum(counts[c][target] for c in selected),total_flipped_training_observations=total_flips,
        end_to_end_with_verification_seconds=perf_counter()-started,peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    _json(path/'verification.json',verification)
    return verification
