"""Replay temporal evidence and independently check saved weighted models."""
from collections import OrderedDict
import csv
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits
from Support.data.latest_audit import source_hash

from Support.data.latest_pilot import _json
from Support.data.prepared import load_prepared_data
from Support.evaluation.metrics import evaluate_model
from Support.federated.types import ClientResult
from Support.federated_runner import _state_sha256
from Support.models.artifact import load_model_artifact
from Support.models.factory import create_model
from Support.Utils.parameters import clone_state,extract_update,flatten_update
from Support.Utils.temporal_config import load_temporal_config


def _write_csv(path,rows):
    with Path(path).open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def penalty_summary(rows,counts,class_names):
    """All CP4 clients are honest. Rates count client-rounds, not traffic rows."""
    scored=[r for r in rows if r['phase']=='scored']
    def rates(subset):
        n=len(subset);rejected=sum(r['decision']=='reject' for r in subset);down=sum(r['decision']=='downweight' for r in subset)
        return dict(scored_client_rounds=n,rejected_client_rounds=rejected,downweighted_client_rounds=down,
            rejection_rate=rejected/n if n else None,downweight_rate=down/n if n else None,
            trust_penalty_rate=(rejected+down)/n if n else None,
            normalized_weight_below_fedavg=sum(r['weight']<r['fedavg_weight']-1e-15 for r in subset),
            mean_trust=float(np.mean([r['trust'] for r in subset])) if n else None)
    totals=np.sum(list(counts.values()),axis=0);minority=np.argsort(totals,kind='stable')[:min(8,len(totals))].tolist()
    by_client={c:{**rates([r for r in scored if r['client_id']==c]),'class_counts':counts[c]} for c in counts}
    by_class=[]
    for i,name in enumerate(class_names):
        clients=[c for c in counts if counts[c][i]>0]
        by_class.append(dict(class_id=i,class_name=name,training_rows=int(totals[i]),minority_lowest_eight=i in minority,
            carrying_clients=clients,**rates([r for r in scored if r['client_id'] in clients])))
    return dict(all_clients_honest=True,test_scored=False,rate_unit='scored client-round; warm-up excluded',
        total_client_rounds=len(rows),warmup_client_rounds=len(rows)-len(scored),overall=rates(scored),by_client=by_client,by_class=by_class,
        limitation='Class-bearing subsets overlap; these rates are not independent and do not establish attack detection.')


def audit_scores(results,rnd,projection,previous_mixture,observations,history,previous_md):
    """Independent specification calculation; never calls the production scorer.

    EM uses the same pinned library, but window selection, component matching,
    blending, posterior assignment, calibration and decisions are derived here.
    """
    from itertools import permutations
    from sklearn.mixture import GaussianMixture
    ordered=sorted(results,key=lambda r:r.client_id)
    vectors=np.stack([flatten_update(r.update).numpy() for r in ordered]);z=vectors@projection
    pool=np.concatenate([v for t,v in observations if rnd-4<=t<rnd]+[z])
    mixture=previous_mixture;distance=[None]*len(ordered);trajectory=[None]*len(ordered);labels=[None]*len(ordered)
    reason=None
    if len(np.unique(pool,axis=0))<2:reason='fewer_than_two_distinct_observations'
    else:
        fitted=GaussianMixture(n_components=2,covariance_type='diag',reg_covar=1e-6,n_init=1,max_iter=200,tol=1e-3,random_state=1729).fit(pool)
        if not fitted.converged_:raise ValueError('Verification EM did not converge')
        fresh=dict(means=fitted.means_,variances=fitted.covariances_,weights=fitted.weights_)
        if previous_mixture is None:mixture=fresh
        else:
            order=min(permutations(range(2)),key=lambda p:sum(float(np.sum((previous_mixture['means'][i]-fresh['means'][p[i]])**2)) for i in range(2)))
            mixture={k:.8*previous_mixture[k]+.2*fresh[k][list(order)] for k in fresh}
        squared=((z[:,None]-mixture['means'][None])**2/mixture['variances'][None]).sum(axis=2)
        log_density=np.log(mixture['weights'])[None]-.5*(squared+np.log(mixture['variances']).sum(axis=1)[None]+z.shape[1]*np.log(2*np.pi))
        labels=log_density.argmax(axis=1).tolist()
        distance=[float(np.sqrt(squared[i,k])) for i,k in enumerate(labels)]
        for i,r in enumerate(ordered):
            old=previous_md.get(r.client_id)
            if old is not None and old[0]==rnd-1:trajectory[i]=abs(distance[i]-old[1])
    limits={}
    for name in ['norm','md','trajectory']:
        values=[r[name] for t,rows in history if rnd-5<=t<rnd for r in rows if r[name] is not None]
        limits[name]=max(float(np.mean(values)+3*np.std(values,ddof=0)),1e-8) if len(values)>=20 else None
    if reason is None:
        if rnd<=3:reason='initial_three_rounds'
        elif limits['norm'] is None or limits['md'] is None:reason='insufficient_prior_calibration'
    norms=np.linalg.norm(vectors,axis=1);rows=[]
    for i,r in enumerate(ordered):
        row=dict(client_id=r.client_id,sample_count=r.sample_count,norm=float(norms[i]),md=distance[i],trajectory=trajectory[i],component=labels[i],trust=1.,decision='warmup',ratio=None)
        if reason is None:
            ratio=max(row[k]/v for k,v in limits.items() if v is not None and row[k] is not None)
            row.update(ratio=ratio,trust=1. if ratio<=1 else .5 if ratio<=2 else 0.,decision='accept' if ratio<=1 else 'downweight' if ratio<=2 else 'reject')
        rows.append(row)
    mass=sum(r['sample_count']*r['trust'] for r in rows);samples=sum(r['sample_count'] for r in rows)
    if mass<=0:raise ValueError('Independent scoring rejects every client')
    for row in rows:row.update(weight=row['sample_count']*row['trust']/mass,fedavg_weight=row['sample_count']/samples)
    diagnostic=dict(round_id=rnd,phase='warmup' if reason else 'scored',reason=reason,thresholds=limits,clients=rows)
    current_md={r.client_id:(rnd,d) for r,d in zip(ordered,distance) if d is not None}
    return diagnostic,mixture,z,current_md


def _equal_numbers(actual,expected):
    if isinstance(expected,dict):return isinstance(actual,dict) and set(actual)==set(expected) and all(_equal_numbers(actual[k],v) for k,v in expected.items())
    if isinstance(expected,list):return isinstance(actual,list) and len(actual)==len(expected) and all(_equal_numbers(a,b) for a,b in zip(actual,expected))
    if isinstance(expected,float):return type(actual) in (int,float) and np.isclose(actual,expected,atol=1e-12,rtol=1e-12)
    return actual==expected


@threadpool_limits.wrap(limits=1)
def verify_run(path,*,started=None):
    torch.set_num_threads(1)
    path=Path(path);started=perf_counter() if started is None else started
    config=load_temporal_config(path/'config.yaml');meta=json.loads((path/'metadata.json').read_text());summary=json.loads((path/'run_summary.json').read_text())
    from Support.research_runner import completed_preparation
    completed_preparation(config)
    inputs=meta['federated_inputs']
    if str(Path(config['dataset']['bundle_path']).resolve())!=inputs['bundle_path'] or source_hash(inputs['bundle_path'])!=inputs['bundle_sha256']:raise ValueError('Changed bundle source')
    for partition in [*inputs['clients'],inputs['validation']]:
        for field in ['manifest','data','evidence']:
            if source_hash(partition[field+'_path'])!=partition[field+'_sha256']:raise ValueError('Changed prepared source '+field)
    artifact=load_model_artifact(path/'model_artifact.pt');checkpoint=torch.load(path/'checkpoint.pt',weights_only=True,map_location='cpu')
    metrics=json.loads((path/'validation_metrics.json').read_text())['metrics'];rounds=config['federated']['rounds']
    expected={p['client_id']:p['sample_count'] for p in meta['federated_inputs']['clients']}
    if checkpoint['round_id']!=rounds or summary['selected_round']!=rounds:raise ValueError('Incomplete rounds')
    if meta['workflow']!='prepared_temporal_adapted' or meta['test_scored'] or summary['test_scored']:raise ValueError('Wrong evaluation scope')
    final_hash=_state_sha256(artifact.model.state_dict())
    if final_hash!=_state_sha256(checkpoint['model_state']):raise ValueError('Artifact/checkpoint mismatch')
    if final_hash==meta['initialization']['model_state_sha256']:raise ValueError('Model did not change')
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(config['experiment']['seed']).get_state());initial=create_model(artifact.contract.model_config())
    previous=clone_state(initial.state_dict())
    if _state_sha256(previous)!=meta['initialization']['model_state_sha256']:raise ValueError('Unmatched initial state')
    with (path/'round_metrics.csv').open() as stream:round_rows=list(csv.DictReader(stream))
    with (path/'aggregation_weights.csv').open() as stream:weight_rows=list(csv.DictReader(stream))
    if len(round_rows)!=rounds or [int(r['round_id']) for r in round_rows]!=list(range(1,rounds+1)) or any(r['evaluation_role']!='validation' for r in round_rows):raise ValueError('Invalid round coverage/scope')
    dimensions=sum(v.numel() for v in previous.values())
    projection=np.random.default_rng(1729).normal(0,1/np.sqrt(5),size=(dimensions,5))
    if not np.array_equal(projection,np.load(path/'temporal_diagnostics/projection.npy',allow_pickle=False)):raise ValueError('Fixed projection differs')
    mixture=None;observations=[];history=[];prior_md={}
    observed={c:set() for c in expected};independent_history=[]
    for rnd in range(1,rounds+1):
        prefix=path/'temporal_diagnostics'/f'round_{rnd:03d}'
        saved=json.loads(prefix.with_suffix('.json').read_text());states=torch.load(prefix.with_suffix('.pt'),weights_only=True,map_location='cpu')
        selected=json.loads(round_rows[rnd-1]['selected_clients'])
        if len(selected)!=len(set(selected)) or not set(selected)<=set(expected) or states['client_order']!=selected or set(states['client_states'])!=set(selected):raise ValueError('Invalid client selection evidence')
        results=[]
        for client in selected:
            update=extract_update(states['client_states'][client],previous)
            if client not in checkpoint['history'] or rnd not in checkpoint['history'][client] or _state_sha256(update)!=_state_sha256(checkpoint['history'][client][rnd]):raise ValueError('History update differs from local minus global')
            observed[client].add(rnd);results.append(ClientResult(client,rnd,expected[client],states['client_states'][client],update,{}))
        calculated,mixture,projected,prior_md=audit_scores(results,rnd,projection,mixture,observations,history,prior_md)
        if not _equal_numbers(saved,calculated):raise ValueError('Saved temporal scores/thresholds/decisions differ from independent equations')
        weights={r['client_id']:r['weight'] for r in calculated['clients']}
        observations=[(t,v) for t,v in observations if t>=rnd-4]+[(rnd,projected)]
        history=[(t,v) for t,v in history if t>=rnd-5]+[(rnd,calculated['clients'])]
        ordered=sorted(results,key=lambda r:r.client_id);vectors=np.stack([flatten_update(r.update).numpy() for r in ordered])
        with np.load(prefix.with_suffix('.npz'),allow_pickle=False) as arrays:
            if not np.array_equal(arrays['projected'],projected):raise ValueError('Projected observations differ')
            if set(arrays.files)!={'projected'}|set(mixture or {}):raise ValueError('Unexpected mixture evidence fields')
            for key,value in (mixture or {}).items():
                if not np.allclose(arrays[key],value,rtol=1e-12,atol=1e-12):raise ValueError('Saved GMM parameters differ')
        # Separate direct equations guard against merely comparing two saved files.
        for i,row in enumerate(saved['clients']):
            norm=float(np.sqrt(np.dot(vectors[i],vectors[i])))
            if not np.isclose(norm,row['norm'],rtol=1e-12,atol=1e-12):raise ValueError('Norm equation differs')
            if row['md'] is not None:
                k=row['component'];mix=mixture;z=vectors[i]@projection
                md=float(np.sqrt(np.sum((z-mix['means'][k])**2/mix['variances'][k])))
                if not np.isclose(md,row['md'],rtol=1e-12,atol=1e-12):raise ValueError('MD equation differs')
            if saved['phase']=='scored':
                vals=[row[k]/v for k,v in saved['thresholds'].items() if v is not None and row[k] is not None]
                score=max(vals);trust=1. if score<=1 else .5 if score<=2 else 0.
                if trust!=row['trust']:raise ValueError('Trust decision equation differs')
            independent_history.append(dict(round_id=rnd,phase=saved['phase'],**row,**{f'threshold_{k}':v for k,v in saved['thresholds'].items()}))
        rows=[r for r in weight_rows if int(r['round_id'])==rnd]
        if len(rows)!=len(selected) or {r['client_id'] for r in rows}!=set(selected):raise ValueError('Missing weights')
        mass=sum(r['sample_count']*r['trust'] for r in saved['clients'])
        for row in saved['clients']:
            if abs(row['weight']-row['sample_count']*row['trust']/mass)>1e-12:raise ValueError('Sample times trust equation differs')
        for row in rows:
            c=row['client_id']
            if int(row['sample_count'])!=expected[c] or not np.isfinite(float(row['weight'])) or abs(float(row['weight'])-weights[c])>1e-12:raise ValueError('Logged aggregation weight differs')
        weighted=OrderedDict()
        for k,v in previous.items():
            acc=torch.zeros_like(v,dtype=torch.float64)
            for c in selected:acc.add_(states['client_states'][c][k].to(torch.float64),alpha=weights[c])
            weighted[k]=acc.to(v.dtype)
        if _state_sha256(weighted)!=_state_sha256(states['aggregated_state']):raise ValueError('Weighted-state equation differs')
        previous=weighted
    if set(checkpoint['history'])!={c for c in observed if observed[c]} or any(set(checkpoint['history'][c])!=observed[c] for c in checkpoint['history']):raise ValueError('Unexpected history coverage')
    if len(weight_rows)!=sum(len(json.loads(r['selected_clients'])) for r in round_rows):raise ValueError('Extra logged weight rows')
    if _state_sha256(previous)!=final_hash:raise ValueError('Final aggregate differs from artifact')
    bundle_path=Path(config['dataset']['bundle_path']);bundle=json.loads(bundle_path.read_text());counts={}
    for client in bundle['clients']:
        data=load_prepared_data(bundle_path.parent/client['manifest_path'],expected_role='train',expected_contract=artifact.contract)
        counts[client['client_id']]=torch.bincount(data.dataset.tensors[1],minlength=artifact.contract.num_classes).tolist()
        del data
    validation=load_prepared_data(bundle_path.parent/bundle['validation']['manifest_path'],expected_role='validation',expected_contract=artifact.contract)
    reloaded=evaluate_model(artifact.model,DataLoader(validation.dataset,batch_size=config['training']['batch_size'],shuffle=False),'cpu',artifact.contract.num_classes)
    if reloaded!=metrics:raise ValueError('Reloaded validation metrics differ')
    paired=config['temporal']['reference_run_path']
    if paired is not None:
        with (Path(paired)/'round_metrics.csv').open() as stream:base_rows=list(csv.DictReader(stream))
        bv=json.loads((Path(paired)/'verification.json').read_text())
        if bv['initial_state_sha256']!=meta['initialization']['model_state_sha256'] or [(r['round_id'],r['selected_clients']) for r in round_rows]!=[(r['round_id'],r['selected_clients']) for r in base_rows]:raise ValueError('Unmatched paired initialization/selection')
    penalties=penalty_summary(independent_history,counts,artifact.contract.class_names)
    _write_csv(path/'trust_scores.csv',independent_history);_json(path/'honest_client_penalties.json',penalties)
    verification=dict(initial_state_sha256=meta['initialization']['model_state_sha256'],final_state_sha256=final_hash,
        model_artifact_sha256=artifact.artifact_sha256,projection_sha256=source_hash(path/'temporal_diagnostics/projection.npy'),rounds=rounds,train_rows=sum(expected.values()),client_rows=expected,
        validation_rows=len(validation.dataset),history_verified=True,history_update_reference_verified=True,
        temporal_equations_verified=True,aggregation_verified=True,reloaded_validation_exact=True,
        artifact_matches_checkpoint=True,model_changed=True,paired_controls_verified=paired is not None,
        testing_only=artifact.testing_only,test_scored=False,adapted_not_exact_reproduction=True,
        end_to_end_with_verification_seconds=perf_counter()-started,peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        honest_client_penalties=penalties['overall'])
    _json(path/'verification.json',verification);_json(path/'reloaded_validation_metrics.json',reloaded)
    _write_csv(path/'validation_per_class.csv',[dict(class_id=r['class_id'],class_name=artifact.contract.class_names[r['class_id']],**{k:r[k] for k in ['support','precision','recall','f1']}) for r in metrics['per_class']])
    return verification
