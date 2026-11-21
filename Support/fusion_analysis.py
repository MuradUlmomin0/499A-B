"""Descriptive CP6 ablations with explicitly reused frozen endpoint results."""
import argparse
import json
from pathlib import Path
import numpy as np
from Support.data.latest_audit import source_hash
from Support.data.latest_pilot import _json
from Support.fusion_experiments import case_order,_checked
from Support.poisoning_analysis import _read,_rows,_csv,metric_differences,client_penalties

REPOSITORY=Path(__file__).resolve().parents[1]


def fusion_penalties(path):
    result=client_penalties(path,'temporal')
    result.update(attack_detection_available=False,temporal_component_detection_available=True,
        interpretation='Rejection/downweight rates refer only to the temporal component. Zero final weight and below-FedAvg rates use fused weights. Fusion is not a calibrated malicious-client detector; class-bearing subsets overlap.')
    rows=_rows(Path(path)/'aggregation_weights.csv');rescued=[]
    for f in sorted((Path(path)/'temporal_diagnostics').glob('round_*.json')):
        d=_read(f)
        if d['phase']!='scored':continue
        weights={r['client_id']:float(r['weight']) for r in rows if int(r['round_id'])==d['round_id']}
        for c in d['clients']:
            if c['decision']=='reject':rescued.append(weights[c['client_id']]>0)
    result['temporal_rejected_client_rounds']=len(rescued)
    result['temporal_rejected_with_positive_fused_weight']=sum(rescued)
    return result


def analyze(index_path,output_dir):
    entries=_read(index_path);expected=case_order()
    if [e['name'] for e in entries]!=[c['name'] for c in expected]:raise ValueError('All seventeen CP6 cases required')
    for e in entries:
        _checked(e['run_path'],e['control'])
        if e['verification_sha256']!=source_hash(Path(e['run_path'])/'verification.json'):raise ValueError('Run verification changed')
    out=Path(output_dir)
    if out.exists():raise FileExistsError('Refusing to overwrite CP6 analysis')
    out.mkdir(parents=True)
    baseline=_read(REPOSITORY/'Support/docs/cp5_evidence/comparison.json')
    pairs=[{**r,'reused_frozen_endpoint':True} for r in baseline['pairs'] if r['method'] in ['fedmade','temporal']]
    per_class=[{**r,'reused_frozen_endpoint':True} for r in _rows(REPOSITORY/'Support/docs/cp5_evidence/paired_per_class.csv') if r['method'] in ['fedmade','temporal']]
    penalty_rows=[];class_penalties=[];all_runs=[]
    main=[e for e in entries if not e['control'] and not e['repeat']]
    for e in main:
        path=Path(e['run_path']);meta=_read(path/'metadata.json');names=meta['federated_inputs']['contract']['class_names']
        p=fusion_penalties(path);_json(out/(e['name']+'_penalties.json'),p)
        penalty_rows.append(dict(case=e['name'],condition=e['condition'],seed=e['seed'],attacked=e['attacked'],
            malicious_aggregation_mass_mean=p['malicious_aggregation_mass_mean'],temporal_rejected_client_rounds=p['temporal_rejected_client_rounds'],
            temporal_rejected_with_positive_fused_weight=p['temporal_rejected_with_positive_fused_weight'],
            **{f'{group}_{k}':v for group in ['honest','malicious'] for k,v in p[group].items()}))
        for r in p['honest_by_class']:
            class_penalties.append(dict(case=e['name'],condition=e['condition'],seed=e['seed'],attacked=e['attacked'],class_name=names[r['class_id']],
                **{k:v for k,v in r.items() if k!='carrying_honest_clients'},carrying_honest_clients=len(r['carrying_honest_clients'])))
        metrics=_read(path/'validation_metrics.json')['metrics'];all_runs.append({**e,**{k:metrics[k] for k in ['accuracy','balanced_accuracy','macro_f1','weighted_f1','loss']},
            'zero_recall_class_ids':[r['class_id'] for r in metrics['per_class'] if r['recall']==0]})
    for attack in [e for e in main if e['attacked']]:
        clean=next(e for e in main if not e['attacked'] and (e['condition'],e['seed'])==(attack['condition'],attack['seed']))
        cp,ap=Path(clean['run_path']),Path(attack['run_path']);cm,am=_read(cp/'metadata.json'),_read(ap/'metadata.json')
        if cm['federated_inputs']!=am['federated_inputs']:raise ValueError('Fusion paired inputs differ')
        cv,av=_read(cp/'verification.json'),_read(ap/'verification.json')
        for key in ['initial_state_sha256','rounds','train_rows','client_rows','validation_rows']:
            if cv[key]!=av[key]:raise ValueError('Unmatched fusion paired '+key)
        before,after=_read(cp/'validation_metrics.json')['metrics'],_read(ap/'validation_metrics.json')['metrics']
        pairs.append(dict(method='fusion',condition=attack['condition'],seed=attack['seed'],clean_run=str(cp),attacked_run=str(ap),
            **metric_differences(before,after,18,1),reused_frozen_endpoint=False))
        names=cm['federated_inputs']['contract']['class_names']
        for i,(b,a) in enumerate(zip(before['per_class'],after['per_class'])):
            if b['class_id']!=i or a['class_id']!=i or b['support']!=a['support']:raise ValueError('Unmatched class identity/support')
            row=dict(method='fusion',condition=attack['condition'],seed=attack['seed'],class_id=i,class_name=names[i],support=b['support'])
            for k in ['precision','recall','f1']:row.update({f'clean_{k}':b[k],f'attacked_{k}':a[k],f'delta_{k}':a[k]-b[k]})
            per_class.append({**row,'reused_frozen_endpoint':False})
    means={}
    for method in ['fedmade','temporal','fusion']:
        means[method]={}
        for condition in ['iid','dirichlet']:
            subset=[r for r in pairs if r['method']==method and r['condition']==condition]
            if sorted(r['seed'] for r in subset)!=[42,43,44]:raise ValueError('Missing/duplicate ablation seed')
            keys=[k for k,v in subset[0].items() if type(v) in (int,float) and k!='seed']
            means[method][condition]={k:dict(mean=float(np.mean([r[k] for r in subset])),sample_std=float(np.std([r[k] for r in subset],ddof=1))) for k in keys}
    repeat=entries[-1];original=next(e for e in main if e['name']=='fused_poisoned_dirichlet_seed42')
    rv,ov=_read(Path(repeat['run_path'])/'verification.json'),_read(Path(original['run_path'])/'verification.json')
    if rv['final_state_sha256']!=ov['final_state_sha256']:raise ValueError('Full repeat differs')
    result=dict(evaluation_role='validation',test_scored=False,lambda_main=.5,lambda_retuned=False,stealth_claim=False,
        controls=[e for e in entries if e['control']],runs=all_runs,pairs=pairs,means=means,
        reproducibility=dict(original=original['run_path'],repeat=repeat['run_path'],exact_final_state=True,final_state_sha256=rv['final_state_sha256']),
        limitations='Descriptive three-seed results; endpoint controls and repeat excluded from means. CP3/CP4 clean and CP5 attacked endpoints reused explicitly. Target exploratory; targeted label attack, not stealth. Overlapping class-bearing subsets; zero recall classes remain learning limitations.')
    _json(out/'comparison.json',result)
    for name,rows in [('paired_overall.csv',pairs),('paired_per_class.csv',per_class),('client_penalties.csv',penalty_rows),('honest_class_bearing_penalties.csv',class_penalties)]:_csv(out/name,rows)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.index,a.output);print(json.dumps(result['means'],indent=2))


if __name__=='__main__':main()
