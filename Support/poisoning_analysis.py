"""Descriptive clean/poisoned comparison; no retuning, stealth or significance claim."""
from Support.Utils.runtime_paths import load_json as runtime_json, load_json_file as runtime_json_file, load_yaml as runtime_yaml, safe_load_yaml as runtime_safe_yaml
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import yaml
from Support.Utils.prepared_config import _PreparedConfigLoader
from Support.data.latest_audit import source_hash
from Support.data.latest_pilot import _json
from Support.poisoning_experiments import case_order,_checked

METRICS=['accuracy','balanced_accuracy','macro_f1','weighted_f1','loss']


def _read(path):return runtime_json(Path(path).read_text())
def _rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def _csv(path,rows):
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def metric_differences(clean,attacked,target,destination):
    left,right=clean['per_class'][target],attacked['per_class'][target]
    if left['support']!=right['support'] or left['support']<=0:raise ValueError('Unmatched/missing target support')
    support=left['support'];cm=clean['confusion_matrix'];am=attacked['confusion_matrix']
    result=dict(clean_target_recall=left['recall'],attacked_target_recall=right['recall'],recall_degradation=left['recall']-right['recall'],
        clean_target_f1=left['f1'],attacked_target_f1=right['f1'],f1_degradation=left['f1']-right['f1'],target_support=support,
        clean_target_to_benign_count=cm[target][destination],attacked_target_to_benign_count=am[target][destination],
        clean_target_to_benign_rate=cm[target][destination]/support,attacked_target_to_benign_rate=am[target][destination]/support,
        delta_target_to_benign_rate=(am[target][destination]-cm[target][destination])/support)
    for key in METRICS:result.update({f'clean_{key}':clean[key],f'attacked_{key}':attacked[key],f'delta_{key}':attacked[key]-clean[key]})
    if not np.isfinite(list(result.values())).all():raise ValueError('Nonfinite comparison metric')
    return result


def client_penalties(path,method):
    """Use evaluator truth after training; optimizer zero mass is not detection."""
    path=Path(path);truth=_read(path/'attack_ground_truth.json');malicious=set(truth['selected_malicious']);counts=_read(path/'client_class_counts.json')
    a=truth['attack'];rows=_rows(path/'aggregation_weights.csv');prepared=[];masses=[]
    rounds=sorted({int(r['round_id']) for r in rows if a['start_round']<=int(r['round_id'])<=a['end_round']})
    for rnd in rounds:
        selected=[r for r in rows if int(r['round_id'])==rnd];n=sum(int(r['sample_count']) for r in selected)
        diagnoses={}
        if method=='temporal':
            d=_read(path/'temporal_diagnostics'/f'round_{rnd:03d}.json')
            if d['phase']!='scored':continue
            diagnoses={r['client_id']:r for r in d['clients']}
        mass=0.
        for r in selected:
            c=r['client_id'];weight=float(r['weight']);ordinary=int(r['sample_count'])/n
            if not np.isfinite(weight):raise ValueError('Nonfinite penalty weight')
            if c in malicious:mass+=weight
            decision=diagnoses[c]['decision'] if method=='temporal' else None
            prepared.append(dict(round_id=rnd,client_id=c,malicious=c in malicious,zero_weight=weight==0,
                below_fedavg=weight<ordinary-1e-15,rejected=decision=='reject' if decision else None,
                downweighted=decision=='downweight' if decision else None,weight=weight))
        masses.append(mass)
    def rates(subset):
        n=len(subset)
        return dict(client_rounds=n,zero_weight_rate=sum(r['zero_weight'] for r in subset)/n if n else None,
            weight_below_fedavg_rate=sum(r['below_fedavg'] for r in subset)/n if n else None,
            rejection_rate=sum(r['rejected'] for r in subset)/n if n and method=='temporal' else None,
            downweight_rate=sum(r['downweighted'] for r in subset)/n if n and method=='temporal' else None)
    return dict(attack_detection_available=method=='temporal',rate_unit='client-round within attack period; temporal unscored rounds excluded',
        malicious_aggregation_mass_mean=float(np.mean(masses)) if masses else None,
        honest=rates([r for r in prepared if not r['malicious']]),malicious=rates([r for r in prepared if r['malicious']]),
        by_client={c:rates([r for r in prepared if r['client_id']==c]) for c in counts},
        honest_by_class=[dict(class_id=i,carrying_honest_clients=[c for c in counts if c not in malicious and counts[c][i]>0],
            **rates([r for r in prepared if not r['malicious'] and counts[r['client_id']][i]>0])) for i in range(len(next(iter(counts.values()))))],
        interpretation='FedAvg has no detector; FedMADE zero weights are optimization exclusions. Temporal rejection/downweighting are declared decisions, not proof of attacker detection.')


def analyze(index_path,output_dir):
    entries=_read(index_path);expected=case_order()
    if len(entries)!=21 or [e['name'] for e in entries]!=[c['name'] for c in expected]:raise ValueError('Expected three controls and all eighteen attacked cases')
    for e in entries:
        _checked(e['run_path'],e['control'])
        if e.get('verification_sha256')!=source_hash(Path(e['run_path'])/'verification.json'):raise ValueError('Registry verification digest differs')
    out=Path(output_dir)
    if out.exists():raise FileExistsError('Refusing to overwrite poisoning comparison')
    pairs=[];per_class=[];penalties=[];class_penalties=[];details={}
    for e in entries:
        if e['control']:continue
        path=Path(e['run_path']);meta=_read(path/'metadata.json');config=runtime_yaml((path/'config.yaml').read_text(),Loader=_PreparedConfigLoader);clean=Path(meta['paired_clean_run'])
        cm=_read(clean/'metadata.json');v=_read(path/'verification.json');cv=_read(clean/'verification.json')
        if (meta['workflow']!='prepared_poisoning' or meta['method']!=e['method'] or meta['seed']!=e['seed'] or cm['seed']!=e['seed']
                or meta['federated_inputs']!=cm['federated_inputs'] or meta['evaluation_role']!='validation' or cm['evaluation_role']!='validation'):
            raise ValueError('Unmatched method/data/evaluation/seed')
        for key in ['initial_state_sha256','rounds','train_rows','client_rows','validation_rows']:
            if v[key]!=cv[key]:raise ValueError('Unmatched paired '+key)
        target=config['attack']['target_class'];destination=config['attack']['destination_class']
        before=_read(clean/'validation_metrics.json')['metrics'];after=_read(path/'validation_metrics.json')['metrics']
        row=dict(method=e['method'],condition=e['condition'],seed=e['seed'],clean_run=str(clean),attacked_run=str(path),
            **metric_differences(before,after,target,destination));pairs.append(row)
        names=meta['federated_inputs']['contract']['class_names']
        if len(before['per_class'])!=len(names) or len(after['per_class'])!=len(names):raise ValueError('Missing class metrics')
        for i,(b,a) in enumerate(zip(before['per_class'],after['per_class'])):
            if b['class_id']!=i or a['class_id']!=i or b['support']!=a['support']:raise ValueError('Unmatched class identity/support')
            r=dict(method=e['method'],condition=e['condition'],seed=e['seed'],class_id=i,class_name=names[i],support=b['support'])
            for k in ['precision','recall','f1']:r.update({f'clean_{k}':b[k],f'attacked_{k}':a[k],f'delta_{k}':a[k]-b[k]})
            per_class.append(r)
        p=client_penalties(path,e['method']);details[e['name']]=p
        penalties.append(dict(method=e['method'],condition=e['condition'],seed=e['seed'],
            malicious_aggregation_mass_mean=p['malicious_aggregation_mass_mean'],attack_detection_available=p['attack_detection_available'],
            **{f'{group}_{key}':value for group in ['honest','malicious'] for key,value in p[group].items()}))
        for r in p['honest_by_class']:
            class_penalties.append(dict(method=e['method'],condition=e['condition'],seed=e['seed'],class_name=names[r['class_id']],
                **{k:v for k,v in r.items() if k!='carrying_honest_clients'},carrying_honest_clients=len(r['carrying_honest_clients'])))
    means={}
    for method in ['fedavg','fedmade','temporal']:
        means[method]={}
        for condition in ['iid','dirichlet']:
            subset=[r for r in pairs if r['method']==method and r['condition']==condition]
            if sorted(r['seed'] for r in subset)!=[42,43,44]:raise ValueError('Missing/duplicate seed')
            keys=[k for k,v in subset[0].items() if isinstance(v,(int,float)) and k!='seed']
            means[method][condition]={k:dict(mean=float(np.mean([r[k] for r in subset])),sample_std=float(np.std([r[k] for r in subset],ddof=1))) for k in keys}
    result=dict(evaluation_role='validation',test_scored=False,target_selection_exploratory=True,stealth_claim=False,
        controls=[e for e in entries if e['control']],pairs=pairs,means=means,
        limitations='Descriptive three-seed paired outcomes. Label corruption is not evidence of successful degradation; target-to-benign errors include natural errors. Honest class-bearing subsets overlap.')
    out.mkdir(parents=True,exist_ok=False);_json(out/'comparison.json',result)
    for name,rows in [('paired_overall.csv',pairs),('paired_per_class.csv',per_class),('client_penalties.csv',penalties),('honest_class_bearing_penalties.csv',class_penalties)]:_csv(out/name,rows)
    for name,value in details.items():_json(out/(name+'_penalties.json'),value)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.index,a.output);print(json.dumps(result['means'],indent=2))


if __name__=='__main__':main()
