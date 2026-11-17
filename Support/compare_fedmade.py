"""Descriptive paired clean FedAvg/adapted-FedMADE validation comparison."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from support.data.latest_pilot import _json

METRICS=['accuracy','balanced_accuracy','macro_f1','weighted_f1','loss']


def _read(path,name):return json.loads((path/name).read_text())


def compare(summary_path, output_dir):
    """Require matched, verified six-case inputs; include every seed/class."""
    output=Path(output_dir)
    if output.exists():raise FileExistsError('Refusing to overwrite paired comparison')
    source=json.loads(Path(summary_path).read_text());entries=source['runs']
    if len(entries)!=6 or {(e['condition'],e['seed']) for e in entries}!={(c,s) for c in ['iid','dirichlet'] for s in [42,43,44]}:
        raise ValueError('Comparison requires exactly the frozen six conditions/seeds')
    pairs=[];classes={};class_pairs=[]
    for entry in entries:
        made=Path(entry['run_path']);mm=_read(made,'metadata.json');base=Path(mm['paired_fedavg_run']);bm=_read(base,'metadata.json')
        mv=_read(made,'verification.json');bv=_read(base,'verification.json')
        if (mm['workflow']!='prepared_fedmade_adapted' or bm['workflow']!='prepared_fedavg'
                or mm['seed']!=entry['seed'] or bm['seed']!=entry['seed']
                or mm['evaluation_role']!='validation' or bm['evaluation_role']!='validation'
                or mm['federated_inputs']['bundle_sha256']!=bm['federated_inputs']['bundle_sha256']
                or mm['federated_inputs']['contract']!=bm['federated_inputs']['contract']):
            raise ValueError('Unmatched data/model/seed/evaluation scope')
        for v in [mv,bv]:
            if v['testing_only'] or v['test_scored'] or not v['aggregation_verified'] or not v['reloaded_validation_exact']:
                raise ValueError('Only verified real-source validation runs may enter comparison')
        for key in ['initial_state_sha256','rounds','train_rows','client_rows','validation_rows']:
            if mv[key]!=bv[key]:raise ValueError(f'Unmatched reference {key}')
        made_metrics=_read(made,'validation_metrics.json')['metrics'];base_metrics=_read(base,'validation_metrics.json')['metrics']
        row={'condition':entry['condition'],'seed':entry['seed'],'fedavg_run':str(base),'fedmade_run':str(made)}
        for key in METRICS:
            b,m=float(base_metrics[key]),float(made_metrics[key])
            if not np.isfinite([b,m]).all():raise ValueError('Nonfinite comparison metric')
            row.update({f'fedavg_{key}':b,f'fedmade_{key}':m,f'delta_{key}':m-b})
        pairs.append(row)
        tables=[]
        for path in [base,made]:
            with (path/'validation_per_class.csv').open() as stream:tables.append(list(csv.DictReader(stream)))
        if not tables[0] or len(tables[0])!=len(tables[1]):raise ValueError('Missing class table')
        for i,(b,m) in enumerate(zip(*tables)):
            if (b['class_id']!=str(i) or m['class_id']!=str(i) or b['class_name']!=m['class_name']
                    or b['support']!=m['support']):raise ValueError('Unmatched class identity/support')
            if i in classes and classes[i]!=b['class_name']:raise ValueError('Class mapping changed across runs')
            classes[i]=b['class_name'];r={'condition':entry['condition'],'seed':entry['seed'],'class_id':i,'class_name':b['class_name'],'support':int(b['support'])}
            for key in ['precision','recall','f1']:
                bb,mm=float(b[key]),float(m[key])
                if not np.isfinite([bb,mm]).all():raise ValueError('Nonfinite class metric')
                r.update({f'fedavg_{key}':bb,f'fedmade_{key}':mm,f'delta_{key}':mm-bb})
            class_pairs.append(r)
    result={'evaluation_role':'validation','test_scored':False,'method':'adapted_fedmade_nnls_not_exact_paper_reproduction',
            'pairs':pairs,'conditions':{},'limitations':'Three descriptive paired seeds; no significance or poisoning-resistance claim.'}
    for condition in ['iid','dirichlet']:
        subset=[r for r in pairs if r['condition']==condition];result['conditions'][condition]={}
        for key in METRICS:
            result['conditions'][condition][key]={
                'fedavg_mean':float(np.mean([r[f'fedavg_{key}'] for r in subset])),
                'fedmade_mean':float(np.mean([r[f'fedmade_{key}'] for r in subset])),
                'mean_delta':float(np.mean([r[f'delta_{key}'] for r in subset])),
                'sample_std_delta':float(np.std([r[f'delta_{key}'] for r in subset],ddof=1)),
                'fedavg_sample_std':float(np.std([r[f'fedavg_{key}'] for r in subset],ddof=1)),
                'fedmade_sample_std':float(np.std([r[f'fedmade_{key}'] for r in subset],ddof=1))}
    aggregate=[]
    for condition in ['iid','dirichlet']:
        for i,name in classes.items():
            subset=[r for r in class_pairs if r['condition']==condition and r['class_id']==i]
            if len(subset)!=3 or len({r['support'] for r in subset})!=1:raise ValueError('Class support/seeds differ across condition')
            r={'condition':condition,'class_id':i,'class_name':name,'support':subset[0]['support']}
            for key in ['precision','recall','f1']:
                for prefix in ['fedavg','fedmade','delta']:r[f'{prefix}_{key}_mean']=float(np.mean([x[f'{prefix}_{key}'] for x in subset]))
            aggregate.append(r)
    output.mkdir(parents=True,exist_ok=False);_json(output/'paired_comparison.json',result)
    for name,rows in [('paired_overall.csv',pairs),('paired_per_class.csv',class_pairs),('per_class_comparison.csv',aggregate)]:
        with (output/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--summary',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=compare(args.summary,args.output);print(json.dumps(result['conditions'],indent=2))


if __name__=='__main__':main()
