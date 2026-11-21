"""All-method CP7 tables from verified final-test evidence and historical logs."""
import argparse
import csv
from pathlib import Path
import statistics

from Support.final_protocol import METHODS, CONDITIONS, SEEDS, load_freeze, read, write_new
from Support.final_evaluation import verify_case
from Support.poisoning_analysis import metric_differences
from Support.data.latest_audit import source_hash

SCALARS = ('accuracy','balanced_accuracy','macro_precision','macro_recall','macro_f1','weighted_f1','loss')
RARE = ('Uploading_Attack','Recon-PingSweep','Backdoor_Malware','XSS','SqlInjection',
        'CommandInjection','BrowserHijacking','DictionaryBruteForce')


def csv_write(path, rows):
    if not rows: raise ValueError('No rows for '+str(path))
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def csv_read(path):
    with Path(path).open() as stream: return list(csv.DictReader(stream))


def summarize_pairs(rows, *, methods=METHODS, conditions=CONDITIONS):
    summaries=[]
    for method in methods:
        for condition in conditions:
            subset=[r for r in rows if r['method']==method and r['condition']==condition]
            if sorted(r['seed'] for r in subset)!=list(SEEDS): raise ValueError('Exactly three distinct seeds required')
            keys=[k for k,v in subset[0].items() if type(v) in (int,float) and k!='seed']
            summary=dict(method=method,condition=condition,n_seeds=3)
            for k in keys:
                values=[r[k] for r in subset]
                summary[k+'_mean']=statistics.mean(values)
                summary[k+'_sample_std']=statistics.stdev(values)
            summaries.append(summary)
    return summaries


def log_penalties(case, counts, names):
    """Historical rounds 4–20; evaluator truth stays outside defense inputs."""
    path=Path(case['run_path']);malicious=set()
    if case['attacked']: malicious=set(read(path/'attack_ground_truth.json')['selected_malicious'])
    records=[]
    weights=csv_read(path/'aggregation_weights.csv')
    for rnd in range(4,21):
        selected=[r for r in weights if int(r['round_id'])==rnd]
        total=sum(int(r['sample_count']) for r in selected)
        temporal=path/'temporal_diagnostics'/f'round_{rnd:03d}.json'
        decisions={}
        if case['method'] in ['temporal','fusion']:
            d=read(temporal)
            if d['phase']!='scored':continue
            decisions={r['client_id']:r['decision'] for r in d['clients']}
        for r in selected:
            c=r['client_id'];w=float(r['weight'])
            records.append(dict(client=c,malicious=c in malicious,zero=w==0,
                                below=w<int(r['sample_count'])/total-1e-15,
                                rejected=decisions.get(c)=='reject',downweighted=decisions.get(c)=='downweight',weight=w))
    def rate(subset):
        n=len(subset)
        return dict(client_rounds=n,zero_final_weight_rate=sum(r['zero'] for r in subset)/n if n else None,
                    below_fedavg_rate=sum(r['below'] for r in subset)/n if n else None,
                    temporal_rejection_rate=sum(r['rejected'] for r in subset)/n if n and case['method'] in ['temporal','fusion'] else None,
                    temporal_downweight_rate=sum(r['downweighted'] for r in subset)/n if n and case['method'] in ['temporal','fusion'] else None)
    identity={k:case[k] for k in ['name','method','condition','seed','attacked']}
    group_rows=[{**identity,'group':g,**rate([r for r in records if r['malicious']==m])}
                for g,m in [('honest',False),('malicious',True)]]
    class_rows=[]
    for name in names:
        class_rows.append({**identity,'class_name':name,**rate([r for r in records if not r['malicious'] and int(counts[r['client']][name])>0])})
    return group_rows,class_rows


def analyze(freeze_path, matrix, output):
    frozen=load_freeze(freeze_path);entries=read(Path(matrix)/'progress.json')
    if len(entries)!=48 or [e['name'] for e in entries]!=[c['name'] for c in frozen['cases']]:
        raise ValueError('All 48 cases required for final analysis')
    output=Path(output)
    if output.exists():raise FileExistsError('Refusing to overwrite final analysis')
    for e in entries:
        if e['artifacts_sha256']!=source_hash(Path(e['directory'])/'artifacts.json'):
            raise ValueError('Final-analysis artifact manifest changed')
    records=[verify_case(e['directory'],c,frozen,freeze_path) for e,c in zip(entries,frozen['cases'])]
    names=read(frozen['test_manifest'])['contract']['class_names']
    class_counts=csv_read(Path(frozen['test_manifest']).parent/'client_class_distributions.csv')
    overall=[];per_class=[];pairs=[];paired_class=[];convergence=[];resources=[];penalties=[];class_penalties=[]
    for record in records:
        c,m=record['case'],record['metrics'];identity={k:c[k] for k in ['name','method','condition','seed','attacked']}
        overall.append({**identity,**{k:m[k] for k in SCALARS},'zero_recall_classes':'; '.join(r['class_name'] for r in m['per_class'] if r['recall']==0)})
        per_class.extend({**identity,**r,'lowest_training_support':r['class_name'] in RARE} for r in m['per_class'])
        run=Path(c['run_path']);v=read(run/'verification.json')
        rounds=csv_read(run/'round_metrics.csv')
        if [int(r['round_id']) for r in rounds]!=list(range(1,21)):raise ValueError('Incomplete historical convergence')
        for r in rounds:
            if r['evaluation_role']!='validation':raise ValueError('Historical round role differs')
            convergence.append({**identity,'round_id':int(r['round_id']),'evaluation_role':'validation',
                                **{k:float(r[k]) for k in SCALARS},
                                'round_duration_seconds':float(r['round_duration']),'aggregation_duration_seconds':float(r['aggregation_duration'])})
        resources.append({**identity,'historical_peak_rss_kib':v.get('peak_process_rss_kib'),
                          'historical_sum_round_seconds':sum(float(r['round_duration']) for r in rounds),
                          'historical_sum_aggregation_seconds':sum(float(r['aggregation_duration']) for r in rounds),
                          'test_peak_rss_kib':record['peak_process_rss_kib'],'test_inference_and_metrics_seconds':record['inference_and_metrics_seconds']})
        counts={r['client_id']:r for r in class_counts if r['condition']==c['condition'] and int(r['seed'])==c['seed']}
        group,classes=log_penalties(c,counts,names);penalties.extend(group);class_penalties.extend(classes)
        if c['attacked']:
            clean=next(r for r in records if not r['case']['attacked'] and all(r['case'][k]==c[k] for k in ['method','condition','seed']))
            before=clean['metrics']
            pair=dict(method=c['method'],condition=c['condition'],seed=c['seed'],**metric_differences(before,m,18,1))
            for key in ['macro_precision','macro_recall']:
                pair.update({f'clean_{key}':before[key],f'attacked_{key}':m[key],f'delta_{key}':m[key]-before[key]})
            pairs.append(pair)
            for b,a in zip(before['per_class'],m['per_class']):
                if b['class_id']!=a['class_id'] or b['support']!=a['support']:raise ValueError('Class pairing differs')
                paired_class.append(dict(method=c['method'],condition=c['condition'],seed=c['seed'],class_id=a['class_id'],
                    class_name=a['class_name'],support=a['support'],lowest_training_support=a['class_name'] in RARE,
                    **{f'{prefix}_{k}':value for k in ['precision','recall','f1']
                       for prefix,value in [('clean',b[k]),('attacked',a[k]),('delta',a[k]-b[k])]}))
    summaries=summarize_pairs(pairs)
    by_class=[]
    for name in names:
        by_class.extend({**s,'class_name':name} for s in summarize_pairs([r for r in paired_class if r['class_name']==name]))
    output.mkdir(parents=True)
    for filename,rows in [('overall.csv',overall),('per_class.csv',per_class),('paired_overall.csv',pairs),
                          ('paired_per_class.csv',paired_class),('summary.csv',summaries),('per_class_summary.csv',by_class),
                          ('minority_class_summary.csv',[r for r in by_class if r['class_name'] in RARE]),
                          ('validation_convergence.csv',convergence),('resources.csv',resources),
                          ('client_penalties.csv',penalties),('honest_class_bearing_penalties.csv',class_penalties)]:
        csv_write(output/filename,rows)
    comparison=dict(evaluation_role='test',test_scored=True,training_performed=False,case_count=48,paired_count=24,
                    model_selection='previously_frozen_final_round',summary=summaries,
                    limitations='Three-seed descriptive sample standard deviations; targeted label poisoning, exploratory target; simulated CSV clients; no novelty, significance, stealth or minority-recovery inference. Validation convergence and client penalties reused from training logs.')
    write_new(output/'comparison.json',comparison)
    return comparison


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--matrix',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    analyze(a.freeze,a.matrix,a.output);print('All 48 cases and 24 paired comparisons analyzed.')


if __name__=='__main__':main()
