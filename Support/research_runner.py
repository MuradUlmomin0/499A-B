"""CP1 preparation and sequential CP2 FedAvg runs; no final-test scoring."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from support.data.latest_audit import source_hash
from support.data.latest_pilot import _json
from support.data.federated_bundle import load_federated_bundle
from support.data.prepared import load_prepared_data
from support.evaluation.metrics import evaluate_model
from support.federated_runner import run_federated, _state_sha256
from support.models.artifact import load_model_artifact
from support.utils.training_config import load_training_config


def completed_preparation(config):
    """Reject partial or modified research exports before model execution."""
    bundle=Path(config['dataset']['bundle_path']);root=bundle.parent.parent
    report_path=root/'preparation_report.json'
    if not report_path.is_file():raise ValueError('A completed preparation report is required')
    report=json.loads(report_path.read_text());description=json.loads(bundle.read_text())
    required=[bundle,root/'preprocessing.json']
    for ref in [*description['clients'],description['validation']]:
        manifest=(bundle.parent/ref['manifest_path']).resolve()
        evidence=(bundle.parent/ref['evidence_path']).resolve()
        metadata=json.loads(manifest.read_text())
        required.extend([manifest,evidence,(manifest.parent/metadata['data_path']).resolve()])
    for path in required:
        key=str(path.resolve().relative_to(root.resolve()))
        if key not in report['files'] or source_hash(path)!=report['files'][key]:
            raise ValueError('Completed preparation input hash mismatch')
    return report_path


def make_configs(prepared, settings, directory, results):
    """Generate existing strict v1 configurations, referring to validation only."""
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    paths=[]
    for condition in ['iid','dirichlet']:
        for seed in settings['partition_and_training_seeds']:
            name=f'{condition}_seed{seed}'
            value={'schema_version':1,'experiment':{'name':'cp2_'+name,'seed':seed,'device':'cpu',
                'deterministic':True,'output_dir':os.path.relpath(Path(results).resolve(),directory)},
                'dataset':{'bundle_path':os.path.relpath(Path(prepared).resolve()/name/'federated_bundle.json',directory)},
                'initialization':{'mode':'random','artifact_path':None},
                'federated':{'aggregation':'fedavg','rounds':settings['rounds'],'participation_rate':settings['participation_rate']},
                'training':{k:settings[k] for k in ['optimizer','learning_rate','local_epochs','batch_size']},
                'selection':{'policy':'final_round'}}
            path=directory/f'{name}.yaml';path.write_text(yaml.safe_dump(value,sort_keys=False));load_training_config(path);paths.append(path)
    return paths


def preflight(config_path):
    """Load a real full bundle and measure process memory before model training."""
    torch.set_num_threads(1);started=perf_counter();config=load_training_config(config_path)
    report_path=completed_preparation(config)
    bundle=load_federated_bundle(config['dataset']['bundle_path'])
    result={'client_rows':[len(p.data.dataset) for p in bundle.clients],
        'validation_rows':len(bundle.validation.data.dataset),'bundle_sha256':bundle.sha256,
        'seconds':perf_counter()-started,'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'training_performed':False,'test_scored':False,
        'preparation_report_sha256':source_hash(report_path)}
    if result['peak_process_rss_kib']>int(3.5*1024**2):raise MemoryError('Preflight exceeded 3.5 GiB process guard')
    return result


def run_one(config_path):
    """Existing FedAvg runner plus checks of exports, histories, weights and reload."""
    torch.set_num_threads(1);started=perf_counter();config=load_training_config(config_path)
    completed_preparation(config)
    path=run_federated(config_path)
    metadata=json.loads((path/'metadata.json').read_text());summary=json.loads((path/'run_summary.json').read_text())
    metrics=json.loads((path/'validation_metrics.json').read_text())['metrics']
    artifact=load_model_artifact(path/'model_artifact.pt')
    checkpoint=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
    expected={p['client_id']:p['sample_count'] for p in metadata['federated_inputs']['clients']}
    rounds=config['federated']['rounds'];total=sum(expected.values())
    if summary['selected_round']!=rounds or checkpoint['round_id']!=rounds:raise ValueError('Incomplete rounds')
    state_hash=_state_sha256(artifact.model.state_dict())
    if state_hash!=_state_sha256(checkpoint['model_state']):raise ValueError('Artifact state differs from checkpoint')
    if state_hash==metadata['initialization']['model_state_sha256']:raise ValueError('Training did not change model')
    if set(checkpoint['history'])!=set(expected):raise ValueError('Missing client history')
    if any(set(v)!=set(range(1,rounds+1)) for v in checkpoint['history'].values()):raise ValueError('Incomplete history')
    with (path/'aggregation_weights.csv').open() as stream:weights=list(csv.DictReader(stream))
    for rnd in range(1,rounds+1):
        rows=[r for r in weights if int(r['round_id'])==rnd]
        if len(rows)!=len(expected) or {r['client_id'] for r in rows}!=set(expected):raise ValueError('Missing aggregation rows')
        for r in rows:
            n=expected[r['client_id']]
            if int(r['sample_count'])!=n or abs(float(r['weight'])-n/total)>1e-12:raise ValueError('Incorrect sample weight')
        if abs(sum(float(r['weight']) for r in rows)-1)>1e-12:raise ValueError('Weights do not sum to one')
    with (path/'round_metrics.csv').open() as stream:round_rows=list(csv.DictReader(stream))
    if len(round_rows)!=rounds or any(r['evaluation_role']!='validation' for r in round_rows):raise ValueError('Wrong round evaluation')
    bundle_path=Path(config['dataset']['bundle_path']);bundle=json.loads(bundle_path.read_text())
    val_path=(bundle_path.parent/bundle['validation']['manifest_path']).resolve()
    validation=load_prepared_data(val_path,expected_role='validation',expected_contract=artifact.contract)
    reloaded=evaluate_model(artifact.model,DataLoader(validation.dataset,batch_size=config['training']['batch_size'],shuffle=False), 'cpu',artifact.contract.num_classes)
    if reloaded!=metrics:raise ValueError('Reloaded validation differs from logged final metrics')
    verification={'initial_state_sha256':metadata['initialization']['model_state_sha256'],
        'final_state_sha256':state_hash,'model_artifact_sha256':artifact.artifact_sha256,
        'rounds':rounds,'train_rows':total,'client_rows':expected,'validation_rows':len(validation.dataset),
        'history_verified':True,'aggregation_verified':True,'reloaded_validation_exact':True,
        'artifact_matches_checkpoint':True,'model_changed':True,'test_scored':False,
        'testing_only':artifact.testing_only,'end_to_end_with_verification_seconds':perf_counter()-started,
        'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    _json(path/'verification.json',verification)
    with (path/'validation_per_class.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['class_id','class_name','support','precision','recall','f1'])
        for row in metrics['per_class']:
            writer.writerow([row['class_id'],artifact.contract.class_names[row['class_id']],*[row[k] for k in ['support','precision','recall','f1']]])
    _json(path/'reloaded_validation_metrics.json',reloaded)
    print(json.dumps({'run_completed':str(path),'verification':verification}),flush=True)
    return path


def summarize_matrix(entries, destination):
    """Report every seed and sample standard deviation; never pick a best seed."""
    keys=['accuracy','balanced_accuracy','macro_f1','weighted_f1','loss'];rows=[];initial={}
    for entry in entries:
        path=Path(entry['run_path']);v=json.loads((path/'verification.json').read_text())
        metrics=json.loads((path/'validation_metrics.json').read_text())['metrics']
        seed=entry['seed']
        if seed in initial and initial[seed]!=v['initial_state_sha256']:raise ValueError('Unmatched initial states')
        initial[seed]=v['initial_state_sha256']
        if v['testing_only']:raise ValueError('Synthetic data cannot enter research summary')
        rows.append({**entry,**{k:metrics[k] for k in keys},'peak_process_rss_kib':v['peak_process_rss_kib'],
            'duration_seconds':v['end_to_end_with_verification_seconds'],
            'zero_recall_class_ids':[r['class_id'] for r in metrics['per_class'] if r['recall']==0]})
    result={'evaluation_role':'validation','test_scored':False,'runs':rows,'conditions':{},'matched_initial_state_sha256':initial,
            'limitations':'Three seeds describe this fixed bounded benchmark; no inferential significance or capture independence is established.'}
    for condition in ['iid','dirichlet']:
        selected=[r for r in rows if r['condition']==condition]
        result['conditions'][condition]={k:{'mean':float(np.mean([r[k] for r in selected])),
            'sample_std':float(np.std([r[k] for r in selected],ddof=1)) if len(selected)>1 else None} for k in keys}
    _json(Path(destination)/'comparison.json',result)
    with (Path(destination)/'comparison.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result


def run_matrix(config_dir, matrix_output):
    """Each run has a fresh process; preserve failures and completed artifacts."""
    preflight_path=Path(config_dir)/'preflight.json'
    if not preflight_path.is_file():raise ValueError('Measured preflight evidence is required')
    measured=json.loads(preflight_path.read_text())
    first=load_training_config(Path(config_dir)/'iid_seed42.yaml')
    report_path=completed_preparation(first)
    if (measured['preparation_report_sha256']!=source_hash(report_path) or
            measured['bundle_sha256']!=source_hash(first['dataset']['bundle_path']) or
            measured['peak_process_rss_kib']>int(3.5*1024**2) or measured['training_performed']):
        raise ValueError('Measured preflight does not match prepared inputs or memory budget')
    output=Path(matrix_output).resolve();output.mkdir(parents=True,exist_ok=False);entries=[]
    for condition in ['iid','dirichlet']:
        for seed in [42,43,44]:
            config=Path(config_dir).resolve()/f'{condition}_seed{seed}.yaml'
            parsed=load_training_config(config);root=Path(parsed['experiment']['output_dir'])
            before=set(root.glob(parsed['experiment']['name']+'_*'))
            log=output/f'{condition}_seed{seed}.log'
            print(f'Starting {condition}, seed {seed}',flush=True)
            with log.open('x') as stream:
                subprocess.run([sys.executable,'-m','support.research_runner','train-one','--config',str(config)],
                    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'},
                    stdout=stream,stderr=subprocess.STDOUT,check=True)
            new=set(root.glob(parsed['experiment']['name']+'_*'))-before
            if len(new)!=1:raise RuntimeError('Ambiguous completed run identity')
            path=new.pop()
            if not (path/'verification.json').exists():raise RuntimeError('Run verification missing')
            entries.append({'condition':condition,'seed':seed,'run_path':str(path)})
            (output/'progress.json').write_text(json.dumps(entries,indent=2)+'\n')
            print(f'Completed {condition}, seed {seed}: {path}',flush=True)
    return summarize_matrix(entries,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--source-config',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--raw-output',type=Path,required=True);p.add_argument('--prepared-output',type=Path,required=True);p.add_argument('--work-dir',type=Path,required=True)
    p=sub.add_parser('export');p.add_argument('--source-config',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--raw-output',type=Path,required=True);p.add_argument('--prepared-output',type=Path,required=True)
    for name in ['train-one','preflight']:
        p=sub.add_parser(name);p.add_argument('--config',type=Path,required=True)
    p=sub.add_parser('matrix');p.add_argument('--config-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action in ('prepare','export'):
        from support.data.research_source import prepare_source
        from support.data.research_export import export_research
        c=yaml.safe_load(args.source_config.read_text());p=yaml.safe_load(args.protocol.read_text());s=p['proposed_settings']
        if not p['review']['numeric_settings_approved'] or p['source']['archive_sha256']!=c['archive_sha256']:raise ValueError('Protocol approval/source mismatch')
        if args.action=='prepare':prepare_source(c,args.raw_output,args.work_dir,s['training_rows'],s['auxiliary_groups_per_class'],s['preparation_seed'])
        export_research(args.raw_output,{**c,'hidden_dims':s['hidden_dims']},s,args.prepared_output)
    elif args.action=='train-one':run_one(args.config)
    elif args.action=='preflight':print(json.dumps(preflight(args.config),indent=2))
    else:run_matrix(args.config_dir,args.output)


if __name__=='__main__':main()
