"""Matched clean adapted-FedMADE experiments; no attacks or final-test scoring."""
from Support.Utils.runtime_paths import load_json as runtime_json, load_json_file as runtime_json_file, load_yaml as runtime_yaml, safe_load_yaml as runtime_safe_yaml
import argparse
import csv
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

from Support.data.federated_bundle import load_federated_bundle
from Support.data.latest_audit import source_hash
from Support.data.latest_pilot import _json
from Support.data.prepared import load_prepared_data
from Support.data.split_checks import load_split_evidence,check_disjoint_splits
from Support.federated.client import FederatedClient
from Support.federated.server import FederatedServer
from Support.federated.fedmade import FedMADEAggregator
from Support.federated_runner import _state_sha256
from Support.models.artifact import save_model_artifact
from Support.models.factory import create_model
from Support.research_runner import completed_preparation
from Support.Utils.fedmade_config import load_fedmade_config
from Support.Utils.training_config import load_training_config
from Support.Utils.experiment_logger import ExperimentLogger
from Support.Utils.reproducibility import seed_everything,environment_metadata

REPOSITORY=Path(__file__).resolve().parents[1]


def load_inputs(config):
    """Bind auxiliary files to the completed CP1 report and check role isolation."""
    report_path=completed_preparation(config);report=runtime_json(report_path.read_text())
    bundle=load_federated_bundle(config['dataset']['bundle_path']);m=config['fedmade']
    auxiliary=load_prepared_data(m['auxiliary_manifest_path'],expected_role='auxiliary',expected_contract=bundle.contract)
    evidence=load_split_evidence(m['auxiliary_evidence_path'],data_path=auxiliary.manifest.data_path,sample_count=len(auxiliary.dataset))
    for path in [Path(m['auxiliary_manifest_path']),Path(m['auxiliary_evidence_path']),auxiliary.manifest.data_path]:
        relative=str(path.resolve().relative_to(report_path.parent.resolve()))
        if report['files'].get(relative)!=source_hash(path):raise ValueError('Auxiliary file differs from completed preparation')
    partitions={p.client_id:p.evidence for p in bundle.clients};partitions['validation']=bundle.validation.evidence;partitions['auxiliary']=evidence
    isolation=check_disjoint_splits(partitions,policy='group')
    labels=auxiliary.dataset.tensors[1];counts=torch.bincount(labels,minlength=bundle.contract.num_classes)
    if len(counts)!=bundle.contract.num_classes or (counts==0).any():raise ValueError('Missing auxiliary class')
    return bundle,auxiliary,evidence,isolation


def paired_reference(config,bundle,testing_only):
    path=config['fedmade']['reference_run_path']
    if path is None:
        if not testing_only:raise ValueError('Real-data FedMADE requires its paired frozen FedAvg run')
        return None
    path=Path(path);baseline=load_training_config(path/'config.yaml')
    for key in ['training','initialization','selection']:
        if baseline[key]!=config[key]:raise ValueError(f'Unmatched baseline {key}')
    for key in ['rounds','participation_rate']:
        if baseline['federated'][key]!=config['federated'][key]:raise ValueError('Unmatched federation budget')
    for key in ['seed','device','deterministic']:
        if baseline['experiment'][key]!=config['experiment'][key]:raise ValueError('Unmatched baseline execution')
    if Path(baseline['dataset']['bundle_path'])!=bundle.path:raise ValueError('Unmatched data partition')
    metadata=runtime_json((path/'metadata.json').read_text());verification=runtime_json((path/'verification.json').read_text())
    if (metadata.get('workflow')!='prepared_fedavg' or metadata['federated_inputs']['bundle_sha256']!=bundle.sha256
            or verification['test_scored'] or verification['rounds']!=config['federated']['rounds']):
        raise ValueError('Reference is not the completed paired clean FedAvg run')
    return metadata


def preflight(config_path):
    torch.set_num_threads(1);started=perf_counter();config=load_fedmade_config(config_path)
    bundle,auxiliary,evidence,isolation=load_inputs(config)
    testing=bundle.testing_only or auxiliary.manifest.testing_only
    paired_reference(config,bundle,testing)
    result={'config_sha256':source_hash(config_path),'bundle_sha256':bundle.sha256,
        'auxiliary_data_sha256':evidence.data_sha256,'auxiliary_sidecar_sha256':evidence.sidecar_sha256,
        'auxiliary_rows':len(auxiliary.dataset),'auxiliary_class_counts':torch.bincount(auxiliary.dataset.tensors[1],minlength=bundle.contract.num_classes).tolist(),
        'client_rows':[len(p.data.dataset) for p in bundle.clients],'validation_rows':len(bundle.validation.data.dataset),
        'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'seconds':perf_counter()-started,
        'training_performed':False,'test_scored':False,'isolation_verified':True}
    if result['peak_process_rss_kib']>int(3.5*1024**2):raise MemoryError('FedMADE preflight exceeded 3.5 GiB process guard')
    return result


def run_fedmade(config_path):
    """Use the original client/server/model training path with a new aggregator."""
    torch.set_num_threads(1);config=load_fedmade_config(config_path)
    e,f,t,m=(config[k] for k in ['experiment','federated','training','fedmade'])
    if e['device']=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    bundle,auxiliary,evidence,isolation=load_inputs(config)
    testing=bundle.testing_only or auxiliary.manifest.testing_only
    baseline=paired_reference(config,bundle,testing)
    seed_everything(e['seed'],e['deterministic']);model=create_model(bundle.contract.model_config()).to(e['device'])
    initial_hash=_state_sha256(model.state_dict())
    if baseline is not None and initial_hash!=baseline['initialization']['model_state_sha256']:
        raise ValueError('FedMADE initialization differs from paired FedAvg')
    factory=lambda:create_model(bundle.contract.model_config())
    clients=[FederatedClient(p.client_id,factory,p.data.dataset,t,e['device'],e['seed'],e['deterministic']) for p in bundle.clients]
    metadata=environment_metadata(e['seed'],e['device'],e['deterministic'],REPOSITORY,testing_only=testing,purpose='Clean adapted FedMADE; not an exact paper reproduction')
    metadata.update({'workflow':'prepared_fedmade_adapted','config_path':str(Path(config_path).resolve()),
        'evaluation_role':'validation','selection_policy':'final_round','federated_inputs':bundle.source_metadata(),
        'pytorch_threads':torch.get_num_threads(),'optimizer_state':'reset each round',
        'sample_count_policy':'retained training rows; aggregation weights use CPM optimization, not sample counts',
        'initialization':{'mode':'random','model_state_sha256':initial_hash},
        'auxiliary':{'manifest_path':str(auxiliary.manifest.manifest_path),'manifest_sha256':source_hash(auxiliary.manifest.manifest_path),
                     'data_sha256':evidence.data_sha256,'evidence_sha256':evidence.sidecar_sha256,
                     'sample_count':len(auxiliary.dataset),'isolation':isolation},
        'fedmade_adaptation':{'solver':'scipy_nnls','norm':'frobenius','eps':m['eps'],'min_samples':1,'maxiter':m['maxiter'],
                             'paper_cluster_solver':'Adam learning rate 0.01; replaced with user-approved NNLS'},
        'paired_fedavg_run':m['reference_run_path'],'test_scored':False})
    logger=ExperimentLogger(e['output_dir'],e['name'],config,metadata,evaluation_role='validation')
    aggregator=FedMADEAggregator(factory,auxiliary.dataset,bundle.contract.num_classes,eps=m['eps'],batch_size=t['batch_size'],
        maxiter=m['maxiter'],device=e['device'],diagnostic_dir=logger.path/'fedmade_diagnostics')
    validation_loader=DataLoader(bundle.validation.data.dataset,batch_size=t['batch_size'],shuffle=False,num_workers=0,
                                 generator=torch.Generator().manual_seed(e['seed']))
    server=FederatedServer(model,clients,aggregator,validation_loader,bundle.contract.num_classes,f['participation_rate'],e['seed'],e['device'],logger)
    started=perf_counter();server.run(f['rounds'])
    artifact=save_model_artifact(server.model,logger.path/'model_artifact.pt',contract=bundle.contract,model_name=e['name'],training_completed=True,testing_only=testing)
    _json(logger.path/'run_summary.json',{'schema_version':1,'status':'completed','workflow':'prepared_fedmade_adapted',
        'testing_only':testing,'training_completed':True,'selection_policy':'final_round','selected_round':server.current_round,
        'evaluation_role':'validation','model_artifact':artifact.name,'model_artifact_sha256':source_hash(artifact),
        'resume_supported':False,'test_scored':False,'execution_duration_seconds':perf_counter()-started})
    return logger.path


def run_verified(config_path):
    from Support.fedmade_verification import verify_run
    started=perf_counter();path=run_fedmade(config_path);verify_run(path,started=started)
    print(json.dumps({'run_completed':str(path)}),flush=True);return path


def generate_configs(directory,results):
    """Clone the six frozen CP2 configurations and add only method settings."""
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    references=runtime_json((REPOSITORY/'Support/docs/cp2_evidence/verification.json').read_text())['primary_runs']
    aux=REPOSITORY/'data/processed/research_cp1_v1'
    for condition in ['iid','dirichlet']:
        for seed in [42,43,44]:
            name=f'{condition}_seed{seed}';v=load_training_config(REPOSITORY/f'Support/configs/cp2_v1/{name}.yaml')
            reference=next(r['run_path'] for r in references if r['condition']==condition and r['seed']==seed)
            v['experiment']['name']='cp3_'+name;v['experiment']['output_dir']=os.path.relpath(Path(results).resolve(),directory)
            v['dataset']['bundle_path']=os.path.relpath(v['dataset']['bundle_path'],directory);v['federated']['aggregation']='fedmade'
            v['fedmade']={'auxiliary_manifest_path':os.path.relpath(aux/'auxiliary.manifest.json',directory),
                'auxiliary_evidence_path':os.path.relpath(aux/'auxiliary.ids.json',directory),
                'reference_run_path':os.path.relpath(REPOSITORY/reference,directory),'eps':.5,'min_samples':1,
                'solver':'nnls','norm':'frobenius','maxiter':10000}
            path=directory/f'{name}.yaml';path.write_text(yaml.safe_dump(v,sort_keys=False));load_fedmade_config(path)


def run_matrix(config_dir, output_dir):
    directory=Path(config_dir).resolve();output=Path(output_dir).resolve()
    evidence=directory/'preflight.json'
    if not evidence.is_file():raise ValueError('Measured FedMADE preflight required')
    measured=runtime_json(evidence.read_text());first=load_fedmade_config(directory/'iid_seed42.yaml')
    if (measured['config_sha256']!=source_hash(directory/'iid_seed42.yaml') or measured['bundle_sha256']!=source_hash(first['dataset']['bundle_path'])
            or measured['peak_process_rss_kib']>int(3.5*1024**2) or measured['training_performed']):raise ValueError('Stale/invalid preflight')
    output.mkdir(parents=True,exist_ok=False);entries=[]
    for condition in ['iid','dirichlet']:
        for seed in [42,43,44]:
            config=directory/f'{condition}_seed{seed}.yaml';v=load_fedmade_config(config)
            root=Path(v['experiment']['output_dir']);before=set(root.glob(v['experiment']['name']+'_*'))
            print(f'Starting adapted FedMADE {condition} seed {seed}',flush=True)
            with (output/f'{condition}_seed{seed}.log').open('x') as stream:
                subprocess.run([sys.executable,'-m','Support.fedmade_runner','train-one','--config',str(config)],
                    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'},
                    stdout=stream,stderr=subprocess.STDOUT,check=True)
            fresh=set(root.glob(v['experiment']['name']+'_*'))-before
            if len(fresh)!=1:raise RuntimeError('Ambiguous FedMADE run identity')
            path=fresh.pop()
            if not (path/'verification.json').exists():raise RuntimeError('Missing run verification')
            entries.append({'condition':condition,'seed':seed,'run_path':str(path)})
            (output/'progress.json').write_text(json.dumps(entries,indent=2)+'\n')
            print(f'Completed {condition} seed {seed}: {path}',flush=True)
    from Support.research_runner import summarize_matrix
    return summarize_matrix(entries,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    for name in ['train-one','preflight']:
        p=sub.add_parser(name);p.add_argument('--config',type=Path,required=True)
    p=sub.add_parser('configs');p.add_argument('--directory',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    p=sub.add_parser('matrix');p.add_argument('--config-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='train-one':run_verified(args.config)
    elif args.action=='preflight':print(json.dumps(preflight(args.config),indent=2))
    elif args.action=='configs':generate_configs(args.directory,args.results)
    else:run_matrix(args.config_dir,args.output)


if __name__=='__main__':main()
