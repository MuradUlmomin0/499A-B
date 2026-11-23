"""Clean matched CP4 temporal-trust runs; no poisoning or final-test evaluation."""
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

import torch
from torch.utils.data import DataLoader
import yaml
from threadpoolctl import threadpool_limits,threadpool_info

from Support.data.federated_bundle import load_federated_bundle
from Support.data.latest_audit import source_hash
from Support.data.latest_pilot import _json
from Support.federated.client import FederatedClient
from Support.federated.server import FederatedServer
from Support.federated.temporal import TemporalTrustAggregator,DEFAULTS
from Support.federated_runner import _state_sha256
from Support.models.artifact import save_model_artifact
from Support.models.factory import create_model
from Support.research_runner import completed_preparation
from Support.Utils.temporal_config import load_temporal_config
from Support.Utils.training_config import load_training_config
from Support.Utils.experiment_logger import ExperimentLogger
from Support.Utils.reproducibility import seed_everything,environment_metadata

REPOSITORY=Path(__file__).resolve().parents[1]


def load_inputs(config):
    completed_preparation(config)
    return load_federated_bundle(config['dataset']['bundle_path'])


def paired_reference(config,bundle):
    path=config['temporal']['reference_run_path']
    if path is None:
        if not bundle.testing_only:raise ValueError('Real-data temporal runs require the paired frozen FedAvg run')
        return None
    path=Path(path);baseline=load_training_config(path/'config.yaml')
    for key in ['training','initialization','selection']:
        if baseline[key]!=config[key]:raise ValueError(f'Unmatched baseline {key}')
    for key in ['rounds','participation_rate']:
        if baseline['federated'][key]!=config['federated'][key]:raise ValueError('Unmatched federation budget')
    for key in ['seed','device','deterministic']:
        if baseline['experiment'][key]!=config['experiment'][key]:raise ValueError('Unmatched baseline execution')
    if Path(baseline['dataset']['bundle_path'])!=bundle.path:raise ValueError('Unmatched data partition')
    meta=runtime_json((path/'metadata.json').read_text());verification=runtime_json((path/'verification.json').read_text())
    if (meta.get('workflow')!='prepared_fedavg' or meta['federated_inputs']['bundle_sha256']!=bundle.sha256
            or verification['test_scored'] or verification['rounds']!=config['federated']['rounds']):raise ValueError('Invalid paired clean FedAvg reference')
    return meta


@threadpool_limits.wrap(limits=1)
def preflight(config_path):
    torch.set_num_threads(1);start=perf_counter();cfg=load_temporal_config(config_path);bundle=load_inputs(cfg);paired_reference(cfg,bundle)
    result=dict(config_sha256=source_hash(config_path),bundle_sha256=bundle.sha256,
        client_rows=[len(p.data.dataset) for p in bundle.clients],validation_rows=len(bundle.validation.data.dataset),
        numerical_threadpools=threadpool_info(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,seconds=perf_counter()-start,
        training_performed=False,test_scored=False)
    if result['peak_process_rss_kib']>int(3.5*1024**2):raise MemoryError('CP4 preflight exceeded 3.5 GiB process guard')
    return result


@threadpool_limits.wrap(limits=1)
def run_temporal(config_path):
    torch.set_num_threads(1);cfg=load_temporal_config(config_path);e,f,t=(cfg[k] for k in ['experiment','federated','training'])
    bundle=load_inputs(cfg);baseline=paired_reference(cfg,bundle)
    seed_everything(e['seed'],e['deterministic']);model=create_model(bundle.contract.model_config())
    initial_hash=_state_sha256(model.state_dict())
    if baseline is not None and initial_hash!=baseline['initialization']['model_state_sha256']:raise ValueError('Unmatched initial model')
    factory=lambda:create_model(bundle.contract.model_config())
    clients=[FederatedClient(p.client_id,factory,p.data.dataset,t,e['device'],e['seed'],e['deterministic']) for p in bundle.clients]
    meta=environment_metadata(e['seed'],e['device'],e['deterministic'],REPOSITORY,testing_only=bundle.testing_only,purpose='Clean SecureDyn-inspired temporal-trust adaptation')
    meta.update(workflow='prepared_temporal_adapted',config_path=str(Path(config_path).resolve()),evaluation_role='validation',
        selection_policy='final_round',federated_inputs=bundle.source_metadata(),pytorch_threads=torch.get_num_threads(),
        optimizer_state='reset each round',sample_count_policy='retained training rows multiplied by temporal trust',
        initialization=dict(mode='random',model_state_sha256=initial_hash),temporal_adaptation=DEFAULTS,
        paired_fedavg_run=cfg['temporal']['reference_run_path'],test_scored=False,numerical_threadpools=threadpool_info())
    logger=ExperimentLogger(e['output_dir'],e['name'],cfg,meta,evaluation_role='validation')
    aggregator=TemporalTrustAggregator(diagnostic_dir=logger.path/'temporal_diagnostics')
    loader=DataLoader(bundle.validation.data.dataset,batch_size=t['batch_size'],shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(e['seed']))
    server=FederatedServer(model,clients,aggregator,loader,bundle.contract.num_classes,f['participation_rate'],e['seed'],e['device'],logger)
    start=perf_counter();server.run(f['rounds'])
    artifact=save_model_artifact(server.model,logger.path/'model_artifact.pt',contract=bundle.contract,model_name=e['name'],training_completed=True,testing_only=bundle.testing_only)
    _json(logger.path/'run_summary.json',dict(schema_version=1,status='completed',workflow='prepared_temporal_adapted',testing_only=bundle.testing_only,
        training_completed=True,selection_policy='final_round',selected_round=server.current_round,evaluation_role='validation',
        model_artifact=artifact.name,model_artifact_sha256=source_hash(artifact),resume_supported=False,test_scored=False,execution_duration_seconds=perf_counter()-start))
    return logger.path


@threadpool_limits.wrap(limits=1)
def run_verified(config_path):
    from Support.temporal_verification import verify_run
    start=perf_counter();path=run_temporal(config_path);verify_run(path,started=start)
    print(json.dumps(dict(run_completed=str(path))),flush=True);return path


def generate_configs(directory,results):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    references=runtime_json((REPOSITORY/'Support/docs/cp2_evidence/verification.json').read_text())['primary_runs']
    for condition in ['iid','dirichlet']:
        for seed in [42,43,44]:
            name=f'{condition}_seed{seed}';v=load_training_config(REPOSITORY/f'Support/configs/cp2_v1/{name}.yaml')
            reference=next(r['run_path'] for r in references if r['condition']==condition and r['seed']==seed)
            v['experiment']['name']='cp4_'+name;v['experiment']['output_dir']=os.path.relpath(Path(results).resolve(),directory)
            v['dataset']['bundle_path']=os.path.relpath(v['dataset']['bundle_path'],directory);v['federated']['aggregation']='temporal'
            v['temporal']={**DEFAULTS,'reference_run_path':os.path.relpath(REPOSITORY/reference,directory)}
            path=directory/f'{name}.yaml';path.write_text(yaml.safe_dump(v,sort_keys=False));load_temporal_config(path)


def run_matrix(config_dir,output_dir):
    directory=Path(config_dir).resolve();output=Path(output_dir).resolve();evidence=directory/'preflight.json'
    if not evidence.is_file():raise ValueError('Measured CP4 preflight required')
    measured=runtime_json(evidence.read_text());first=load_temporal_config(directory/'iid_seed42.yaml')
    if (measured['config_sha256']!=source_hash(directory/'iid_seed42.yaml') or measured['bundle_sha256']!=source_hash(first['dataset']['bundle_path'])
            or measured['peak_process_rss_kib']>int(3.5*1024**2) or measured['training_performed']):raise ValueError('Stale/invalid preflight')
    output.mkdir(parents=True,exist_ok=False);entries=[]
    for condition in ['iid','dirichlet']:
        for seed in [42,43,44]:
            config=directory/f'{condition}_seed{seed}.yaml';v=load_temporal_config(config);root=Path(v['experiment']['output_dir']);before=set(root.glob(v['experiment']['name']+'_*'))
            print(f'Starting temporal trust {condition} seed {seed}',flush=True)
            with (output/f'{condition}_seed{seed}.log').open('x') as stream:
                subprocess.run([sys.executable,'-m','Support.temporal_runner','train-one','--config',str(config)],
                    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'},stdout=stream,stderr=subprocess.STDOUT,check=True)
            fresh=set(root.glob(v['experiment']['name']+'_*'))-before
            if len(fresh)!=1:raise RuntimeError('Ambiguous CP4 run identity')
            path=fresh.pop()
            if not (path/'verification.json').is_file():raise RuntimeError('Missing verification')
            entries.append(dict(condition=condition,seed=seed,run_path=str(path)))
            (output/'progress.json').write_text(json.dumps(entries,indent=2)+'\n');print(f'Completed {condition} seed {seed}: {path}',flush=True)
    from Support.research_runner import summarize_matrix
    return summarize_matrix(entries,output)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for name in ['train-one','preflight']:
        a=sub.add_parser(name);a.add_argument('--config',type=Path,required=True)
    a=sub.add_parser('configs');a.add_argument('--directory',type=Path,required=True);a.add_argument('--results',type=Path,required=True)
    a=sub.add_parser('matrix');a.add_argument('--config-dir',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.action=='train-one':run_verified(args.config)
    elif args.action=='preflight':print(json.dumps(preflight(args.config),indent=2))
    elif args.action=='configs':generate_configs(args.directory,args.results)
    else:run_matrix(args.config_dir,args.output)


if __name__=='__main__':main()
