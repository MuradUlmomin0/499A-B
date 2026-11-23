"""Approved targeted local-label poisoning for three frozen aggregation methods."""
from Support.Utils.runtime_paths import load_json as runtime_json, load_json_file as runtime_json_file, load_yaml as runtime_yaml, safe_load_yaml as runtime_safe_yaml
from collections import OrderedDict
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits,threadpool_info

from Support.attacks.label_poison import PoisonedClient,select_malicious,original_labels
from Support.data.federated_bundle import load_federated_bundle
from Support.data.latest_audit import source_hash
from Support.data.latest_pilot import _json
from Support.federated.aggregation import FedAvgAggregator
from Support.federated.client import FederatedClient
from Support.federated.server import FederatedServer
from Support.federated.temporal import TemporalTrustAggregator
from Support.federated.fedmade import FedMADEAggregator
from Support.federated_runner import _state_sha256
from Support.models.factory import create_model
from Support.models.artifact import save_model_artifact
from Support.research_runner import completed_preparation
from Support.Utils.poisoning_config import load_poisoning_config,public_config,LOADERS
from Support.Utils.parameters import clone_state
from Support.Utils.reproducibility import seed_everything,environment_metadata
from Support.Utils.experiment_logger import ExperimentLogger

REPOSITORY=Path(__file__).resolve().parents[1]
WORKFLOWS=dict(fedavg='prepared_fedavg',fedmade='prepared_fedmade_adapted',temporal='prepared_temporal_adapted')


def load_inputs(config):
    base=config['_baseline'];auxiliary=None
    if config['method']=='fedmade':
        from Support.fedmade_runner import load_inputs as load_fedmade_inputs
        bundle,auxiliary,_,_=load_fedmade_inputs(base)
    else:
        completed_preparation(base);bundle=load_federated_bundle(base['dataset']['bundle_path'])
    a=config['attack'];classes=bundle.contract.num_classes
    if max(a['target_class'],a['destination_class'])>=classes or a['malicious_clients']>len(bundle.clients):raise ValueError('Attack exceeds label space/client population')
    if not bundle.testing_only:
        expected=dict(type='targeted_label',enabled=a['enabled'],target_class=18,destination_class=1,malicious_clients=2 if a['enabled'] else 0,start_round=4,end_round=20,selection_seed_offset=5005)
        if a!=expected or len(bundle.clients)!=10 or classes!=34 or bundle.contract.class_names[18]!='DoS-HTTP_Flood' or bundle.contract.class_names[1]!='BenignTraffic':raise ValueError('Real-data settings differ from approved CP5 protocol')
    return bundle,auxiliary


def paired_reference(config,bundle):
    path=config['reference_clean_run_path'];base=config['_baseline']
    if path is None:
        if not bundle.testing_only:raise ValueError('Real poisoning requires a frozen clean reference')
        return None
    path=Path(path);reference=LOADERS[config['method']](path/'config.yaml')
    if reference!=base:raise ValueError('Clean reference configuration does not match frozen baseline')
    meta=runtime_json((path/'metadata.json').read_text());v=runtime_json((path/'verification.json').read_text())
    if (meta.get('workflow')!=WORKFLOWS[config['method']] or meta.get('evaluation_role')!='validation'
            or meta['federated_inputs']!=bundle.source_metadata() or v['test_scored'] or v['testing_only']!=bundle.testing_only
            or not v['aggregation_verified'] or not v['reloaded_validation_exact']):raise ValueError('Invalid or unverified paired clean reference')
    return meta


class EvidenceAggregator:
    """Save ordinary results only; no attacker identity is passed to the defense."""
    def __init__(self,delegate,directory):
        self.delegate=delegate;self.directory=Path(directory);self.directory.mkdir(exist_ok=False)

    def aggregate(self,results):
        rnd=results[0].round_id
        evidence=dict(client_order=[r.client_id for r in results],client_states={r.client_id:clone_state(r.state) for r in results})
        state,weights=self.delegate.aggregate(results)
        evidence.update(aggregated_state=clone_state(state),weights=weights)
        with (self.directory/f'round_{rnd:03d}.pt').open('xb') as f:torch.save(evidence,f)
        return state,weights


def make_aggregator(config,bundle,auxiliary,factory,path):
    base=config['_baseline'];method=config['method']
    if method=='fedavg':return FedAvgAggregator()
    if method=='temporal':return TemporalTrustAggregator(diagnostic_dir=path/'temporal_diagnostics')
    m=base['fedmade']
    return FedMADEAggregator(factory,auxiliary.dataset,bundle.contract.num_classes,eps=m['eps'],batch_size=base['training']['batch_size'],
        maxiter=m['maxiter'],device='cpu',diagnostic_dir=path/'fedmade_diagnostics')


@threadpool_limits.wrap(limits=1)
def preflight(config_path):
    torch.set_num_threads(1);config=load_poisoning_config(config_path);bundle,auxiliary=load_inputs(config);paired_reference(config,bundle)
    a=config['attack'];ids=select_malicious([p.client_id for p in bundle.clients],config['_baseline']['experiment']['seed'],a['malicious_clients'])
    counts={p.client_id:int(np.count_nonzero(original_labels(p.data.dataset)==a['target_class'])) for p in bundle.clients if p.client_id in ids}
    if a['enabled'] and any(n==0 for n in counts.values()):raise ValueError('Selected attacker has no target examples; do not silently replace it')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss>int(3.5*1024**2):raise MemoryError('Poisoning preflight exceeded 3.5 GiB process guard')
    return dict(config_sha256=source_hash(config_path),bundle_sha256=bundle.sha256,selected_clients=ids,target_counts=counts,
        validation_rows=len(bundle.validation.data.dataset),peak_process_rss_kib=rss,numerical_threadpools=threadpool_info(),training_performed=False,test_scored=False)


@threadpool_limits.wrap(limits=1)
def run_poisoning(config_path):
    torch.set_num_threads(1);config=load_poisoning_config(config_path);base=config['_baseline'];e=base['experiment'];t=base['training'];f=base['federated']
    bundle,auxiliary=load_inputs(config);reference=paired_reference(config,bundle)
    selected=select_malicious([p.client_id for p in bundle.clients],e['seed'],config['attack']['malicious_clients'])
    target_counts={p.client_id:int(np.count_nonzero(original_labels(p.data.dataset)==config['attack']['target_class'])) for p in bundle.clients}
    if config['attack']['enabled'] and any(target_counts[c]==0 for c in selected):raise ValueError('Selected attacker has no target examples')
    seed_everything(e['seed'],e['deterministic']);factory=lambda:create_model(bundle.contract.model_config());model=factory()
    initial_hash=_state_sha256(model.state_dict())
    if reference is not None and initial_hash!=reference['initialization']['model_state_sha256']:raise ValueError('Unmatched initialization')
    # Construct ordinary clients before the strategy, exactly as the clean runners do.
    ordinary=[FederatedClient(p.client_id,factory,p.data.dataset,t,'cpu',e['seed'],e['deterministic']) for p in bundle.clients]
    meta=environment_metadata(e['seed'],'cpu',e['deterministic'],REPOSITORY,testing_only=bundle.testing_only,purpose='Approved targeted label-poisoning evaluation; not stealth')
    meta.update(workflow='prepared_poisoning',method=config['method'],evaluation_role='validation',selection_policy='final_round',
        federated_inputs=bundle.source_metadata(),initialization=dict(mode='random',model_state_sha256=initial_hash),
        baseline_config_sha256=source_hash(config['baseline_config_path']),paired_clean_run=config['reference_clean_run_path'],
        paired_clean_config_sha256=source_hash(Path(config['reference_clean_run_path'])/'config.yaml') if reference else None,
        numerical_threadpools=threadpool_info(),pytorch_threads=torch.get_num_threads(),test_scored=False)
    if auxiliary is not None:
        meta['auxiliary']=dict(manifest_path=str(auxiliary.manifest.manifest_path),manifest_sha256=source_hash(auxiliary.manifest.manifest_path),
            data_path=str(auxiliary.manifest.data_path),data_sha256=source_hash(auxiliary.manifest.data_path),evidence_path=base['fedmade']['auxiliary_evidence_path'],evidence_sha256=source_hash(base['fedmade']['auxiliary_evidence_path']))
    logger=ExperimentLogger(config['experiment']['output_dir'],config['experiment']['name'],public_config(config),meta,evaluation_role='validation')
    clients=[PoisonedClient(c,config['attack'],c.client_id in selected,bundle.contract.num_classes,logger.path/'label_audits') for c in ordinary]
    defense=make_aggregator(config,bundle,auxiliary,factory,logger.path)
    aggregator=EvidenceAggregator(defense,logger.path/'aggregation_evidence')
    loader=DataLoader(bundle.validation.data.dataset,batch_size=t['batch_size'],shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(e['seed']))
    server=FederatedServer(model,clients,aggregator,loader,bundle.contract.num_classes,f['participation_rate'],e['seed'],'cpu',logger)
    start=perf_counter();server.run(f['rounds'])
    artifact=save_model_artifact(server.model,logger.path/'model_artifact.pt',contract=bundle.contract,model_name=config['experiment']['name'],training_completed=True,testing_only=bundle.testing_only)
    _json(logger.path/'attack_ground_truth.json',dict(selected_malicious=selected,attack=config['attack'],events=sorted([r for c in clients for r in c.events],key=lambda r:(r['round_id'],r['client_id']))))
    _json(logger.path/'run_summary.json',dict(status='completed',workflow='prepared_poisoning',method=config['method'],selected_round=server.current_round,
        testing_only=bundle.testing_only,selection_policy='final_round',evaluation_role='validation',test_scored=False,model_artifact_sha256=source_hash(artifact),
        resume_supported=False,execution_duration_seconds=perf_counter()-start))
    return logger.path


@threadpool_limits.wrap(limits=1)
def run_verified(config_path):
    from Support.poisoning_verification import verify_run
    start=perf_counter();path=run_poisoning(config_path);verify_run(path,started=start)
    print(json.dumps(dict(run_completed=str(path))),flush=True);return path
