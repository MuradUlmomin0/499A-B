"""Prepared-data FedAvg using existing clients/server and final-round export.

Unlike prepared_runner.py this callable performs local optimization when invoked
normally. Infrastructure tests replace Client.train with controlled artificial
updates. No final-test or auxiliary data is accepted by this training interface.
"""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from support.data.federated_bundle import load_federated_bundle
from support.data.split_checks import file_sha256
from support.federated.aggregation import FedAvgAggregator
from support.federated.client import FederatedClient
from support.federated.server import FederatedServer
from support.models.artifact import load_model_artifact, save_model_artifact
from support.models.factory import create_model
from support.utils.experiment_logger import ExperimentLogger
from support.utils.parameters import clone_state
from support.utils.reproducibility import environment_metadata, seed_everything
from support.utils.training_config import load_training_config

REPOSITORY = Path(__file__).resolve().parents[1]


def _state_sha256(state) -> str:
    """Identify ordered parameter names, shapes, dtypes and exact CPU bytes."""
    digest = hashlib.sha256()
    for name, value in clone_state(state).items():
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_federated(config_path: str | Path) -> Path:
    """Train configured prepared partitions and export the final global model.

    Checks configuration/bundle/warm-start compatibility before seeding, model
    execution or output creation. Seeds and deterministic settings are process-wide
    and deliberately not restored. Random initialization uses one shared global
    state, broadcast by the existing server before every local training call.

    Validation is diagnostic only; the approved export policy is final_round.
    Test data, model selection by test scores, raw-data preprocessing, attacks and
    research defenses are outside this callable. A completed protocol/export and
    producer flags do not establish provenance or scientifically valid results.
    An I/O/execution failure may leave diagnostic logs without a completed summary.
    """
    source = Path(config_path).resolve()
    config = load_training_config(source)
    e, f, t = (config[k] for k in ('experiment', 'federated', 'training'))
    if e['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested for prepared FedAvg but unavailable')
    bundle = load_federated_bundle(config['dataset']['bundle_path'])
    init = config['initialization']
    warm = load_model_artifact(init['artifact_path']) if init['mode'] == 'artifact' else None
    if warm is not None:
        warm.contract.assert_compatible(bundle.contract)
    testing_only = bundle.testing_only or (warm.testing_only if warm is not None else False)

    seed_everything(e['seed'], e['deterministic'])
    model = warm.model if warm is not None else create_model(bundle.contract.model_config())
    model.to(e['device'])
    initial_hash = _state_sha256(model.state_dict())
    factory = lambda: create_model(bundle.contract.model_config())
    clients = [FederatedClient(p.client_id, factory, p.data.dataset, t, e['device'], e['seed'], e['deterministic'])
               for p in bundle.clients]
    metadata = environment_metadata(
        e['seed'], e['device'], e['deterministic'], REPOSITORY,
        testing_only=testing_only,
        purpose='Prepared-data FedAvg training' + ('; testing-only verification' if testing_only else ''),
    )
    metadata.update({
        'workflow': 'prepared_fedavg', 'config_path': str(source), 'evaluation_role': 'validation',
        'selection_policy': 'final_round', 'federated_inputs': bundle.source_metadata(),
        'pytorch_threads': torch.get_num_threads(), 'optimizer_state': 'reset each round',
        'sample_count_policy': 'length of prepared training partition, not epoch observations',
        'initialization': {'mode': init['mode'], 'model_state_sha256': initial_hash,
                           'artifact_path': str(warm.artifact_path) if warm else None,
                           'artifact_sha256': warm.artifact_sha256 if warm else None},
    })
    logger = ExperimentLogger(e['output_dir'], e['name'], config, metadata, evaluation_role='validation')
    validation_loader = DataLoader(
        bundle.validation.data.dataset, batch_size=t['batch_size'], shuffle=False,
        num_workers=0, generator=torch.Generator().manual_seed(e['seed']),
    )
    server = FederatedServer(model, clients, FedAvgAggregator(), validation_loader,
                             bundle.contract.num_classes, f['participation_rate'], e['seed'], e['device'], logger)
    started = perf_counter()
    server.run(f['rounds'])
    artifact = save_model_artifact(
        server.model, logger.path / 'model_artifact.pt', contract=bundle.contract,
        model_name=e['name'], training_completed=True, testing_only=testing_only,
    )
    summary = {
        'schema_version': 1, 'status': 'completed', 'workflow': 'prepared_fedavg',
        'testing_only': testing_only, 'training_completed': True,
        'selection_policy': 'final_round', 'selected_round': server.current_round,
        'evaluation_role': 'validation', 'model_artifact': artifact.name,
        'model_artifact_sha256': file_sha256(artifact), 'resume_supported': False,
        'execution_duration_seconds': perf_counter() - started,
    }
    serialized = json.dumps(summary, indent=2, allow_nan=False) + '\n'
    with (logger.path / 'run_summary.json').open('x', encoding='utf-8') as stream:
        stream.write(serialized)
    return logger.path
