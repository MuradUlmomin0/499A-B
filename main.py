"""Command-line entry point for preparation, dummy FedAvg and prepared models.

--dummy-test opts into synthetic federated framework verification only.
--prepared requires an explicit --config YAML whose mode selects saved-model
prediction or evaluation. The prepared workflow does not train or preprocess.
--federated requires explicit prepared-FedAvg YAML and performs local training.
--prepare-data retains the retired 39-feature source adapter for reference.
--stacking explicitly recovers/validates the supplied notebook model; diagnostic only.
"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from support.utils.config import load_config
from support.utils.reproducibility import seed_everything, environment_metadata
from support.utils.experiment_logger import ExperimentLogger
from support.models.factory import create_model
from support.data.dummy import make_dummy_data
from support.federated.client import FederatedClient
from support.federated.server import FederatedServer
from support.federated.aggregation import FedAvgAggregator
from support.prepared_runner import run_prepared
from support.federated_runner import run_federated

REPOSITORY = Path(__file__).resolve().parent


def run_stacking(config_path):
    """Keep the supplied notebook model separate from MLP/FedAvg contracts."""
    from support.stacking_runner import run_stacking as stacking
    return stacking(config_path)


def run_preparation(config_path):
    """Import the raw-data adapter only when preparation is explicitly selected."""
    from support.prepare_runner import run_preparation as prepare
    return prepare(config_path)


def run_dummy_test(config):
    e,m,f,t,d = [config[k] for k in ('experiment','model','federated','training','dataset')]
    if e['device']=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    seed_everything(e['seed'],e['deterministic'])
    # One CPU thread keeps tiny framework tests inexpensive and repeatable.
    torch.set_num_threads(1)
    datasets,evaluation = make_dummy_data(m['input_dim'],m['num_classes'],d['client_samples'],d['evaluation_samples'],e['seed'])
    factory = lambda: create_model(m)
    global_model = factory()
    clients = [FederatedClient(str(i),factory,data,t,e['device'],e['seed'],e['deterministic']) for i,data in enumerate(datasets)]
    metadata = environment_metadata(e['seed'],e['device'],e['deterministic'],REPOSITORY)
    metadata['pytorch_threads'] = torch.get_num_threads()
    metadata['client_selection'] = 'uniform without replacement; ceil(rate * clients)'
    metadata['optimizer_state'] = 'reset each round'
    metadata['sample_count'] = 'local training dataset length, not epoch observations'
    output = Path(e['output_dir'])
    if not output.is_absolute():
        output = REPOSITORY/output
    logger = ExperimentLogger(output,e['name'],config,metadata)
    server = FederatedServer(global_model,clients,FedAvgAggregator(),
              DataLoader(evaluation,batch_size=t['batch_size'],shuffle=False),m['num_classes'],
              f['participation_rate'],e['seed'],e['device'],logger)
    metrics = server.run(f['rounds'])
    return logger.path,metrics


def main():
    """Select one explicit workflow and print its completed output directory."""
    parser = argparse.ArgumentParser(
        description='PRIVATE SHIELD: data preparation, dummy verification, prepared prediction/evaluation or FedAvg training',
    )
    parser.add_argument(
        '--config', type=Path,
        help='YAML path; required for preparation, prepared execution or federated training; dummy has a bundled default',
    )
    workflow = parser.add_mutually_exclusive_group(required=True)
    workflow.add_argument(
        '--dummy-test', action='store_true',
        help='Explicitly run synthetic federated framework verification only',
    )
    workflow.add_argument(
        '--prepared', action='store_true',
        help='Use --config for saved-model prediction or evaluation without training',
    )
    workflow.add_argument(
        '--federated', action='store_true',
        help='Train prepared client partitions with FedAvg and round validation; requires --config',
    )
    workflow.add_argument(
        '--prepare-data', action='store_true',
        help='Reference adapter for the retired 39-feature dataset, without model training; requires --config',
    )
    workflow.add_argument(
        '--stacking', action='store_true',
        help='Recover or validate the supplied stacking model without training; diagnostic only; requires --config',
    )
    args = parser.parse_args()
    if args.stacking:
        if args.config is None:
            parser.error('--stacking requires --config with an explicit recovery/validation YAML')
        path = run_stacking(args.config)
        print('Stacking diagnostic completed. Reconstructed preprocessing; NOT research results.')
        print(f'Run artifacts: {path}')
        return
    if args.prepare_data:
        if args.config is None:
            parser.error('--prepare-data requires --config with a preparation YAML')
        path = run_preparation(args.config)
        print('Data preparation completed (diagnostic only; no model training).')
        print(f'Prepared artifacts: {path}')
        return
    if args.federated:
        if args.config is None:
            parser.error('--federated requires --config with a prepared-FedAvg YAML')
        path = run_federated(args.config)
        print('Prepared FedAvg run completed (round evaluation: validation).')
        print(f'Run artifacts: {path}')
        return
    if args.prepared:
        if args.config is None:
            parser.error('--prepared requires --config with a prepared-run YAML')
        path = run_prepared(args.config)
        print('Prepared-data run completed.')
        print(f'Run artifacts: {path}')
        return

    config_path = args.config if args.config is not None else REPOSITORY/'support/configs/fedavg_dummy_test.yaml'
    config = load_config(config_path)
    path,_ = run_dummy_test(config)
    print('Synthetic FedAvg framework test completed. Metrics are NOT research results.')
    print(f'Test artifacts: {path}')


if __name__ == '__main__':
    main()
