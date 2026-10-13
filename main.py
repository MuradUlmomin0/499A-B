"""Milestone A entry point. Synthetic execution requires --dummy-test."""
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

REPOSITORY = Path(__file__).resolve().parent


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
    parser = argparse.ArgumentParser(description='PRIVATE SHIELD Milestone A framework test')
    parser.add_argument('--config',type=Path,default=REPOSITORY/'support/configs/fedavg_dummy_test.yaml')
    parser.add_argument('--dummy-test',action='store_true',help='Explicitly run synthetic framework verification only')
    args = parser.parse_args()
    if not args.dummy_test:
        parser.error('Use --dummy-test. Real dataset execution is not implemented yet.')
    config = load_config(args.config)
    path,_ = run_dummy_test(config)
    print('Synthetic FedAvg framework test completed. Metrics are NOT research results.')
    print(f'Test artifacts: {path}')


if __name__ == '__main__':
    main()
