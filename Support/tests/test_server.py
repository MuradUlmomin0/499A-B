import torch
from torch.utils.data import DataLoader
from support.data.dummy import make_dummy_data
from support.models.factory import create_model
from support.federated.client import FederatedClient
from support.federated.server import FederatedServer
from support.federated.aggregation import FedAvgAggregator
from support.utils.reproducibility import seed_everything


def build_server(config, evaluation_labels=None):
    seed_everything(config['experiment']['seed'])
    torch.set_num_threads(1)
    m=config['model']; d=config['dataset']
    datasets,evaluation=make_dummy_data(m['input_dim'],m['num_classes'],d['client_samples'],d['evaluation_samples'],42)
    if evaluation_labels is not None:
        evaluation.tensors[1].fill_(evaluation_labels)
    factory=lambda:create_model(m)
    global_model=factory()
    clients=[FederatedClient(str(i),factory,data,config['training']) for i,data in enumerate(datasets)]
    return FederatedServer(global_model,clients,FedAvgAggregator(),DataLoader(evaluation,batch_size=8),m['num_classes'],config['federated']['participation_rate'],42)


def test_selection_and_history(config):
    config['federated']['participation_rate']=0.5
    a,b=build_server(config),build_server(config)
    assert [c.client_id for c in a.select_clients(1)]==[c.client_id for c in b.select_clients(1)]
    assert len(a.select_clients(1))==2
    a.run(2)
    for record in a.round_records:
        for client in record['selected_clients']:
            assert record['round_id'] in a.history.records[client]
    assert sum(len(v) for v in a.history.records.values())==4


def test_test_labels_cannot_change_training(config):
    a=build_server(config,evaluation_labels=0); a.run(2)
    b=build_server(config,evaluation_labels=1); b.run(2)
    assert all(torch.equal(a.model.state_dict()[k],b.model.state_dict()[k]) for k in a.model.state_dict())
