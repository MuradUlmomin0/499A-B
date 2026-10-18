"""Local training accepts any compatible labeled PyTorch dataset."""
import torch
from torch.utils.data import DataLoader
from support.utils.parameters import clone_state, extract_update, validate_state
from support.utils.reproducibility import seed_everything
from support.evaluation.metrics import evaluate_model
from .types import ClientResult


class FederatedClient:
    def __init__(self, client_id, model_factory, train_dataset, training, device='cpu', seed=42, deterministic=True, evaluation_dataset=None):
        if not len(train_dataset):
            raise ValueError('Client training dataset cannot be empty')
        self.client_id = str(client_id)
        self.model = model_factory().to(device)
        validate_state(self.model.state_dict())
        self.train_dataset = train_dataset
        self.evaluation_dataset = evaluation_dataset
        self.training, self.device, self.seed, self.deterministic = dict(training), device, seed, deterministic
        self.global_state = None

    def receive_global(self, state):
        validate_state(state,self.model.state_dict())
        self.global_state = clone_state(state)
        self.model.load_state_dict(self.global_state,strict=True)

    def train(self, round_id):
        if self.global_state is None:
            raise ValueError('Global parameters must be received before training')
        if type(round_id) is not int or round_id < 1:
            raise ValueError('round_id must be positive')
        # Stable client/round stream does not depend on participation ordering.
        import hashlib
        client_seed = int.from_bytes(hashlib.sha256(self.client_id.encode()).digest()[:4],'big')
        local_seed = (self.seed + client_seed + round_id * 1000003) % 2**32
        seed_everything(local_seed,self.deterministic)
        generator = torch.Generator().manual_seed(local_seed)
        loader = DataLoader(self.train_dataset,batch_size=self.training['batch_size'],shuffle=True,generator=generator,num_workers=0)
        # Reset local optimizer each communication round, including Adam moments.
        optimizer_class = {'adam':torch.optim.Adam,'sgd':torch.optim.SGD}[self.training['optimizer']]
        optimizer = optimizer_class(self.model.parameters(),lr=self.training['learning_rate'])
        self.model.train()
        total_loss, observations = 0.0, 0
        for _ in range(self.training['local_epochs']):
            for features,labels in loader:
                features,labels = features.to(self.device),labels.to(self.device)
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(self.model(features),labels)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite client training loss')
                loss.backward()
                optimizer.step()
                total_loss += loss.item()*len(labels)
                observations += len(labels)
        state = clone_state(self.model.state_dict())
        return ClientResult(self.client_id,round_id,len(self.train_dataset),state,
                            extract_update(state,self.global_state),{'train_loss':total_loss/observations})

    def evaluate(self, num_classes):
        if self.evaluation_dataset is None:
            raise ValueError('No local evaluation dataset provided')
        loader = DataLoader(self.evaluation_dataset,batch_size=self.training['batch_size'],shuffle=False)
        return evaluate_model(self.model,loader,self.device,num_classes)
