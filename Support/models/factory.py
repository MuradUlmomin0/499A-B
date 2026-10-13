from .mlp import MLP


def create_model(config):
    if config['type'] != 'mlp':
        raise ValueError('Unsupported model')
    return MLP(config['input_dim'], config['num_classes'], config['hidden_dims'])
