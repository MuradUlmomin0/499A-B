"""Strict configuration for the testing-only entry point."""
from pathlib import Path
import math
import yaml


def validate_config(config):
    required = {
        'experiment': {'name', 'seed', 'device', 'deterministic', 'output_dir'},
        'model': {'type', 'input_dim', 'num_classes', 'hidden_dims'},
        'federated': {'num_clients', 'rounds', 'participation_rate', 'aggregation'},
        'training': {'local_epochs', 'batch_size', 'optimizer', 'learning_rate'},
        'dataset': {'type', 'client_samples', 'evaluation_samples'},
        'attack': {'enabled'},
    }
    if not isinstance(config, dict) or set(config) != set(required):
        raise ValueError('Configuration sections must match the documented schema')
    for section, keys in required.items():
        if not isinstance(config[section], dict) or set(config[section]) != keys:
            raise ValueError(f'Invalid or missing settings in {section}')
    def integer(value, minimum=1):
        if type(value) is not int or value < minimum:
            raise ValueError(f'Expected integer >= {minimum}, got {value!r}')
    e, m, f, t, d = [config[k] for k in ('experiment','model','federated','training','dataset')]
    if not isinstance(e['name'], str) or not e['name'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in e['name']):
        raise ValueError('Experiment name must use letters, numbers, underscores or hyphens')
    integer(e['seed'], 0)
    if e['seed'] >= 2**32:
        raise ValueError('Seed must be below 2**32')
    if e['device'] not in ('cpu', 'cuda') or type(e['deterministic']) is not bool:
        raise ValueError('Invalid device or deterministic setting')
    if not isinstance(e['output_dir'], str) or not e['output_dir'].strip():
        raise ValueError('output_dir must be a nonempty path')
    if m['type'] != 'mlp' or f['aggregation'] != 'fedavg' or d['type'] != 'dummy':
        raise ValueError('Milestone A CLI supports only MLP, FedAvg and testing-only dummy data')
    if config['attack']['enabled'] is not False:
        raise ValueError('Attacks are unsupported in Milestone A')
    for value in (m['input_dim'], f['num_clients'], f['rounds'], t['local_epochs'], t['batch_size'], d['evaluation_samples']):
        integer(value)
    integer(m['num_classes'], 2)
    if not isinstance(m['hidden_dims'], list) or not m['hidden_dims']:
        raise ValueError('hidden_dims must be a nonempty list')
    for value in m['hidden_dims']:
        integer(value)
    for value in (f['participation_rate'], t['learning_rate']):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Rates must be positive finite numbers')
    if f['participation_rate'] > 1 or t['optimizer'] not in ('adam','sgd'):
        raise ValueError('Invalid participation rate or optimizer')
    if not isinstance(d['client_samples'], list) or len(d['client_samples']) != f['num_clients']:
        raise ValueError('client_samples must have one count per client')
    for value in d['client_samples']:
        integer(value)
    return config


def load_config(path):
    with Path(path).open() as stream:
        return validate_config(yaml.safe_load(stream))
