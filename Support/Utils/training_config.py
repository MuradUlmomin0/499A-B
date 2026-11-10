"""Strict prepared-FedAvg settings; architecture comes from the bundle contract."""

import math
from pathlib import Path
import re

import yaml

from support.utils.prepared_config import _PreparedConfigLoader, _path_setting


def validate_training_config(config: dict) -> dict:
    """Validate explicit settings and return fresh sections without opening inputs."""
    sections = {
        'experiment': {'name', 'seed', 'device', 'deterministic', 'output_dir'},
        'dataset': {'bundle_path'}, 'initialization': {'mode', 'artifact_path'},
        'federated': {'aggregation', 'rounds', 'participation_rate'},
        'training': {'optimizer', 'learning_rate', 'local_epochs', 'batch_size'},
        'selection': {'policy'},
    }
    if not isinstance(config, dict) or set(config) != {'schema_version', *sections}:
        raise ValueError('Training configuration fields must exactly match the version 1 schema')
    if type(config['schema_version']) is not int or config['schema_version'] != 1:
        raise ValueError('Only training configuration schema version 1 is supported')
    for key, fields in sections.items():
        if not isinstance(config[key], dict) or set(config[key]) != fields:
            raise ValueError(f'Invalid or missing training settings in {key}')
    e, init, f, t = (config[k] for k in ('experiment', 'initialization', 'federated', 'training'))
    if not isinstance(e['name'], str) or re.fullmatch(r'[A-Za-z0-9_-]+', e['name']) is None:
        raise ValueError('Experiment name must use letters, numbers, underscores or hyphens')
    if type(e['seed']) is not int or not 0 <= e['seed'] < 2**32:
        raise ValueError('Seed must be an integer from 0 to 2**32 - 1')
    if e['device'] not in ('cpu', 'cuda') or type(e['deterministic']) is not bool:
        raise ValueError('Invalid device or deterministic setting')
    for section, field in [('experiment', 'output_dir'), ('dataset', 'bundle_path')]:
        _path_setting(config[section][field], f'{section}.{field}')
    if init['mode'] not in ('random', 'artifact'):
        raise ValueError('Initialization mode must be random or artifact')
    if init['mode'] == 'random':
        if init['artifact_path'] is not None:
            raise ValueError('Random initialization requires artifact_path: null')
    else:
        _path_setting(init['artifact_path'], 'initialization.artifact_path')
    if f['aggregation'] != 'fedavg' or t['optimizer'] not in ('adam', 'sgd'):
        raise ValueError('Only FedAvg with adam or sgd is supported')
    for value in (f['rounds'], t['local_epochs'], t['batch_size']):
        if type(value) is not int or value < 1:
            raise ValueError('Rounds, local epochs and batch size must be positive integers')
    for value in (f['participation_rate'], t['learning_rate']):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Learning/participation rates must be positive finite numbers')
    if f['participation_rate'] > 1:
        raise ValueError('participation_rate must not exceed one')
    if config['selection']['policy'] != 'final_round':
        raise ValueError('Only the approved final_round export policy is supported')
    return {'schema_version': 1, **{key: dict(config[key]) for key in sections}}


def load_training_config(path: str | Path) -> dict:
    """Read safe unique-key YAML; resolve paths against the YAML's directory.

    No artifacts, arrays, optimizers or output folders are opened/created.
    Paths are literal; '~' and environment substitutions are not expanded.
    """
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f'Training configuration file not found: {source}')
    try:
        value = yaml.load(source.read_text(encoding='utf-8'), Loader=_PreparedConfigLoader)
    except yaml.YAMLError as error:
        raise ValueError(f'Invalid training YAML: {error}') from error
    config = validate_training_config(value)
    for section, field in [('experiment', 'output_dir'), ('dataset', 'bundle_path'),
                           ('initialization', 'artifact_path')]:
        value = config[section][field]
        if value is not None:
            candidate = Path(value)
            config[section][field] = str((candidate if candidate.is_absolute() else source.parent / candidate).resolve())
    return config
