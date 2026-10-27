"""Strict YAML settings for future prepared-data prediction/evaluation runs.

Version 1 requires exactly the settings shown below. All paths are literal;
relative paths resolve against the YAML file's directory, and absolute paths
are accepted. Environment variables and '~' are not expanded.

    schema_version: 1
    mode: evaluate
    experiment:
      name: prepared_evaluation
      seed: 42
      device: cpu
      deterministic: true
      output_dir: ../results
    model:
      artifact_path: ../results/checkpoints/model.pt
    dataset:
      manifest_path: ../../data/processed/test_manifest.json
      expected_role: test
    inference:
      batch_size: 256

This reader does not open model/data artifacts, check CUDA availability, create
outputs, seed random generators, or execute a workflow. Those are runner tasks.
Model architecture and feature/class meanings belong to the artifacts' contracts.
Testing-only status belongs to the model/data declarations; YAML cannot override
it. Declared completed training and false testing-only flags do not establish
research validity. The existing Milestone A configuration API is unchanged.
"""

from pathlib import Path
import re

import yaml


class _PreparedConfigLoader(yaml.SafeLoader):
    """Use safe YAML types with explicit, unique string mapping keys."""


def _unique_mapping(loader, node):
    if not isinstance(node, yaml.MappingNode):
        raise ValueError('Prepared configuration mappings must contain key/value pairs')
    values = {}
    for key_node, value_node in node.value:
        if key_node.tag == 'tag:yaml.org,2002:merge':
            raise ValueError('YAML merge keys are unsupported; provide each setting explicitly')
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            raise ValueError('Prepared configuration keys must be strings')
        if key in values:
            raise ValueError(f'Duplicate prepared configuration field: {key}')
        values[key] = loader.construct_object(value_node, deep=True)
    return values


_PreparedConfigLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping,
)


def _path_setting(value, field):
    if (not isinstance(value, str) or not value.strip()
            or value != value.strip() or '\x00' in value):
        raise ValueError(f'{field} must be a nonblank path without surrounding whitespace or NUL')


def validate_prepared_config(config: dict) -> dict:
    """Validate version 1 settings and return a fresh dictionary of scalars.

    No values are inferred or coerced. This function preserves path strings;
    load_prepared_config() supplies the YAML directory for path resolution.
    All supported roles can be explicitly requested for diagnostics, including
    labeled 'predict' data for evaluation. The data loader/evaluator must check
    the actual role, labels, contracts, and artifact contents later.

    Raises ValueError for missing/unknown sections or fields, unsupported mode,
    version, device or role, and invalid scalar values. Input is never mutated.
    """
    sections = {
        'experiment': {'name', 'seed', 'device', 'deterministic', 'output_dir'},
        'model': {'artifact_path'},
        'dataset': {'manifest_path', 'expected_role'},
        'inference': {'batch_size'},
    }
    required = {'schema_version', 'mode'} | set(sections)
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('Prepared configuration fields must exactly match the version 1 schema')
    if type(config['schema_version']) is not int or config['schema_version'] != 1:
        raise ValueError('Only prepared configuration schema version 1 is supported')
    if config['mode'] not in ('predict', 'evaluate'):
        raise ValueError('Prepared mode must be predict or evaluate')
    for section, fields in sections.items():
        if not isinstance(config[section], dict) or set(config[section]) != fields:
            raise ValueError(f'Invalid or missing prepared settings in {section}')

    experiment = config['experiment']
    name = experiment['name']
    if not isinstance(name, str) or re.fullmatch(r'[A-Za-z0-9_-]+', name) is None:
        raise ValueError('Experiment name must use letters, numbers, underscores or hyphens')
    seed = experiment['seed']
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Experiment seed must be an integer from 0 to 2**32 - 1')
    if experiment['device'] not in ('cpu', 'cuda'):
        raise ValueError('Prepared device must be cpu or cuda')
    if type(experiment['deterministic']) is not bool:
        raise ValueError('Experiment deterministic must be an explicit boolean')
    if config['dataset']['expected_role'] not in ('train', 'validation', 'auxiliary', 'test', 'predict'):
        raise ValueError('dataset.expected_role must be a supported data role')
    batch_size = config['inference']['batch_size']
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('inference.batch_size must be a positive integer')
    for section, field in (
        ('experiment', 'output_dir'), ('model', 'artifact_path'), ('dataset', 'manifest_path'),
    ):
        _path_setting(config[section][field], f'{section}.{field}')

    # All accepted leaf values are immutable scalars. Copy each section so
    # callers can edit returned settings without changing the supplied mapping.
    return {
        'schema_version': config['schema_version'],
        'mode': config['mode'],
        **{section: dict(config[section]) for section in sections},
    }


def load_prepared_config(path: str | Path) -> dict:
    """Read one YAML document and return validated settings with absolute paths.

    Output/model/manifest paths are normalized without checking their existence.
    This allows configuration before real artifacts are supplied. The future
    runner must still call load_model_artifact() and load_prepared_data(). It
    must retain testing-only status if either artifact declares testing-only use.

    Raises:
        FileNotFoundError: The configuration path is missing or is a directory.
        ValueError: YAML is malformed or has duplicate/non-string keys, merge
            keys, unsafe tags, multiple documents, or invalid settings.
        OSError or UnicodeError: The configuration cannot be read as UTF-8.

    The returned dictionaries contain ordinary scalars, including string paths,
    and can be saved by the existing YAML experiment logger. No files are written.
    """
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'Prepared configuration file not found: {config_path}')
    try:
        description = yaml.load(config_path.read_text(encoding='utf-8'), Loader=_PreparedConfigLoader)
    except yaml.YAMLError as error:
        raise ValueError(f'Invalid prepared configuration YAML: {error}') from error
    config = validate_prepared_config(description)
    for section, field in (
        ('experiment', 'output_dir'), ('model', 'artifact_path'), ('dataset', 'manifest_path'),
    ):
        candidate = Path(config[section][field])
        if not candidate.is_absolute():
            candidate = config_path.parent / candidate
        config[section][field] = str(candidate.resolve())
    return config
