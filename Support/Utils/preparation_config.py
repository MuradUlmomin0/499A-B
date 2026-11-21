"""Explicit diagnostic preparation settings with conservative memory bounds."""
from pathlib import Path
import math
import re
import yaml
from Support.data.ciciot2023 import CLASS_NAMES
from Support.Utils.prepared_config import _PreparedConfigLoader, _path_setting


def load_preparation_config(path):
    """Read strict unique-key YAML; no source/model/data execution or outputs."""
    source = Path(path).resolve()
    try:
        value = yaml.load(source.read_text(encoding='utf-8'), Loader=_PreparedConfigLoader)
    except yaml.YAMLError as error:
        raise ValueError(f'Invalid preparation YAML: {error}') from error
    fields = {
        'source': {'archive_path', 'sha256', 'feature_names', 'expected_valid_counts'},
        'output': {'directory', 'work_dir'},
        'pilot': {'max_rows', 'chunk_rows', 'seed', 'num_clients', 'dirichlet_alpha'},
        'model': {'hidden_dims'},
    }
    if not isinstance(value, dict) or set(value) != {'schema_version', *fields} or type(value['schema_version']) is not int or value['schema_version'] != 1:
        raise ValueError('Preparation requires the exact version 1 schema')
    for key, names in fields.items():
        if not isinstance(value[key], dict) or set(value[key]) != names:
            raise ValueError(f'Invalid preparation section: {key}')
    s, p, m = value['source'], value['pilot'], value['model']
    if not isinstance(s['sha256'], str) or re.fullmatch('[a-f0-9]{64}', s['sha256']) is None:
        raise ValueError('Require the audited archive SHA-256')
    names = s['feature_names']
    if not isinstance(names, list) or not names or any(not isinstance(x, str) or not x.strip() or x != x.strip() for x in names) or len(set(names)) != len(names):
        raise ValueError('Require explicit unique ordered feature names')
    counts = s['expected_valid_counts']
    if not isinstance(counts, dict) or set(counts) != set(CLASS_NAMES) or any(type(n) is not int or n < 4 for n in counts.values()):
        raise ValueError('Require audited positive counts for all eight classes')
    for name, low, high in [('max_rows', 32, 500000), ('chunk_rows', 1, 25000),
                            ('seed', 0, 2**32-1), ('num_clients', 2, 10)]:
        if type(p[name]) is not int or not low <= p[name] <= high:
            raise ValueError(f'{name} outside diagnostic memory/safety bounds')
    if type(p['dirichlet_alpha']) not in (int, float) or not math.isfinite(p['dirichlet_alpha']) or p['dirichlet_alpha'] <= 0:
        raise ValueError('dirichlet_alpha must be finite and positive')
    if not isinstance(m['hidden_dims'], list) or not m['hidden_dims'] or any(type(n) is not int or n < 1 for n in m['hidden_dims']):
        raise ValueError('Invalid hidden_dims')
    if p['max_rows'] * len(names) * 4 > 128 * 1024**2 or p['chunk_rows'] * len(names) * 8 > 32 * 1024**2:
        raise ValueError('Feature buffers exceed the diagnostic memory budget')
    widths = [len(names), *m['hidden_dims'], len(CLASS_NAMES)]
    parameters = sum((a + 1) * b for a, b in zip(widths, widths[1:]))
    if parameters * (p['num_clients'] + 3) * 4 > 128 * 1024**2:
        raise ValueError('Configured model copies exceed the diagnostic memory budget')
    for section, field in [('source', 'archive_path'), ('output', 'directory'), ('output', 'work_dir')]:
        text = value[section][field]
        _path_setting(text, f'{section}.{field}')
        candidate = Path(text)
        value[section][field] = str((candidate if candidate.is_absolute() else source.parent/candidate).resolve())
    out, work = map(Path, [value['output']['directory'], value['output']['work_dir']])
    if out == work or out.is_relative_to(work) or work.is_relative_to(out):
        raise ValueError('Output and scratch roots must not contain one another')
    return value
