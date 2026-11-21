"""Strict adapted-FedMADE configuration without relaxing FedAvg's schema."""
import math
from pathlib import Path
import yaml
from Support.Utils.training_config import validate_training_config
from Support.Utils.prepared_config import _PreparedConfigLoader, _path_setting


METHOD_FIELDS={'auxiliary_manifest_path','auxiliary_evidence_path','reference_run_path',
               'eps','min_samples','solver','norm','maxiter'}


def validate_fedmade_config(value):
    if not isinstance(value,dict) or 'fedmade' not in value or not isinstance(value['fedmade'],dict) or set(value['fedmade'])!=METHOD_FIELDS:
        raise ValueError('FedMADE settings must exactly match the adapted-method schema')
    method=dict(value['fedmade'])
    common={k:v for k,v in value.items() if k!='fedmade'}
    if not isinstance(common.get('federated'),dict) or common['federated'].get('aggregation')!='fedmade':
        raise ValueError('Explicit fedmade aggregation is required')
    # Reuse shared validation rules; do not change what the FedAvg loader accepts.
    common['federated']={**common['federated'],'aggregation':'fedavg'}
    common=validate_training_config(common);common['federated']['aggregation']='fedmade'
    if common['experiment']['device']!='cpu':
        raise ValueError('CP3 supports the approved CPU benchmark; CUDA verification is not implemented')
    for field in ['auxiliary_manifest_path','auxiliary_evidence_path']:_path_setting(method[field],field)
    if method['reference_run_path'] is not None:_path_setting(method['reference_run_path'],'reference_run_path')
    if type(method['eps']) not in (int,float) or not math.isfinite(method['eps']) or method['eps']<=0:
        raise ValueError('eps must be positive and finite')
    if (type(method['min_samples']) is not int or method['min_samples']!=1 or method['solver']!='nnls'
            or method['norm']!='frobenius' or type(method['maxiter']) is not int or method['maxiter']<1):
        raise ValueError('Unsupported clustering/solver adaptation')
    if common['initialization']['mode']!='random':raise ValueError('Common benchmark uses fresh shared random initialization')
    return {**common,'fedmade':method}


def load_fedmade_config(path):
    source=Path(path).resolve()
    try:value=yaml.load(source.read_text(encoding='utf-8'),Loader=_PreparedConfigLoader)
    except yaml.YAMLError as error:raise ValueError(f'Invalid FedMADE YAML: {error}') from error
    value=validate_fedmade_config(value)
    fields=[('experiment','output_dir'),('dataset','bundle_path'),
            *[('fedmade',f) for f in ['auxiliary_manifest_path','auxiliary_evidence_path','reference_run_path']]]
    for section,field in fields:
        raw=value[section][field]
        if raw is not None:
            p=Path(raw);value[section][field]=str((p if p.is_absolute() else source.parent/p).resolve())
    return value
