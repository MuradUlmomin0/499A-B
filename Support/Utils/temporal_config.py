"""Strict configuration for the approved CP4 temporal-trust adaptation."""
from pathlib import Path
import yaml
from Support.federated.temporal import DEFAULTS
from Support.Utils.training_config import validate_training_config
from Support.Utils.prepared_config import _PreparedConfigLoader,_path_setting


def validate_temporal_config(value):
    if not isinstance(value,dict) or not isinstance(value.get('temporal'),dict):raise ValueError('Missing temporal settings')
    method=dict(value['temporal'])
    if set(method)!=set(DEFAULTS)|{'reference_run_path'}:raise ValueError('Temporal settings must match the frozen schema')
    for k,v in DEFAULTS.items():
        if type(method[k]) is not type(v) or method[k]!=v:raise ValueError(f'Unapproved temporal setting: {k}')
    if method['reference_run_path'] is not None:_path_setting(method['reference_run_path'],'reference_run_path')
    common={k:v for k,v in value.items() if k!='temporal'}
    if not isinstance(common.get('federated'),dict) or common['federated'].get('aggregation')!='temporal':raise ValueError('Explicit temporal aggregation required')
    common['federated']={**common['federated'],'aggregation':'fedavg'}
    common=validate_training_config(common);common['federated']['aggregation']='temporal'
    if common['experiment']['device']!='cpu' or common['initialization']['mode']!='random':raise ValueError('CP4 requires CPU and fresh random initialization')
    return {**common,'temporal':method}


def load_temporal_config(path):
    path=Path(path).resolve()
    try:v=yaml.load(path.read_text(),Loader=_PreparedConfigLoader)
    except yaml.YAMLError as error:raise ValueError(f'Invalid temporal YAML: {error}') from error
    v=validate_temporal_config(v)
    for section,key in [('experiment','output_dir'),('dataset','bundle_path'),('temporal','reference_run_path')]:
        if v[section][key] is not None:
            p=Path(v[section][key]);v[section][key]=str((p if p.is_absolute() else path.parent/p).resolve())
    return v
