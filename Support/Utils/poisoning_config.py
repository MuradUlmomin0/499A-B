"""Separate strict attack configuration; clean method schemas remain unchanged."""
from Support.Utils.runtime_paths import load_json as runtime_json, load_json_file as runtime_json_file, load_yaml as runtime_yaml, safe_load_yaml as runtime_safe_yaml
from pathlib import Path
import re
import yaml
from Support.Utils.prepared_config import _PreparedConfigLoader,_path_setting
from Support.Utils.training_config import load_training_config
from Support.Utils.fedmade_config import load_fedmade_config
from Support.Utils.temporal_config import load_temporal_config

LOADERS=dict(fedavg=load_training_config,fedmade=load_fedmade_config,temporal=load_temporal_config)
ATTACK_FIELDS={'type','enabled','target_class','destination_class','malicious_clients','start_round','end_round','selection_seed_offset'}


def load_poisoning_config(path):
    path=Path(path).resolve()
    try:v=runtime_yaml(path.read_text(),Loader=_PreparedConfigLoader)
    except yaml.YAMLError as error:raise ValueError(f'Invalid attack YAML: {error}') from error
    if not isinstance(v,dict) or set(v)!={'schema_version','method','experiment','baseline_config_path','reference_clean_run_path','attack'}:raise ValueError('Invalid poisoning configuration fields')
    if type(v['schema_version']) is not int or v['schema_version']!=1 or v['method'] not in LOADERS:raise ValueError('Unsupported poisoning schema/method')
    if not isinstance(v['experiment'],dict) or set(v['experiment'])!={'name','output_dir'}:raise ValueError('Invalid poisoning experiment')
    if not isinstance(v['experiment']['name'],str) or not re.fullmatch(r'[A-Za-z0-9_-]+',v['experiment']['name']):raise ValueError('Invalid experiment name')
    for section,key in [(None,'baseline_config_path'),(None,'reference_clean_run_path'),('experiment','output_dir')]:
        holder=v[section] if section else v;raw=holder[key]
        if raw is None and key=='reference_clean_run_path':continue
        _path_setting(raw,key);p=Path(raw);holder[key]=str((p if p.is_absolute() else path.parent/p).resolve())
    a=v['attack']
    if not isinstance(a,dict) or set(a)!=ATTACK_FIELDS or a['type']!='targeted_label' or type(a['enabled']) is not bool:raise ValueError('Unsupported attack definition')
    for key in ATTACK_FIELDS-{'type','enabled'}:
        if type(a[key]) is not int:raise ValueError(f'Attack {key} must be an integer')
    if (a['target_class']<0 or a['destination_class']<0 or a['target_class']==a['destination_class'] or a['malicious_clients']<0
            or (a['enabled'] and a['malicious_clients']==0) or (not a['enabled'] and a['malicious_clients']!=0)
            or a['start_round']<1 or a['end_round']<a['start_round'] or a['selection_seed_offset']!=5005):raise ValueError('Invalid attack settings')
    base=LOADERS[v['method']](v['baseline_config_path'])
    if base['experiment']['device']!='cpu' or base['initialization']['mode']!='random' or a['end_round']>base['federated']['rounds']:raise ValueError('Unsupported execution budget')
    if base['federated']['aggregation']!=v['method']:raise ValueError('Method/baseline configuration mismatch')
    v['_baseline']=base
    return v


def public_config(config):return {k:v for k,v in config.items() if k!='_baseline'}
