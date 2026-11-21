"""Strict CP6 wrapper; referenced CP3 and CP4 configurations stay immutable."""
import math
from pathlib import Path
import re
import yaml
from Support.Utils.prepared_config import _PreparedConfigLoader,_path_setting
from Support.Utils.poisoning_config import ATTACK_FIELDS
from Support.Utils.fedmade_config import load_fedmade_config
from Support.Utils.temporal_config import load_temporal_config


def load_fusion_config(path):
    path=Path(path).resolve()
    try:v=yaml.load(path.read_text(),Loader=_PreparedConfigLoader)
    except yaml.YAMLError as e:raise ValueError(f'Invalid fusion YAML: {e}') from e
    fields={'schema_version','method','experiment','baseline_config_path','temporal_config_path','reference_clean_run_path','endpoint_reference_run_path','lambda_value','attack'}
    if not isinstance(v,dict) or set(v)!=fields:raise ValueError('Invalid fusion configuration fields')
    if type(v['schema_version']) is not int or v['schema_version']!=1 or v['method']!='fusion':raise ValueError('Unsupported fusion schema/method')
    lam=v['lambda_value']
    if type(lam) not in (int,float) or not math.isfinite(lam) or lam not in (0,.5,1):raise ValueError('CP6 permits only fixed lambda 0, 0.5, 1')
    if not isinstance(v['experiment'],dict) or set(v['experiment'])!={'name','output_dir'}:raise ValueError('Invalid fusion experiment')
    if not isinstance(v['experiment']['name'],str) or not re.fullmatch(r'[A-Za-z0-9_-]+',v['experiment']['name']):raise ValueError('Invalid experiment name')
    for section,key in [(None,'baseline_config_path'),(None,'temporal_config_path'),(None,'reference_clean_run_path'),(None,'endpoint_reference_run_path'),('experiment','output_dir')]:
        holder=v[section] if section else v;raw=holder[key]
        if raw is None and key.endswith('run_path'):continue
        _path_setting(raw,key);p=Path(raw);holder[key]=str((p if p.is_absolute() else path.parent/p).resolve())
    if lam==.5 and v['endpoint_reference_run_path'] is not None:raise ValueError('Endpoint reference requires endpoint lambda')
    a=v['attack']
    if not isinstance(a,dict) or set(a)!=ATTACK_FIELDS or a['type']!='targeted_label' or type(a['enabled']) is not bool:raise ValueError('Unsupported attack')
    if any(type(a[k]) is not int for k in ATTACK_FIELDS-{'type','enabled'}):raise ValueError('Attack integers required')
    if (a['target_class']<0 or a['destination_class']<0 or a['target_class']==a['destination_class'] or a['malicious_clients']<0
        or (a['enabled'] and a['malicious_clients']==0) or (not a['enabled'] and a['malicious_clients']!=0)
        or a['start_round']<1 or a['end_round']<a['start_round'] or a['selection_seed_offset']!=5005):raise ValueError('Invalid attack settings')
    base=load_fedmade_config(v['baseline_config_path']);temporal=load_temporal_config(v['temporal_config_path'])
    for key in ['dataset','initialization','training','selection']:
        if base[key]!=temporal[key]:raise ValueError(f'Component {key} mismatch')
    for key in ['seed','device','deterministic']:
        if base['experiment'][key]!=temporal['experiment'][key]:raise ValueError('Component experiment mismatch')
    for key in ['rounds','participation_rate']:
        if base['federated'][key]!=temporal['federated'][key]:raise ValueError('Component federated mismatch')
    if a['end_round']>base['federated']['rounds']:raise ValueError('Attack exceeds training budget')
    v['_baseline']=base;v['_temporal']=temporal
    return v


def public_config(config):return {k:v for k,v in config.items() if not k.startswith('_')}
