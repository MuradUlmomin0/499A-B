"""Prepare/audit the active dataset for existing MLP FedAvg; never trains here."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
from time import perf_counter

import numpy as np
import yaml

from support.data.latest_audit import ROLES, audit_latest, input_digest, source_hash
from support.data.latest_pilot import export_latest_pilot
from support.data.federated_bundle import load_federated_bundle
from support.utils.prepared_config import _PreparedConfigLoader, _path_setting


def _json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def load_latest_config(path):
    source = Path(path).resolve()
    with source.open() as stream:
        c = yaml.load(stream, Loader=_PreparedConfigLoader)
    fields = {'schema_version','archive_path','archive_sha256','feature_names','class_names',
              'expected_rows','seed','chunk_rows','train_per_class','validation_per_class',
              'num_clients','hidden_dims','output_dir','work_dir'}
    if not isinstance(c, dict) or set(c) != fields or type(c['schema_version']) is not int or c['schema_version'] != 1:
        raise ValueError('Expected exact latest-pilot configuration schema version 1')
    for field,low,high in [('seed',0,2**32-1),('chunk_rows',1,20000),('train_per_class',1,2000),
                           ('validation_per_class',1,1000),('num_clients',2,10)]:
        if type(c[field]) is not int or not low <= c[field] <= high:
            raise ValueError(f'Invalid diagnostic {field}')
    for field,low,high in [('feature_names',1,512),('class_names',2,256)]:
        names = c[field]
        if (not isinstance(names, list) or not low <= len(names) <= high
                or any(not isinstance(v, str) or not v or v != v.strip() for v in names)
                or len(set(names)) != len(names)):
            raise ValueError(f'Invalid ordered {field}')
    if (not isinstance(c['expected_rows'], dict) or set(c['expected_rows']) != set(ROLES)
            or any(type(n) is not int or not 1 <= n <= 10000000 for n in c['expected_rows'].values())):
        raise ValueError('Require explicit expected rows for all original source splits')
    if (not isinstance(c['hidden_dims'], list) or not c['hidden_dims']
            or any(type(n) is not int or n < 1 for n in c['hidden_dims'])):
        raise ValueError('Invalid MLP hidden_dims')
    fingerprint = c['archive_sha256']
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(v not in '0123456789abcdef' for v in fingerprint):
        raise ValueError('Require pinned archive SHA-256')
    candidate_bytes = len(c['class_names'])*(c['train_per_class']+c['validation_per_class'])*len(c['feature_names'])*4
    widths = [len(c['feature_names']),*c['hidden_dims'],len(c['class_names'])]
    model_bytes = sum((a+1)*b for a,b in zip(widths,widths[1:]))*4*(c['num_clients']+3)
    if candidate_bytes > 128*1024**2 or model_bytes > 128*1024**2:
        raise ValueError('Pilot candidate/model buffers exceed the diagnostic memory budget')
    for field in ('archive_path','output_dir','work_dir'):
        _path_setting(c[field],field)
        c[field] = str((source.parent/Path(c[field])).resolve())
    out,work = Path(c['output_dir']),Path(c['work_dir'])
    if out == work or out.is_relative_to(work) or work.is_relative_to(out):
        raise ValueError('Output and scratch roots must not contain one another')
    return c


def prepare_latest(config_path, *, audit_only=False):
    """Audit first, or finish an unchanged owned audit into prepared v1 inputs.

    --audit-only allows inspection before any split/scaling export. Reusing the
    exact audited configuration verifies source/candidate hashes. Completed or
    failed outputs and arbitrary existing directories are never overwritten.
    """
    started = perf_counter()
    c = load_latest_config(config_path)
    output,work = Path(c['output_dir']),Path(c['work_dir'])
    config_hash = hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
    if output.exists() and (audit_only or not (output/'audit_summary.json').is_file()
                            or (output/'run_summary.json').exists() or (output/'failure.json').exists()):
        raise FileExistsError(f'Existing output is preserved: {output}')
    if source_hash(c['archive_path']) != c['archive_sha256']:
        raise ValueError('Archive SHA-256 mismatch')
    work.mkdir(parents=True,exist_ok=True)
    required = sum(c['expected_rows'].values())*71*3 + 256*1024**2
    if shutil.disk_usage(work).free < required:
        raise ValueError(f'Insufficient disk space for split audit; need approximately {required} bytes')
    if not output.exists():
        output.mkdir(parents=True,exist_ok=False)
    try:
        if (output/'audit_summary.json').is_file():
            stored = json.loads((output/'audit_summary.json').read_text())
            if stored['config_sha256'] != config_hash or source_hash(output/'candidates.npz') != stored['candidates_sha256']:
                raise ValueError('Audit config or candidate fingerprint changed')
            with np.load(output/'candidates.npz',allow_pickle=False) as data:
                samples = {role:{field:data[f'{role}_{field}'].copy() for field in ['features','labels','ordinals']}
                           for role in ROLES[:2]}
            for s in samples.values():
                s['hashes'] = [input_digest(row) for row in s['features']]
        else:
            audit,samples = audit_latest(c,output,work)
            with (output/'candidates.npz').open('xb') as stream:
                np.savez(stream, **{f'{role}_{field}':s[field] for role,s in samples.items()
                                   for field in ['features','labels','ordinals']})
            _json(output/'audit_summary.json', {'status':'audit_completed','config_sha256':config_hash,
                  'candidates_sha256':source_hash(output/'candidates.npz'),
                  'archive_sha256':c['archive_sha256'],'testing_only':True})
            with (output/'config.yaml').open('x') as stream:
                yaml.safe_dump(c,stream,sort_keys=False)
        if audit_only:
            return output
        report = export_latest_pilot(samples,c,output)
        _json(output/'preparation_report.json',report)
        bundle = load_federated_bundle(output/'federated_bundle.json')
        if source_hash(c['archive_path']) != c['archive_sha256']:
            raise ValueError('Source changed during export')
        _json(output/'run_summary.json', {
            'status':'completed','workflow':'latest_MLP_pilot_preparation','testing_only':True,
            'training_performed':False,'source_provenance_verified':False,
            'train_rows':report['train_rows'],'validation_rows':report['validation_rows'],
            'client_count':len(bundle.clients),'bundle_sha256':bundle.sha256,
            'test_used_for_selection':False,'test_scored':False,
            'duration_seconds':perf_counter()-started,
            'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    except Exception as error:
        _json(output/'failure.json', {'status':'failed','error':str(error),'training_performed':False})
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description='Audit/prepare the latest bounded MLP FedAvg pilot, without training')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--audit-only',action='store_true',help='Report split audit and save bounded candidates; export later with the same config')
    args = parser.parse_args()
    path = prepare_latest(args.config,audit_only=args.audit_only)
    print(f'Latest pilot artifacts: {path}')


if __name__ == '__main__':
    main()
