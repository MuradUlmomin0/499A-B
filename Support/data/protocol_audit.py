"""Read-only support audit under train > validation > test input-group priority.

Counts original and eligible rows/groups; does not sample, reserve auxiliary
data, fit preprocessing, allocate clients, train or score final-test records.
Run with --config pointing to the existing latest-source preparation YAML.
"""
from Support.Utils.runtime_paths import load_json as runtime_json, load_json_file as runtime_json_file, load_yaml as runtime_yaml, safe_load_yaml as runtime_safe_yaml
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yaml

from Support.data.latest_audit import ROLES, _sort, source_hash


def summarize_sorted(stream, classes):
    """Reduce sorted canonical digest/role/label records with bounded memory.

    The earliest original role owns the entire input group. Test membership
    cannot remove or select earlier-role rows. Mixed labels stop the audit.
    """
    result = {name: {role: [0]*len(classes) for role in ROLES}
              for name in ('original_rows','original_groups','eligible_rows',
                           'eligible_groups','excluded_rows')}
    previous, counts, label = None, [0,0,0], None
    unique = 0

    def finish():
        nonlocal unique
        if previous is None:
            return
        unique += 1
        owner = next(i for i,n in enumerate(counts) if n)
        for i,role in enumerate(ROLES):
            if not counts[i]:
                continue
            result['original_rows'][role][label] += counts[i]
            result['original_groups'][role][label] += 1
            if i == owner:
                result['eligible_rows'][role][label] += counts[i]
                result['eligible_groups'][role][label] += 1
            else:
                result['excluded_rows'][role][label] += counts[i]

    for line in stream:
        digest, role_text, label_text = line.rstrip('\n').split('\t')
        role, current_label = int(role_text), int(label_text)
        if (len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest)
                or role not in range(3) or current_label not in range(len(classes))
                or previous is not None and digest < previous):
            raise ValueError('Malformed or unsorted fingerprint record')
        if digest != previous:
            finish()
            previous, counts, label = digest, [0,0,0], current_label
        if current_label != label:
            raise ValueError('Conflicting labels within a canonical input group')
        counts[role] += 1
    finish()
    result['unique_groups'] = unique
    return result


def audit_protocol_support(config, output_dir, work_dir):
    """Write a fresh, hash-bound support report using a bounded external sort."""
    output_dir, work_dir = Path(output_dir), Path(work_dir)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite support evidence: {output_dir}')
    if source_hash(config['archive_path']) != config['archive_sha256']:
        raise ValueError('Active archive SHA-256 mismatch')
    classes, features = config['class_names'], config['feature_names']
    if not classes or len(classes)!=len(set(classes)) or not features or len(features)!=len(set(features)):
        raise ValueError('Class and feature names must be nonempty and unique')
    if type(config['chunk_rows']) is not int or config['chunk_rows']<=0:
        raise ValueError('chunk_rows must be a positive integer')
    mapping = {name:i for i,name in enumerate(classes)}
    with tempfile.TemporaryDirectory(prefix='protocol_support_',dir=work_dir) as tmp:
        scratch=Path(tmp); signatures=scratch/'inputs.tsv'; ordered=scratch/'ordered.tsv'
        with signatures.open('x',buffering=1024**2) as target, ZipFile(config['archive_path']) as archive:
            for role_id,role in enumerate(ROLES):
                member=f'Dataset/{role}.csv'
                if archive.namelist().count(member)!=1:
                    raise ValueError(f'Expected exactly one source member: {member}')
                count=0
                with archive.open(member) as stream:
                    for frame in pd.read_csv(stream, dtype={f:np.float32 for f in features},
                            chunksize=config['chunk_rows'], skip_blank_lines=False,
                            encoding='utf-8-sig',on_bad_lines='error'):
                        if list(frame.columns)!=[*features,'label'] or not isinstance(frame.index,pd.RangeIndex):
                            raise ValueError(f'Unexpected source schema: {member}')
                        values=frame[features].to_numpy(dtype='<f4',copy=True)
                        if not np.isfinite(values).all():
                            raise ValueError(f'Nonfinite source inputs: {member}')
                        labels=frame['label'].map(mapping)
                        if labels.isna().any():
                            raise ValueError(f'Unknown or missing label: {member}')
                        values[values==0]=0
                        for row,label in zip(values,labels.to_numpy(dtype=np.int64)):
                            target.write(f'{hashlib.sha256(row.tobytes()).hexdigest()}\t{role_id}\t{int(label):03d}\n')
                        count+=len(frame)
                if count!=config['expected_rows'][role]:
                    raise ValueError(f'Unexpected {role} row count: {count}')
                print(json.dumps({'stage':'protocol_source_scan','role':role,'rows':count}),flush=True)
        print(json.dumps({'stage':'protocol_sort','bytes':signatures.stat().st_size}),flush=True)
        _sort(signatures,ordered,scratch)
        with ordered.open() as stream:
            report=summarize_sorted(stream,classes)
        for role in ROLES:
            if sum(report['original_rows'][role])!=config['expected_rows'][role]:
                raise RuntimeError('Sorted record totals differ from source')
        if source_hash(config['archive_path'])!=config['archive_sha256']:
            raise ValueError('Source archive changed during audit')
    report.update(schema_version=1,archive_sha256=config['archive_sha256'],
        feature_names=features,class_names=classes,priority=list(ROLES),
        representation='canonical little-endian float32 features; signed zero normalized; labels excluded',
        auxiliary_reserved=False,clients_allocated=False,preprocessing_fitted=False,
        test_scored=False,source_modified=False,provenance_verified=False,
        scope='Original-source eligibility only; before auxiliary reservation, sampling and post-scaling collision checks')
    output_dir.mkdir(parents=True,exist_ok=False)
    with (output_dir/'Support.json').open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False); stream.write('\n')
    with (output_dir/'support_by_class.csv').open('x',newline='') as stream:
        writer=csv.writer(stream)
        columns=[(role,kind) for role in ROLES for kind in
                 ('original_rows','original_groups','eligible_rows','eligible_groups','excluded_rows')]
        writer.writerow(['class_id','class_name',*[f'{role}_{kind}' for role,kind in columns]])
        for i,name in enumerate(classes):
            writer.writerow([i,name,*[report[kind][role][i] for role,kind in columns]])
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--work-dir',type=Path,required=True)
    args=parser.parse_args()
    with args.config.open() as stream:
        config=runtime_safe_yaml(stream)
    audit_protocol_support(config,args.output_dir,args.work_dir)
    print(f'Support evidence: {args.output_dir}')


if __name__=='__main__':
    main()
