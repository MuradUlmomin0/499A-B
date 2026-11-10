"""Export the active dataset's bounded MLP pilot into existing v1 FL formats."""
import hashlib
import json
from pathlib import Path

import numpy as np

from support.data.contract import ModelDataContract
from support.data.latest_audit import input_digest, source_hash


def _json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def export_latest_pilot(samples, config, output_dir):
    """Fit selected training only; export group-isolated clients and validation.

    Duplicate/mixed-label training inputs remain unchanged and enter the same
    client. Validation is never fitted or split into clients. Post-transform
    validation inputs matching selected final training inputs are excluded and
    counted, without refill. Empty validation fails before bundle publication.
    """
    output = Path(output_dir)
    classes, names = config['class_names'], config['feature_names']
    training = samples['train']['features']
    if (training.ndim != 2 or training.shape[1] != len(names) or not len(training)
            or not np.isfinite(training).all()):
        raise ValueError('Invalid training pilot features')
    for role in ('train', 'validation'):
        s = samples[role]
        if (s['features'].ndim != 2 or s['features'].shape[1] != len(names)
                or not len(s['features']) or len(s['labels']) != len(s['features'])
                or len(s['ordinals']) != len(s['features'])
                or not np.isfinite(s['features']).all()
                or not np.issubdtype(s['labels'].dtype, np.integer)
                or not np.isin(s['labels'], np.arange(len(classes))).all()
                or len(set(s['ordinals'].tolist())) != len(s['ordinals'])):
            raise ValueError(f'Invalid pilot {role} arrays')
    if not np.bincount(samples['train']['labels'], minlength=len(classes)).all():
        raise ValueError('Every configured class needs selected training examples')
    values = training.astype(np.float64)
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale == 0] = 1.
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError('Nonfinite training preprocessing')
    transformed = {}
    for role in ('train', 'validation'):
        with np.errstate(over='ignore', invalid='ignore'):
            transformed[role] = ((samples[role]['features'].astype(np.float64)-mean)/scale).astype(np.float32)
        if not np.isfinite(transformed[role]).all():
            raise ValueError('Nonfinite standardized inputs')
    groups = {role: [input_digest(row) for row in transformed[role]] for role in transformed}
    training_groups = set(groups['train'])
    keep = np.array([digest not in training_groups for digest in groups['validation']])
    final_exclusions = int((~keep).sum())
    if not keep.any():
        raise ValueError('No eligible validation remains after final float32 input exclusion')
    original = samples['validation']
    samples = {'train': samples['train'], 'validation': {
        'features': original['features'][keep], 'labels': original['labels'][keep],
        'ordinals': original['ordinals'][keep]}}
    transformed['validation'] = transformed['validation'][keep]
    groups['validation'] = [digest for digest,selected in zip(groups['validation'],keep) if selected]
    assignment = np.array([
        int.from_bytes(hashlib.sha256(f"{config['seed']}:{digest}".encode()).digest()[:8], 'big')
        % config['num_clients'] for digest in groups['train']], dtype=np.int64)
    partitions = {f'client_{i:03d}': ('train', np.flatnonzero(assignment == i))
                  for i in range(config['num_clients'])}
    if any(not len(indices) for role,indices in partitions.values()):
        raise ValueError('Seeded group assignment created an empty client')
    partitions['validation'] = ('validation', np.arange(len(samples['validation']['labels'])))
    preprocess = {'schema_version': 1, 'method': 'population_standardization',
                  'feature_names': names, 'class_names': classes,
                  'fit_sample_count': len(training), 'mean': mean.tolist(), 'scale': scale.tolist(),
                  'fit_role': 'selected_training_only', 'testing_only': True}
    _json(output/'preprocessing.json', preprocess)
    contract = ModelDataContract('mlp', len(names), len(classes), tuple(config['hidden_dims']),
                                 tuple(names), tuple(classes), source_hash(output/'preprocessing.json'))
    report = {'testing_only': True, 'train_rows': len(training),
              'validation_rows': len(samples['validation']['labels']), 'test_scored': False,
              'test_used_for_selection': False, 'simulated_clients': True,
              'client_assignment': 'seeded final-input group hash modulo clients; near-IID diagnostic',
              'selection': 'class-capped subsets; not natural source distribution',
              'validation_final_input_overlap_rows_removed': final_exclusions,
              'validation_exclusion_policy': 'raw input overlap with full training, then final float32 overlap with selected training; no refill',
              'prepared_train_validation_groups_disjoint': True,
              'checked_roles': ['train', 'validation'], 'partitions': {}}
    clients, validation = [], None
    with (output/'lineage.jsonl').open('x') as lineage:
        for name,(role,indices) in partitions.items():
            client = name if role == 'train' else None
            path = output/f'{name}.npz'
            with path.open('xb') as stream:
                np.savez(stream, features=transformed[role][indices], labels=samples[role]['labels'][indices],
                         feature_names=np.array(names), class_names=np.array(classes))
            ids = [f'Dataset/{role}.csv:row:{int(samples[role]["ordinals"][i])}' for i in indices]
            manifest = {'schema_version': 1, 'dataset_name': 'latest-source-diagnostic-pilot',
                        'partition_name': name, 'role': role, 'client_id': client,
                        'data_format': 'npz', 'data_path': path.name,
                        'preprocessing_path': 'preprocessing.json', 'contract': contract.to_dict(),
                        'testing_only': True}
            evidence = {'schema_version': 1, 'source_namespace': 'latest:sha256:'+config['archive_sha256'],
                        'data_sha256': source_hash(path), 'record_ids': ids,
                        'group_ids': [groups[role][i] for i in indices]}
            _json(output/f'{name}.manifest.json', manifest)
            _json(output/f'{name}.ids.json', evidence)
            for local,i in enumerate(indices):
                lineage.write(json.dumps({'record_id': ids[local], 'partition': name,
                                          'class_id': int(samples[role]['labels'][i]),
                                          'raw_group': input_digest(samples[role]['features'][i]),
                                          'final_group': groups[role][i]})+'\n')
            report['partitions'][name] = {'rows': len(indices), 'class_counts':
                np.bincount(samples[role]['labels'][indices], minlength=len(classes)).tolist()}
            ref = {'manifest_path': f'{name}.manifest.json', 'evidence_path': f'{name}.ids.json'}
            if client is not None:
                clients.append({'client_id': client, **ref})
            else:
                validation = ref
    _json(output/'federated_bundle.json', {'schema_version': 1, 'bundle_name': 'latest-MLP-FedAvg-diagnostic',
          'contract': contract.to_dict(), 'split_policy': 'group', 'clients': clients, 'validation': validation})
    return report
