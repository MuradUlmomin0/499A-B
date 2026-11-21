"""Diagnostic role/client splits, train-only scaling and existing v1 exports."""
import json
from pathlib import Path
import numpy as np
from Support.data.ciciot2023 import CLASS_NAMES
from Support.data.contract import ModelDataContract
from Support.data.pilot_sampling import feature_digest
from Support.data.split_checks import file_sha256


def _role_sizes(n):
    if n < 4:
        raise ValueError('Every class needs four eligible groups for the four roles')
    shares = np.array([.7, .1, .1, .1]) * n
    counts = np.maximum(1, np.floor(shares).astype(int))
    while counts.sum() > n:
        counts[np.argmax(counts-1)] -= 1
    while counts.sum() < n:
        counts[np.argmax(shares-counts)] += 1
    return counts.tolist()


def partition_indices(labels, config):
    """Stratified 70/10/10/10 roles; only train indices enter non-IID clients."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in 'iu' or not np.isin(labels, np.arange(8)).all():
        raise ValueError('Invalid pilot labels')
    rng = np.random.default_rng(config['pilot']['seed'])
    pools = {name: [] for name in ['train', 'validation', 'auxiliary', 'test']}
    for label in range(8):
        indices = np.flatnonzero(labels == label)
        counts = _role_sizes(len(indices))
        rng.shuffle(indices)
        start = 0
        for role, count in zip(pools, counts):
            pools[role].append(indices[start:start+count])
            start += count
    num_clients = config['pilot']['num_clients']
    for _ in range(100):
        clients = [[] for _ in range(num_clients)]
        for class_indices in pools['train']:
            indices = class_indices.copy()
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.full(num_clients, config['pilot']['dirichlet_alpha']))
            counts = rng.multinomial(len(indices), proportions)
            start = 0
            for client, count in zip(clients, counts):
                client.extend(indices[start:start+count].tolist())
                start += count
        if all(clients):
            result = {f'client_{i:03d}': np.array(rows, dtype=np.int64) for i, rows in enumerate(clients)}
            result.update({role: np.concatenate(pools[role]) for role in ['validation', 'auxiliary', 'test']})
            return result
    raise ValueError('Non-IID allocation produced empty clients after 100 attempts')


def fit_preprocessing(features, training_indices, feature_names, *, chunk_rows):
    """Population standardization, fitted only on the explicit training indices."""
    if not len(training_indices):
        raise ValueError('Cannot fit preprocessing without training rows')
    mean = np.zeros(features.shape[1], dtype=np.float64)
    for start in range(0, len(training_indices), chunk_rows):
        block = features[training_indices[start:start+chunk_rows]].astype(np.float64)
        if not np.isfinite(block).all():
            raise ValueError('Training features must be finite')
        mean += block.sum(axis=0)
    mean /= len(training_indices)
    variance = np.zeros_like(mean)
    for start in range(0, len(training_indices), chunk_rows):
        block = features[training_indices[start:start+chunk_rows]].astype(np.float64)
        variance += ((block-mean)**2).sum(axis=0)
    scale = np.sqrt(variance / len(training_indices))
    scale[scale == 0] = 1
    return {'schema_version': 1, 'method': 'population_standardization',
            'canonical_input_dtype': 'little_endian_float32', 'feature_names': list(feature_names),
            'fit_sample_count': len(training_indices), 'mean': mean.tolist(), 'scale': scale.tolist()}


def _json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def export_pilot(sample, config, output_dir):
    """Write all roles; reject cross-role transformed-input collisions.

    Group IDs are final model-input fingerprints, not device IDs. Source IDs
    retain member/row ordinals. Final test/auxiliary are not in the FL bundle.
    This writes no trained model or completed-run marker.
    """
    output_dir = Path(output_dir)
    features, labels = sample['features'], sample['labels']
    if features.ndim != 2 or len(features) != len(labels) or not np.isfinite(features).all():
        raise ValueError('Invalid selected pilot arrays')
    partitions = partition_indices(labels, config)
    training = np.concatenate([indices for name, indices in partitions.items() if name.startswith('client_')])
    transform = fit_preprocessing(features, training, config['source']['feature_names'], chunk_rows=config['pilot']['chunk_rows'])
    transform.update(class_names=list(CLASS_NAMES), label_source='folder_name_unverified', testing_only=True)
    _json(output_dir/'preprocessing.json', transform)
    preprocessing_hash = file_sha256(output_dir/'preprocessing.json')
    contract = ModelDataContract('mlp', features.shape[1], 8, tuple(config['model']['hidden_dims']),
                                 tuple(config['source']['feature_names']), CLASS_NAMES, preprocessing_hash)
    mean, scale = np.array(transform['mean']), np.array(transform['scale'])
    namespace = 'ciciot2023:sha256:' + config['source']['sha256']
    report = {'schema_version': 1, 'testing_only': True, 'purpose': 'diagnostic_real_data_pilot',
              'simulated_clients': True, 'label_source': 'folder_name_unverified',
              'provenance_verified': False, 'split_basis': 'unique feature groups, not devices/sessions/time',
              'partitions': {}, 'within_partition_transformed_duplicate_rows': 0,
              'preprocessing_sha256': preprocessing_hash, 'contract': contract.to_dict()}
    seen, record_ids_seen = {}, set()
    client_refs = []
    with (output_dir/'lineage.jsonl').open('x') as lineage:
        for name, indices in partitions.items():
            role = 'train' if name.startswith('client_') else name
            client_id = name if role == 'train' else None
            values = np.empty((len(indices), features.shape[1]), dtype=np.float32)
            for start in range(0, len(indices), config['pilot']['chunk_rows']):
                take = indices[start:start+config['pilot']['chunk_rows']]
                with np.errstate(over='ignore', invalid='ignore'):
                    values[start:start+len(take)] = (features[take].astype(np.float64)-mean)/scale
            if not np.isfinite(values).all():
                raise ValueError('Standardized inputs overflow float32')
            ids, groups = [], []
            for local, source_index in enumerate(indices):
                group = feature_digest(values[local])
                previous = seen.get(group)
                if previous is not None:
                    if previous != (name, int(labels[source_index])):
                        raise ValueError('Final float32 collision crosses partitions or labels; review the preparation protocol')
                    report['within_partition_transformed_duplicate_rows'] += 1
                seen[group] = (name, int(labels[source_index]))
                file_index, ordinal = int(sample['file_indices'][source_index]), int(sample['row_ordinals'][source_index])
                record_id = f'member:{file_index}:row:{ordinal}'
                if record_id in record_ids_seen:
                    raise ValueError('Duplicate source ID in selected pilot')
                record_ids_seen.add(record_id)
                ids.append(record_id)
                groups.append(group.hex())
                lineage.write(json.dumps({'record_id': record_id, 'member': sample['members'][file_index]['path'],
                    'row_ordinal': ordinal, 'subtype': sample['subtype_names'][int(sample['subtypes'][source_index])],
                    'broad_class': CLASS_NAMES[int(labels[source_index])], 'partition': name,
                    'raw_feature_group': sample['hashes'][source_index].tobytes().hex()})+'\n')
            data_path = output_dir / f'{name}.npz'
            with data_path.open('xb') as stream:
                np.savez(stream, features=values, labels=labels[indices],
                         feature_names=np.array(contract.feature_names), class_names=np.array(CLASS_NAMES))
            manifest = {'schema_version': 1, 'dataset_name': 'CICIoT2023-diagnostic-pilot', 'partition_name': name,
                        'role': role, 'client_id': client_id, 'data_format': 'npz', 'data_path': data_path.name,
                        'preprocessing_path': 'preprocessing.json', 'contract': contract.to_dict(), 'testing_only': True}
            evidence = {'schema_version': 1, 'source_namespace': namespace, 'data_sha256': file_sha256(data_path),
                        'record_ids': ids, 'group_ids': groups}
            _json(output_dir/f'{name}.manifest.json', manifest)
            _json(output_dir/f'{name}.ids.json', evidence)
            report['partitions'][name] = {'role': role, 'rows': len(indices),
                'class_counts': np.bincount(labels[indices], minlength=8).tolist(),
                'manifest': f'{name}.manifest.json', 'evidence': f'{name}.ids.json'}
            ref = {'manifest_path': f'{name}.manifest.json', 'evidence_path': f'{name}.ids.json'}
            if client_id is not None:
                client_refs.append({'client_id': client_id, **ref})
            elif role == 'validation':
                validation_ref = ref
    _json(output_dir/'federated_bundle.json', {'schema_version': 1, 'bundle_name': 'CICIoT2023-diagnostic-pilot',
        'contract': contract.to_dict(), 'split_policy': 'group', 'clients': client_refs, 'validation': validation_ref})
    report['all_role_record_ids_disjoint'] = len(record_ids_seen) == len(features)
    report['all_role_feature_groups_disjoint'] = True
    report['fit_sample_count'] = len(training)
    return report
