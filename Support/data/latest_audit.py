"""Bounded exact-model-input audit and pilot candidates for the active CSV ZIP.

Fingerprints describe parsed float32 feature vectors, not device/capture identity
or original CSV bytes. Original files and labels are never modified. Final test
is audited, but cannot affect candidate selection or preprocessing.
"""
from collections import Counter
import hashlib
import heapq
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import tempfile
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROLES = ('train', 'validation', 'test')


def source_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def input_digest(row):
    values = np.array(row, dtype='<f4', copy=True)
    values[values == 0] = 0
    return hashlib.sha256(values.tobytes()).hexdigest()


class _Candidates:
    """Bounded, deterministic bottom-k source-row ranks per true class."""
    def __init__(self, cap, classes, seed, role):
        self.cap, self.seed, self.role = cap, seed, role
        self.heaps = [[] for _ in classes]

    def offer(self, label, ordinal, features, digest):
        rank = int.from_bytes(hashlib.blake2b(
            f'{self.seed}:{self.role}:{ordinal}'.encode(), digest_size=16).digest(), 'big')
        heap = self.heaps[label]
        if len(heap) < self.cap:
            heapq.heappush(heap, (-rank, ordinal, label, features.copy(), digest))
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, (-rank, ordinal, label, features.copy(), digest))

    def arrays(self, excluded=()):
        rows = sorted((item for heap in self.heaps for item in heap if item[4] not in excluded),
                      key=lambda item: item[1])
        if not rows:
            raise ValueError(f'No eligible {self.role} candidates remain')
        return {'features': np.stack([r[3] for r in rows]),
                'ordinals': np.array([r[1] for r in rows], dtype=np.int64),
                'labels': np.array([r[2] for r in rows], dtype=np.int64),
                'hashes': [r[4] for r in rows]}


def _sort_limit():
    ceiling = 512 * 1024**2
    resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))


def _sort(source, destination, directory):
    executable = shutil.which('sort')
    if not executable:
        raise RuntimeError('A compatible coreutils sort is required')
    process = subprocess.run(
        [executable, '--parallel=1', '-S', '32M', '--batch-size=16', '-T', str(directory),
         str(source), '-o', str(destination)],
        env={**os.environ, 'LC_ALL': 'C'}, preexec_fn=_sort_limit,
        capture_output=True, text=True)
    if process.returncode or process.stderr.strip() or not destination.is_file():
        raise RuntimeError(f'Split audit sort failed: {process.stderr.strip()}')


def audit_latest(config, output_dir, work_dir):
    """Return full-original-split audit and bounded train/validation candidates.

    Sample ranks depend only on seed/role/original ordinal, not test contents.
    Validation candidates matching any original training input are excluded;
    exclusions and missing classes are reported, with no hidden refill.
    """
    if source_hash(config['archive_path']) != config['archive_sha256']:
        raise ValueError('Active archive SHA-256 mismatch')
    output_dir, work_dir = Path(output_dir), Path(work_dir)
    classes = config['class_names']
    mapping = {name: i for i, name in enumerate(classes)}
    pools = {role: _Candidates(config[f'{role if role == "train" else "validation"}_per_class'],
                               classes, config['seed'], role) for role in ROLES[:2]}
    audit = {'schema_version': 1, 'archive_sha256': config['archive_sha256'],
             'representation': 'canonical little-endian float32 input features; signed zero normalized; labels excluded',
             'rows': {}, 'class_counts': {}, 'class_names': classes, 'unique_groups': 0,
             'unique_groups_by_role': dict.fromkeys(ROLES, 0),
             'duplicate_excess_by_role': dict.fromkeys(ROLES, 0),
             'overlap_groups': dict.fromkeys(['train_validation','train_test','validation_test'], 0),
             'mixed_label_groups': 0, 'mixed_label_rows': 0,
             'cross_split_rows_by_role': dict.fromkeys(ROLES, 0),
             'test_used_for_selection': False, 'test_scored': False,
             'source_modified': False, 'provenance_verified': False,
             'hash_collisions_verified': False, 'near_duplicates_checked': False}
    with tempfile.TemporaryDirectory(prefix='latest_split_audit_', dir=work_dir) as temp:
        scratch = Path(temp)
        signatures, ordered = scratch/'inputs.tsv', scratch/'inputs.sorted.tsv'
        with signatures.open('x', buffering=1024**2) as target, ZipFile(config['archive_path']) as archive:
            for role_id, role in enumerate(ROLES):
                member = f'Dataset/{role}.csv'
                if archive.namelist().count(member) != 1:
                    raise ValueError(f'Expected one source member: {member}')
                count, distribution = 0, Counter()
                with archive.open(member) as stream:
                    for frame in pd.read_csv(stream, dtype={name: np.float32 for name in config['feature_names']},
                                             chunksize=config['chunk_rows'], skip_blank_lines=False,
                                             encoding='utf-8-sig', on_bad_lines='error'):
                        if list(frame.columns) != [*config['feature_names'], 'label'] or not isinstance(frame.index, pd.RangeIndex):
                            raise ValueError(f'Unexpected source schema: {member}')
                        values = frame[config['feature_names']].to_numpy(dtype='<f4', copy=True)
                        if not np.isfinite(values).all():
                            raise ValueError(f'Nonfinite source input: {member}')
                        values[values == 0] = 0
                        labels = frame['label'].map(mapping)
                        if labels.isna().any():
                            raise ValueError(f'Unknown/missing source label: {member}')
                        for offset, (row, label) in enumerate(zip(values, labels.to_numpy(dtype=np.int64))):
                            digest = hashlib.sha256(row.tobytes()).hexdigest()
                            label = int(label)
                            target.write(f'{digest}\t{role_id}\t{label:03d}\n')
                            distribution[label] += 1
                            if role in pools:
                                pools[role].offer(label, count+offset, row, digest)
                        count += len(frame)
                if count != config['expected_rows'][role]:
                    raise ValueError(f'Unexpected {role} row count: {count}')
                audit['rows'][role] = count
                audit['class_counts'][role] = [distribution[i] for i in range(len(classes))]
                print(json.dumps({'stage': 'source_scan', 'role': role, 'rows': count}), flush=True)
        if any(n == 0 for n in audit['class_counts']['train']):
            raise ValueError('Every configured class must occur in source training')
        print(json.dumps({'stage': 'sort_split_fingerprints', 'bytes': signatures.stat().st_size}), flush=True)
        _sort(signatures, ordered, scratch)
        validation_hashes = {r[4] for heap in pools['validation'].heaps for r in heap}
        excluded = set()
        previous, counts, label_mask = None, [0,0,0], 0

        def finish():
            if previous is None:
                return
            audit['unique_groups'] += 1
            roles_present = sum(n > 0 for n in counts)
            for i, role in enumerate(ROLES):
                if counts[i]:
                    audit['unique_groups_by_role'][role] += 1
                    audit['duplicate_excess_by_role'][role] += counts[i]-1
                    if roles_present > 1:
                        audit['cross_split_rows_by_role'][role] += counts[i]
            for i,j,name in [(0,1,'train_validation'),(0,2,'train_test'),(1,2,'validation_test')]:
                if counts[i] and counts[j]:
                    audit['overlap_groups'][name] += 1
            if label_mask.bit_count() > 1:
                audit['mixed_label_groups'] += 1
                audit['mixed_label_rows'] += sum(counts)
            if counts[0] and counts[1] and previous in validation_hashes:
                excluded.add(previous)

        consumed = 0
        with ordered.open() as stream:
            for line in stream:
                digest, role, label = line.rstrip('\n').split('\t')
                role, label = int(role), int(label)
                if (len(digest) != 64 or not 0 <= role < 3 or not 0 <= label < len(classes)
                        or previous is not None and digest < previous):
                    raise RuntimeError('Malformed sorted split signatures')
                if digest != previous:
                    finish()
                    previous, counts, label_mask = digest, [0,0,0], 0
                counts[role] += 1
                label_mask |= 1 << label
                consumed += 1
            finish()
        if consumed != sum(audit['rows'].values()):
            raise RuntimeError('Sort changed the source fingerprint count')
        before = sum(len(heap) for heap in pools['validation'].heaps)
        audit['selected_class_counts'] = {
            role: [sum(item[4] not in excluded for item in heap) if role == 'validation' else len(heap)
                   for heap in pool.heaps] for role,pool in pools.items()}
        audit['validation_candidates_removed_for_training_overlap'] = before-sum(audit['selected_class_counts']['validation'])
        audit['sampling'] = 'seeded class-capped source-row sample; validation candidates with full-training input overlap excluded; no refill'
        if source_hash(config['archive_path']) != config['archive_sha256']:
            raise ValueError('Source archive changed during audit')
        with (output_dir/'split_audit.json').open('x') as stream:
            json.dump(audit, stream, indent=2, allow_nan=False)
        # Keep the full-source findings even when no usable pilot can be made.
        samples = {'train': pools['train'].arrays(), 'validation': pools['validation'].arrays(excluded)}
    print(json.dumps({'stage': 'split_audit_complete', 'overlap_groups': audit['overlap_groups'],
                      'validation_excluded': audit['validation_candidates_removed_for_training_overlap']}), flush=True)
    return audit, samples
