"""Bounded unique-group sampling and full-archive, disk-backed fingerprint audit."""
import hashlib
import heapq
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import numpy as np
from Support.data.ciciot2023 import CLASS_NAMES, SUBTYPES, stream_chunks


def feature_digest(row):
    """SHA-256 of ordered canonical little-endian float32 inputs, without labels."""
    values = np.array(row, dtype='<f4', copy=True)
    values[values == 0] = 0
    return hashlib.sha256(values.tobytes()).digest()


def _quotas(counts, target):
    target = min(target, sum(counts))
    shares = np.asarray(counts, dtype=np.float64) * target / sum(counts)
    quotas = np.minimum(counts, np.maximum(4, np.floor(shares).astype(np.int64)))
    while quotas.sum() > target:
        quotas[np.argmax(quotas - 4)] -= 1
    while quotas.sum() < target:
        choices = [i for i in range(8) if quotas[i] < counts[i]]
        i = max(choices, key=lambda k: (shares[k] - quotas[k], -k))
        quotas[i] += 1
    return quotas.tolist()


class _BottomK:
    """Only cap group payloads and cap heap/dictionary entries can be retained."""
    def __init__(self, cap, dimensions, seed):
        self.features = np.empty((cap, dimensions), dtype='<f4')
        self.hashes = np.empty(cap, dtype='V32')
        self.files = np.empty(cap, dtype=np.int32)
        self.rows = np.empty(cap, dtype=np.uint64)
        self.subtypes = np.empty(cap, dtype=np.uint8)
        self.live = np.zeros(cap, dtype=bool)
        self.cap, self.seed, self.heap, self.lookup = cap, seed.to_bytes(4, 'little'), [], {}

    def offer(self, digest, row, file_index, ordinal, subtype):
        if digest in self.lookup:
            return 0
        rank = int.from_bytes(hashlib.blake2b(self.seed + digest, digest_size=16).digest(), 'big')
        growing = len(self.heap) < self.cap
        if growing:
            slot = len(self.heap)
            heapq.heappush(self.heap, (-rank, slot))
        elif rank < -self.heap[0][0]:
            _, slot = heapq.heapreplace(self.heap, (-rank, self.heap[0][1]))
            del self.lookup[self.hashes[slot].tobytes()]
        else:
            return 0
        self.lookup[digest] = slot
        self.features[slot], self.hashes[slot] = row, np.void(digest)
        self.files[slot], self.rows[slot], self.subtypes[slot] = file_index, ordinal, subtype
        self.live[slot] = True
        return int(growing)

    def discard(self, digest):
        slot = self.lookup.pop(digest, None)
        if slot is not None:
            self.live[slot] = False

    def arrays(self):
        selected = np.flatnonzero(self.live)
        selected = selected[np.argsort(self.hashes[selected])]
        return {'features': self.features[selected], 'hashes': self.hashes[selected],
                'file_indices': self.files[selected], 'row_ordinals': self.rows[selected],
                'subtypes': self.subtypes[selected]}


def _limit_sort_memory():
    ceiling = 512 * 1024**2
    resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))


def _external_sort(source, destination, work_dir):
    executable = shutil.which('sort')
    if executable is None:
        raise RuntimeError('External sort is required for the disk-backed audit')
    environment = {**os.environ, 'LC_ALL': 'C'}
    # Fixed-width digest + tab + one-digit class sorts correctly bytewise.
    # Avoid per-field key allocations; leave room beyond the 32 MiB input buffer
    # for sort/merge structures under the independent address-space ceiling.
    result = subprocess.run([executable, '--parallel=1', '-S', '32M', '--batch-size=16',
                             '-T', str(work_dir), str(source), '-o', str(destination)],
                            env=environment, capture_output=True, text=True, preexec_fn=_limit_sort_memory)
    if result.returncode or result.stderr.strip() or not destination.is_file():
        raise RuntimeError(f'External sort failed: {result.stderr.strip()} (code {result.returncode})')


def audit_and_sample(config, work_dir, progress):
    """Return bounded eligible groups after auditing every valid source row.

    Conflicting groups are removed globally, even if one conflicting row was
    never retained. No hidden refill occurs. Class quotas approximate original
    valid-row proportions; unique-group sampling changes the population and is
    diagnostic, not an unbiased row-weighted research estimate.
    """
    work_dir = Path(work_dir)
    quotas = _quotas([config['source']['expected_valid_counts'][c] for c in CLASS_NAMES], config['pilot']['max_rows'])
    samplers = [_BottomK(q, len(config['source']['feature_names']), config['pilot']['seed']) for q in quotas]
    stats, retained, last_file = {}, 0, -1
    signatures = work_dir / 'fingerprints.tsv'
    sorted_signatures = work_dir / 'fingerprints.sorted.tsv'
    with signatures.open('x', buffering=1024*1024) as output, (work_dir/'quarantine.jsonl').open('x') as quarantine:
        for chunk in stream_chunks(config, quarantine, stats):
            for row, ordinal in zip(chunk['features'], chunk['row_ordinals']):
                digest = hashlib.sha256(row.tobytes()).digest()  # stream canonicalizes dtype and signed zero.
                output.write(f"{digest.hex()}\t{chunk['label']}\n")
                retained += samplers[chunk['label']].offer(digest, row, chunk['file_index'], int(ordinal), chunk['subtype'])
            if chunk['file_index'] != last_file:
                last_file = chunk['file_index']
                progress({'stage': 'streaming', 'member_index': last_file, 'rows': stats['rows'], 'retained_groups': retained})
    if stats['valid_counts'] != config['source']['expected_valid_counts']:
        raise ValueError(f"Actual valid counts differ from pinned counts: {stats['valid_counts']}")
    progress({'stage': 'external_sort', 'signature_bytes': signatures.stat().st_size})
    _external_sort(signatures, sorted_signatures, work_dir)
    audit = {'valid_rows': sum(stats['valid_counts'].values()), 'unique_feature_groups': 0,
             'duplicate_excess_rows': 0, 'conflicting_groups': 0, 'conflicting_rows': 0,
             'eligible_unique_groups_by_class': dict.fromkeys(CLASS_NAMES, 0),
             'peak_retained_groups': retained, 'quota_by_class': dict(zip(CLASS_NAMES, quotas)),
             'source': stats, 'fingerprint': 'SHA-256 of canonical little-endian float32 ordered input bytes; signed zero normalized',
             'hash_collision_verification': False, 'near_duplicates_audited': False}
    previous, count, mask, consumed = None, 0, 0, 0
    with (work_dir/'conflicting_groups.jsonl').open('x') as conflicts:
        def finish_group():
            if previous is None:
                return
            audit['unique_feature_groups'] += 1
            audit['duplicate_excess_rows'] += count - 1
            if mask.bit_count() > 1:
                audit['conflicting_groups'] += 1
                audit['conflicting_rows'] += count
                digest = bytes.fromhex(previous)
                for sampler in samplers:
                    sampler.discard(digest)
                conflicts.write(json.dumps({'group': previous, 'class_mask': mask, 'rows': count})+'\n')
            else:
                audit['eligible_unique_groups_by_class'][CLASS_NAMES[mask.bit_length()-1]] += 1
        with sorted_signatures.open() as stream:
            for line in stream:
                digest, label_text = line.rstrip('\n').split('\t')
                label = int(label_text)
                if len(digest) != 64 or not 0 <= label < 8 or (previous is not None and digest < previous):
                    raise RuntimeError('Malformed or unsorted fingerprint output')
                if digest != previous:
                    finish_group()
                    previous, count, mask = digest, 0, 0
                count += 1
                mask |= 1 << label
                consumed += 1
            finish_group()
    if consumed != audit['valid_rows']:
        raise RuntimeError('External sort changed the fingerprint row count')
    groups = [sampler.arrays() for sampler in samplers]
    sizes = [len(x['features']) for x in groups]
    if min(sizes) < 4:
        raise ValueError(f'Fewer than four eligible retained groups in a class; no refill: {dict(zip(CLASS_NAMES, sizes))}')
    sample = {key: np.concatenate([g[key] for g in groups]) for key in groups[0]}
    sample['labels'] = np.concatenate([np.full(n, i, dtype=np.int64) for i, n in enumerate(sizes)])
    sample['members'] = stats['members']
    sample['subtype_names'] = SUBTYPES
    audit['retained_groups_by_class'] = dict(zip(CLASS_NAMES, sizes))
    audit['retained_groups'] = sum(sizes)
    progress({'stage': 'group_audit_complete', 'retained_groups': sum(sizes), 'conflicting_groups': audit['conflicting_groups']})
    return sample, audit
