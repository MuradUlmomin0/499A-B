"""Versioned ID sidecars and explicit checks of declared partition membership.

IDs must identify original source records globally within source_namespace,
not freshly assigned row numbers in each exported partition. The producer is
responsible for truthful IDs and row alignment. Hashes/counts bind declarations
to files; they cannot prove preprocessing or identity claims are correct.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from support.data.manifest import _unique_json_object


def file_sha256(path: str | Path) -> str:
    """Fingerprint a file without loading its entire contents into memory."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _name(value, field):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f'{field} must be a nonblank string without surrounding whitespace')


@dataclass(frozen=True)
class SplitEvidence:
    source_namespace: str
    data_sha256: str
    record_ids: tuple[str, ...]
    group_ids: tuple[str, ...] | None
    sidecar_path: Path
    sidecar_sha256: str


def load_split_evidence(path: str | Path, *, data_path: str | Path, sample_count: int) -> SplitEvidence:
    """Read the exact v1 sidecar fields and verify data hash, count and ID types.

    Schema: schema_version=1, source_namespace, data_sha256, record_ids,
    group_ids (explicit null or one nonblank group ID per row). Group IDs may
    repeat inside a partition. Record IDs must not repeat inside a partition.
    """
    if type(sample_count) is not int or sample_count < 1:
        raise ValueError('sample_count must be a positive integer')
    source = Path(path).resolve()
    content = source.read_bytes()
    value = json.loads(content, object_pairs_hook=_unique_json_object)
    required = {'schema_version', 'source_namespace', 'data_sha256', 'record_ids', 'group_ids'}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError('Split sidecar fields must exactly match the version 1 schema')
    if type(value['schema_version']) is not int or value['schema_version'] != 1:
        raise ValueError('Only split sidecar schema version 1 is supported')
    _name(value['source_namespace'], 'source_namespace')
    fingerprint = value['data_sha256']
    if not isinstance(fingerprint, str) or re.fullmatch(r'[0-9a-f]{64}', fingerprint) is None:
        raise ValueError('data_sha256 must be 64 lowercase hexadecimal characters')
    if fingerprint != file_sha256(data_path):
        raise ValueError('Sidecar data fingerprint does not match the NPZ file')
    for field in ('record_ids', 'group_ids'):
        values = value[field]
        if field == 'group_ids' and values is None:
            continue
        if not isinstance(values, list) or len(values) != sample_count:
            raise ValueError(f'{field} must contain one ID per data row')
        for identifier in values:
            _name(identifier, field)
    if len(set(value['record_ids'])) != sample_count:
        raise ValueError('Duplicate record IDs inside a partition')
    return SplitEvidence(
        value['source_namespace'], fingerprint, tuple(value['record_ids']),
        None if value['group_ids'] is None else tuple(value['group_ids']),
        source, hashlib.sha256(content).hexdigest(),
    )


def check_disjoint_splits(partitions: Mapping[str, SplitEvidence], *, policy: str) -> dict:
    """Check declared record IDs, plus group IDs when policy='group'.

    Explicit policy='record' checks records only; policy='group' additionally
    requires groups disjoint across all supplied partitions. A standalone caller
    may supply test/auxiliary evidence for an external split audit. The training
    bundle supplies training/validation only and does not inspect final test data.
    """
    if policy not in ('record', 'group'):
        raise ValueError('Split policy must be record or group')
    if not isinstance(partitions, Mapping) or len(partitions) < 2:
        raise ValueError('Supply at least two named partitions')
    namespaces, records, groups = set(), {}, {}
    counts = {}
    for name, evidence in partitions.items():
        _name(name, 'partition name')
        if not isinstance(evidence, SplitEvidence):
            raise TypeError('Each partition must have SplitEvidence from its sidecar loader')
        namespaces.add(evidence.source_namespace)
        counts[name] = len(evidence.record_ids)
        for identifier in evidence.record_ids:
            if identifier in records:
                raise ValueError(f'Declared record overlap between {records[identifier]} and {name}')
            records[identifier] = name
        if policy == 'group':
            if evidence.group_ids is None:
                raise ValueError('Group policy requires group IDs for every partition')
            for identifier in set(evidence.group_ids):
                if identifier in groups:
                    raise ValueError(f'Declared group overlap between {groups[identifier]} and {name}')
                groups[identifier] = name
    if len(namespaces) != 1:
        raise ValueError('Partitions must share one globally meaningful source_namespace')
    return {
        'policy': policy, 'source_namespace': next(iter(namespaces)),
        'partition_counts': counts, 'declared_record_ids_disjoint': True,
        'declared_group_ids_disjoint': True if policy == 'group' else None,
        'provenance_verified': False,
        'limitations': 'Checks supplied ID declarations and file fingerprints, not truth of provenance, preprocessing or row alignment. Final-test isolation needs a separate audit when test evidence is not supplied.',
    }
