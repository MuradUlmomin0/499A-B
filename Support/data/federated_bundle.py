"""Explicit prepared client/validation inputs; no final-test references or splits.

Bundle v1 fields: schema_version, bundle_name, contract, split_policy, clients,
validation. Each client has client_id, manifest_path and evidence_path. Validation
has manifest_path/evidence_path. ID sidecars keep the existing NPZ schema intact.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from support.data.contract import ModelDataContract
from support.data.manifest import _unique_json_object, load_data_manifest
from support.data.prepared import PreparedData, load_prepared_data
from support.data.split_checks import SplitEvidence, check_disjoint_splits, file_sha256, load_split_evidence


def _name(value, field):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f'{field} must be a nonblank string without surrounding whitespace')


def _path(value, directory):
    _name(value, 'path')
    path = Path(value)
    return (path if path.is_absolute() else directory / path).resolve()


@dataclass(frozen=True)
class FederatedPartition:
    client_id: str | None
    data: PreparedData
    evidence: SplitEvidence
    manifest_sha256: str

    def source_metadata(self) -> dict:
        manifest = self.data.manifest
        return {
            'client_id': self.client_id, 'role': manifest.role,
            'dataset_name': manifest.dataset_name, 'partition_name': manifest.partition_name,
            'sample_count': len(self.data.dataset), 'testing_only': manifest.testing_only,
            'manifest_path': str(manifest.manifest_path), 'manifest_sha256': self.manifest_sha256,
            'data_path': str(manifest.data_path), 'data_sha256': self.evidence.data_sha256,
            'evidence_path': str(self.evidence.sidecar_path), 'evidence_sha256': self.evidence.sidecar_sha256,
            'preprocessing_path': str(manifest.preprocessing_path) if manifest.preprocessing_path else None,
        }


@dataclass(frozen=True)
class FederatedDataBundle:
    path: Path
    sha256: str
    name: str
    contract: ModelDataContract
    clients: tuple[FederatedPartition, ...]
    validation: FederatedPartition
    split_report: dict

    @property
    def testing_only(self) -> bool:
        return any(p.data.manifest.testing_only for p in (*self.clients, self.validation))

    def source_metadata(self) -> dict:
        """Log identities/hashes/counts, not raw features, labels or record IDs."""
        return {
            'bundle_path': str(self.path), 'bundle_sha256': self.sha256, 'bundle_name': self.name,
            'contract': self.contract.to_dict(), 'testing_only': self.testing_only,
            'clients': [p.source_metadata() for p in self.clients],
            'validation': self.validation.source_metadata(), 'split_checks': self.split_report,
        }


def _partition(description, directory, contract, role, client_id):
    path = _path(description['manifest_path'], directory)
    before_manifest = file_sha256(path)
    manifest = load_data_manifest(path)
    if manifest.role != role or manifest.client_id != client_id:
        raise ValueError(f'Partition must declare role={role!r}, client_id={client_id!r}')
    contract.assert_compatible(manifest.contract)
    before_data = file_sha256(manifest.data_path)
    data = load_prepared_data(path, expected_role=role, expected_contract=contract)
    evidence = load_split_evidence(
        _path(description['evidence_path'], directory),
        data_path=manifest.data_path, sample_count=len(data.dataset),
    )
    if (before_manifest != file_sha256(path) or before_data != evidence.data_sha256
            or manifest != data.manifest):
        raise ValueError('Prepared source changed while loading; supply stable input files')
    return FederatedPartition(client_id, data, evidence, before_manifest)


def load_federated_bundle(path: str | Path) -> FederatedDataBundle:
    """Validate bundle and membership evidence, then return canonically ordered clients.

    Training roles must be train with matching IDs; server data must be validation
    with client_id=null. Final-test/auxiliary inputs are excluded by this schema.
    Fingerprints bracket data loading to detect ordinary source changes. Producers
    must keep inputs stable and provide truthful, globally meaningful source IDs;
    these checks do not establish real-data provenance or final-test isolation.
    """
    source = Path(path).resolve()
    content = source.read_bytes()
    value = json.loads(content, object_pairs_hook=_unique_json_object)
    fields = {'schema_version', 'bundle_name', 'contract', 'split_policy', 'clients', 'validation'}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('Federated bundle fields must exactly match the version 1 schema')
    if type(value['schema_version']) is not int or value['schema_version'] != 1:
        raise ValueError('Only federated bundle schema version 1 is supported')
    _name(value['bundle_name'], 'bundle_name')
    if value['split_policy'] not in ('record', 'group'):
        raise ValueError('split_policy must be record or group')
    contract = ModelDataContract.from_dict(value['contract'])
    descriptions = value['clients']
    if not isinstance(descriptions, list) or not descriptions:
        raise ValueError('clients must be a nonempty list')
    identifiers = []
    for item in descriptions:
        if not isinstance(item, dict) or set(item) != {'client_id', 'manifest_path', 'evidence_path'}:
            raise ValueError('Each client needs client_id, manifest_path and evidence_path')
        _name(item['client_id'], 'client_id')
        identifiers.append(item['client_id'])
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate bundle client IDs')
    validation = value['validation']
    if not isinstance(validation, dict) or set(validation) != {'manifest_path', 'evidence_path'}:
        raise ValueError('Validation needs manifest_path and evidence_path only')
    clients = tuple(_partition(item, source.parent, contract, 'train', item['client_id'])
                    for item in sorted(descriptions, key=lambda item: item['client_id']))
    server_data = _partition(validation, source.parent, contract, 'validation', None)
    all_parts = (*clients, server_data)
    paths = [part.data.manifest.data_path for part in all_parts]
    if len(set(paths)) != len(paths):
        raise ValueError('A prepared data file is reused across partitions')
    evidence = {f'client:{p.client_id}': p.evidence for p in clients}
    evidence['server:validation'] = server_data.evidence
    report = check_disjoint_splits(evidence, policy=value['split_policy'])
    return FederatedDataBundle(source, hashlib.sha256(content).hexdigest(), value['bundle_name'],
                               contract, clients, server_data, report)
