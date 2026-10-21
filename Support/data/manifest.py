"""Read an explicit description of one already-prepared data artifact.

Version 1 uses JSON with these exact fields:
    schema_version, dataset_name, partition_name, role, client_id, data_format,
    data_path, preprocessing_path, contract, testing_only.

``contract`` contains a ModelDataContract.to_dict() description. ``data_format``
must be 'npz': the supported exchange format for prepared arrays, not raw traffic.
The actual array loader will check the file contents in a later component.

Paths may be absolute or relative to the manifest's directory. ``client_id`` is
a nonblank string or explicit null for an unassigned/global artifact. Each
artifact declares its role; no partitioning or client mapping is inferred.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from support.data.contract import ModelDataContract


@dataclass(frozen=True)
class PreparedDataManifest:
    """Validated declarations and resolved local paths for a prepared artifact.

    ``testing_only`` is an explicit producer declaration, not proof of real-data
    provenance. The manifest cannot prove that partitions contain disjoint rows
    or that preprocessing was fitted only on training data.

    ``preprocessing_path`` points to the exact artifact identified by the
    contract's SHA-256, or is None when no preprocessing is declared. Prepared
    features must already have that transformation applied; this reader never
    applies or fits it. It verifies the artifact hash without deserializing it.
    """

    manifest_path: Path
    dataset_name: str
    partition_name: str
    role: str
    client_id: str | None
    data_format: str
    data_path: Path
    preprocessing_path: Path | None
    contract: ModelDataContract
    testing_only: bool


def _unique_json_object(pairs):
    """Reject duplicate keys rather than silently discarding a declaration."""
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f'Duplicate manifest field: {name}')
        result[name] = value
    return result


def _require_name(value, field):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f'{field} must be a nonblank string without surrounding whitespace')


def _resolve_file(value, field, directory):
    _require_name(value, field)
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = directory / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f'{field} is not an existing file: {candidate}')
    return candidate


def load_data_manifest(path: str | Path) -> PreparedDataManifest:
    """Read and validate a version 1 JSON manifest without loading its arrays.

    Roles are 'train', 'validation', 'auxiliary', 'test', or 'predict'. The caller
    must enforce role-specific use: in particular, final test data must not be
    used for training, aggregation decisions, or checkpoint selection.

    Raises:
        FileNotFoundError: A manifest, data, or declared preprocessing file is
            missing or is a directory.
        ValueError: JSON, schema, contract, declarations, or preprocessing hash
            are invalid. Missing metadata is never filled with guessed values.

    No dataset is downloaded, generated, partitioned, or preprocessed. Actual
    array validation and cross-partition leakage checks remain separate work.
    """
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Manifest file not found: {manifest_path}')
    description = json.loads(
        manifest_path.read_text(encoding='utf-8'),
        object_pairs_hook=_unique_json_object,
    )
    required = {
        'schema_version', 'dataset_name', 'partition_name', 'role', 'client_id',
        'data_format', 'data_path', 'preprocessing_path', 'contract', 'testing_only',
    }
    if not isinstance(description, dict) or set(description) != required:
        raise ValueError('Manifest fields must exactly match the version 1 schema')
    version = description['schema_version']
    if type(version) is not int or version != 1:
        raise ValueError('Only manifest schema version 1 is supported')
    for field in ('dataset_name', 'partition_name', 'role', 'data_format'):
        _require_name(description[field], field)
    if description['role'] not in ('train', 'validation', 'auxiliary', 'test', 'predict'):
        raise ValueError('Unsupported artifact role')
    if description['client_id'] is not None:
        _require_name(description['client_id'], 'client_id')
    if description['data_format'] != 'npz':
        raise ValueError('Only prepared npz artifacts are supported in version 1')
    if type(description['testing_only']) is not bool:
        raise ValueError('testing_only must be an explicit boolean')

    contract = ModelDataContract.from_dict(description['contract'])
    data_path = _resolve_file(description['data_path'], 'data_path', manifest_path.parent)
    preprocessing_path = None
    if contract.preprocessing_sha256 is None:
        if description['preprocessing_path'] is not None:
            raise ValueError('preprocessing_path must be null when no preprocessing is declared')
    else:
        preprocessing_path = _resolve_file(
            description['preprocessing_path'], 'preprocessing_path', manifest_path.parent,
        )
        with preprocessing_path.open('rb') as stream:
            actual_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual_hash != contract.preprocessing_sha256:
            raise ValueError('Preprocessing artifact does not match the contract SHA-256')

    return PreparedDataManifest(
        manifest_path=manifest_path,
        dataset_name=description['dataset_name'],
        partition_name=description['partition_name'],
        role=description['role'],
        client_id=description['client_id'],
        data_format=description['data_format'],
        data_path=data_path,
        preprocessing_path=preprocessing_path,
        contract=contract,
        testing_only=description['testing_only'],
    )
