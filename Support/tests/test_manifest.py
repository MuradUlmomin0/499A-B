"""Verify prepared-data manifests using temporary synthetic fixtures only."""

from dataclasses import FrozenInstanceError
import hashlib
import json

import numpy as np
import pytest

from support.data.manifest import load_data_manifest


@pytest.fixture
def prepared_files(tmp_path):
    directory = tmp_path / 'prepared'
    directory.mkdir()
    data_path = directory / 'fixture.npz'
    np.savez(
        data_path,
        features=np.zeros((2, 2), dtype=np.float32),
        labels=np.array([0, 1], dtype=np.int64),
    )
    manifest_path = directory / 'manifest.json'
    description = {
        'schema_version': 1,
        'dataset_name': 'test_fixture',
        'partition_name': 'sample',
        'role': 'test',
        'client_id': None,
        'data_format': 'npz',
        'data_path': 'fixture.npz',
        'preprocessing_path': None,
        'testing_only': True,
        'contract': {
            'schema_version': 1,
            'model_type': 'mlp',
            'input_dim': 2,
            'num_classes': 2,
            'hidden_dims': [4],
            'feature_names': ['feature_a', 'feature_b'],
            'class_names': ['class_a', 'class_b'],
            'preprocessing_sha256': None,
        },
    }
    return manifest_path, description, data_path


def write_manifest(path, description):
    path.write_text(json.dumps(description), encoding='utf-8')


@pytest.mark.parametrize('role', ['train', 'validation', 'auxiliary', 'test', 'predict'])
def test_declared_roles_and_relative_paths(prepared_files, monkeypatch, tmp_path, role):
    path, description, data_path = prepared_files
    description['role'] = role
    description['client_id'] = 'client_1' if role == 'train' else None
    write_manifest(path, description)
    # A relative data path belongs to the manifest, not the terminal directory.
    monkeypatch.chdir(tmp_path)

    result = load_data_manifest(path)

    assert result.data_path == data_path.resolve()
    assert result.manifest_path == path.resolve()
    assert result.role == role
    assert result.client_id == description['client_id']
    assert result.testing_only is True
    assert result.contract.class_names == ('class_a', 'class_b')


def test_absolute_data_path(prepared_files):
    path, description, data_path = prepared_files
    description['data_path'] = str(data_path)
    write_manifest(path, description)

    assert load_data_manifest(path).data_path == data_path


def test_matching_preprocessing_fingerprint(prepared_files):
    path, description, _ = prepared_files
    artifact = path.parent / 'preprocessing.json'
    # This is a fingerprint fixture, not a fitted scaler or training artifact.
    artifact.write_bytes(b'{"purpose":"test-only fingerprint fixture"}')
    description['preprocessing_path'] = artifact.name
    description['contract']['preprocessing_sha256'] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    write_manifest(path, description)

    result = load_data_manifest(path)

    assert result.preprocessing_path == artifact
    assert result.contract.preprocessing_sha256 == description['contract']['preprocessing_sha256']


def test_changed_preprocessing_is_rejected(prepared_files):
    path, description, _ = prepared_files
    artifact = path.parent / 'preprocessing.json'
    artifact.write_bytes(b'first test version')
    description['preprocessing_path'] = artifact.name
    description['contract']['preprocessing_sha256'] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    write_manifest(path, description)
    artifact.write_bytes(b'different test version')

    with pytest.raises(ValueError, match='SHA-256'):
        load_data_manifest(path)


@pytest.mark.parametrize('field,value', [
    ('schema_version', 2),
    ('schema_version', True),
    ('role', 'mystery'),
    ('data_format', 'csv'),
    ('testing_only', 'false'),
    ('testing_only', None),
    ('client_id', 1),
    ('client_id', ' '),
    ('dataset_name', ''),
    ('partition_name', ' sample'),
    ('data_path', ''),
    ('data_path', 1),
    ('contract', None),
])
def test_rejects_invalid_declarations(prepared_files, field, value):
    path, description, _ = prepared_files
    description[field] = value
    write_manifest(path, description)

    with pytest.raises(ValueError):
        load_data_manifest(path)


@pytest.mark.parametrize('field', ['contract', 'testing_only', 'role', 'data_path'])
def test_required_metadata_is_not_guessed(prepared_files, field):
    path, description, _ = prepared_files
    del description[field]
    write_manifest(path, description)

    with pytest.raises(ValueError):
        load_data_manifest(path)


@pytest.mark.parametrize('nested', [False, True])
def test_duplicate_fields_are_rejected(prepared_files, nested):
    path, description, _ = prepared_files
    text = json.dumps(description)
    if nested:
        text = text.replace('"input_dim": 2', '"input_dim": 2, "input_dim": 3')
    else:
        text = text[:-1] + ', "role": "train"}'
    path.write_text(text, encoding='utf-8')

    with pytest.raises(ValueError, match='Duplicate'):
        load_data_manifest(path)


@pytest.mark.parametrize('description', [[], None])
def test_top_level_must_be_object(prepared_files, description):
    path, _, _ = prepared_files
    write_manifest(path, description)

    with pytest.raises(ValueError):
        load_data_manifest(path)


@pytest.mark.parametrize('missing', ['manifest', 'data', 'preprocessing'])
def test_missing_files_are_rejected(prepared_files, missing):
    path, description, _ = prepared_files
    if missing == 'data':
        description['data_path'] = 'missing.npz'
    elif missing == 'preprocessing':
        description['preprocessing_path'] = 'missing.json'
        description['contract']['preprocessing_sha256'] = 'a' * 64
    if missing != 'manifest':
        write_manifest(path, description)

    with pytest.raises(FileNotFoundError):
        load_data_manifest(path)


@pytest.mark.parametrize('problem', [
    'path_without_hash', 'hash_without_path', 'non_string_path',
])
def test_preprocessing_declarations_must_agree(prepared_files, problem):
    path, description, _ = prepared_files
    if problem == 'path_without_hash':
        description['preprocessing_path'] = 'unused.json'
    else:
        description['contract']['preprocessing_sha256'] = 'a' * 64
        description['preprocessing_path'] = None if problem == 'hash_without_path' else 1
    write_manifest(path, description)

    with pytest.raises(ValueError):
        load_data_manifest(path)


def test_reader_does_not_modify_files_and_result_is_immutable(prepared_files):
    path, description, _ = prepared_files
    write_manifest(path, description)
    before = {entry: entry.read_bytes() for entry in path.parent.iterdir()}

    result = load_data_manifest(path)

    assert before == {entry: entry.read_bytes() for entry in path.parent.iterdir()}
    with pytest.raises(FrozenInstanceError):
        result.role = 'train'


def test_malformed_json(prepared_files):
    path, _, _ = prepared_files
    path.write_text('{', encoding='utf-8')

    with pytest.raises(ValueError):
        load_data_manifest(path)


def test_directory_is_not_data_file(prepared_files):
    path, description, _ = prepared_files
    description['data_path'] = '.'
    write_manifest(path, description)

    with pytest.raises(FileNotFoundError):
        load_data_manifest(path)


def test_unknown_field_rejected(prepared_files):
    path, description, _ = prepared_files
    description['unexpected'] = True
    write_manifest(path, description)

    with pytest.raises(ValueError):
        load_data_manifest(path)
