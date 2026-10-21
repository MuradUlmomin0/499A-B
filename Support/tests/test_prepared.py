"""Verify prepared-array loading with tiny synthetic files and no training."""

import copy
import io
import json
import warnings
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from support.data.contract import ModelDataContract
from support.data.prepared import load_prepared_data


@pytest.fixture
def prepared_artifact(tmp_path):
    path = tmp_path / 'manifest.json'
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
    arrays = {
        'features': np.array([[1., 2.], [-3., 4.]], dtype=np.float64),
        'labels': np.array([1, 0], dtype=np.int64),
        'feature_names': np.array(['feature_a', 'feature_b']),
        'class_names': np.array(['class_a', 'class_b']),
    }
    return path, description, arrays


def write_artifact(path, description, arrays):
    path.write_text(json.dumps(description), encoding='utf-8')
    np.savez(path.parent / 'fixture.npz', **arrays)


def test_values_row_order_and_cpu_dtypes(prepared_artifact):
    path, description, arrays = prepared_artifact
    write_artifact(path, description, arrays)

    loaded = load_prepared_data(path, expected_role='test')
    features, labels = loaded.dataset.tensors

    assert loaded.has_labels and len(loaded.dataset) == 2
    assert loaded.manifest.testing_only is True
    assert features.dtype == torch.float32
    assert labels.dtype == torch.int64
    assert features.device.type == 'cpu'
    assert features.tolist() == [[1., 2.], [-3., 4.]]
    assert labels.tolist() == [1, 0]
    batch = next(iter(DataLoader(loaded.dataset, batch_size=2)))
    assert torch.equal(batch[0], features)
    assert torch.equal(batch[1], labels)


@pytest.mark.parametrize('role', ['train', 'validation', 'auxiliary', 'test', 'predict'])
def test_labeled_roles(prepared_artifact, role):
    path, description, arrays = prepared_artifact
    description['role'] = role
    write_artifact(path, description, arrays)

    assert load_prepared_data(path, expected_role=role).has_labels


def test_unlabeled_prediction(prepared_artifact):
    path, description, arrays = prepared_artifact
    description['role'] = 'predict'
    del arrays['labels']
    write_artifact(path, description, arrays)

    loaded = load_prepared_data(path, expected_role='predict')

    assert not loaded.has_labels
    assert len(loaded.dataset[0]) == 1
    batch = next(iter(DataLoader(loaded.dataset, batch_size=2)))
    assert batch[0].shape == (2, 2)


@pytest.mark.parametrize('role', ['train', 'validation', 'auxiliary', 'test'])
def test_non_prediction_roles_require_labels(prepared_artifact, role):
    path, description, arrays = prepared_artifact
    description['role'] = role
    del arrays['labels']
    write_artifact(path, description, arrays)

    with pytest.raises(ValueError):
        load_prepared_data(path, expected_role=role)


def test_role_mismatch_precedes_array_loading(prepared_artifact, monkeypatch):
    path, description, arrays = prepared_artifact
    write_artifact(path, description, arrays)

    def forbidden_load(*args, **kwargs):
        raise AssertionError('Arrays should not be opened for the wrong role')

    monkeypatch.setattr(np, 'load', forbidden_load)

    with pytest.raises(ValueError, match='Expected role'):
        load_prepared_data(path, expected_role='train')


def test_incompatible_contract_precedes_array_loading(prepared_artifact, monkeypatch):
    path, description, arrays = prepared_artifact
    write_artifact(path, description, arrays)
    changed = copy.deepcopy(description['contract'])
    changed['feature_names'].reverse()
    expected = ModelDataContract.from_dict(changed)

    def forbidden_load(*args, **kwargs):
        raise AssertionError('Arrays should not be opened for an incompatible contract')

    monkeypatch.setattr(np, 'load', forbidden_load)

    with pytest.raises(ValueError, match='feature_names'):
        load_prepared_data(path, expected_role='test', expected_contract=expected)


def test_matching_model_contract(prepared_artifact):
    path, description, arrays = prepared_artifact
    write_artifact(path, description, arrays)
    expected = ModelDataContract.from_dict(description['contract'])

    loaded = load_prepared_data(path, expected_role='test', expected_contract=expected)

    assert loaded.has_labels


@pytest.mark.parametrize('array,value', [
    ('features', np.ones(2)),
    ('features', np.ones((0, 2))),
    ('features', np.ones((2, 3))),
    ('features', np.array([[np.nan, 0], [0, 0]])),
    ('features', np.array([[np.inf, 0], [0, 0]])),
    ('features', np.full((2, 2), np.finfo(np.float64).max)),
    ('features', np.ones((2, 2), dtype=complex)),
    ('features', np.ones((2, 2), dtype=bool)),
    ('features', np.array([['1', '2'], ['3', '4']])),
    ('features', np.array([[1, 2], [3, 4]], dtype=object)),
    ('feature_names', np.array(['feature_b', 'feature_a'])),
    ('feature_names', np.array(['feature_a', 'feature_a'])),
    ('feature_names', np.array([['feature_a', 'feature_b']])),
    ('feature_names', np.array(['feature_a', 'feature_b'], dtype=object)),
    ('feature_names', np.array(['feature_a', 'feature_b'], dtype='S')),
    ('class_names', np.array(['class_b', 'class_a'])),
    ('class_names', np.array(['class_a'])),
    ('class_names', np.array([0, 1])),
    ('labels', np.array([0., 1.])),
    ('labels', np.array([False, True])),
    ('labels', np.array([[0], [1]])),
    ('labels', np.array([-1, 0])),
    ('labels', np.array([0, 2])),
    ('labels', np.array([0])),
    ('labels', np.array([0, 1], dtype=object)),
    ('labels', np.array([0, np.iinfo(np.uint64).max], dtype=np.uint64)),
])
def test_invalid_arrays_are_rejected(prepared_artifact, array, value):
    path, description, arrays = prepared_artifact
    arrays[array] = value
    write_artifact(path, description, arrays)

    with pytest.raises(ValueError):
        load_prepared_data(path, expected_role='test')


@pytest.mark.parametrize('name', ['features', 'feature_names', 'class_names'])
def test_missing_required_arrays(prepared_artifact, name):
    path, description, arrays = prepared_artifact
    del arrays[name]
    write_artifact(path, description, arrays)

    with pytest.raises(ValueError):
        load_prepared_data(path, expected_role='test')


def test_unknown_arrays_are_rejected(prepared_artifact):
    path, description, arrays = prepared_artifact
    arrays['unexpected'] = np.zeros(2)
    write_artifact(path, description, arrays)

    with pytest.raises(ValueError):
        load_prepared_data(path, expected_role='test')


def test_duplicate_archive_keys(prepared_artifact):
    path, description, arrays = prepared_artifact
    write_artifact(path, description, arrays)
    buffer = io.BytesIO()
    np.save(buffer, arrays['features'])
    # Deliberately malformed archive: the loader must not choose between copies.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with ZipFile(path.parent / 'fixture.npz', 'a') as archive:
            archive.writestr('features.npy', buffer.getvalue())

    with pytest.raises(ValueError, match='duplicate'):
        load_prepared_data(path, expected_role='test')


def test_npy_file_is_not_npz(prepared_artifact):
    path, description, arrays = prepared_artifact
    write_artifact(path, description, arrays)
    with (path.parent / 'fixture.npz').open('wb') as stream:
        np.save(stream, arrays['features'])

    with pytest.raises(ValueError, match='NPZ'):
        load_prepared_data(path, expected_role='test')


def test_corrupt_archive(prepared_artifact):
    path, description, arrays = prepared_artifact
    write_artifact(path, description, arrays)
    (path.parent / 'fixture.npz').write_bytes(b'not a numpy file')

    with pytest.raises(ValueError):
        load_prepared_data(path, expected_role='test')


def test_prediction_does_not_ignore_invalid_supplied_labels(prepared_artifact):
    path, description, arrays = prepared_artifact
    description['role'] = 'predict'
    arrays['labels'] = np.array([0, 2])
    write_artifact(path, description, arrays)

    with pytest.raises(ValueError):
        load_prepared_data(path, expected_role='predict')


def test_unsigned_labels_normalized_after_range_check(prepared_artifact):
    path, description, arrays = prepared_artifact
    arrays['labels'] = np.array([0, 1], dtype=np.uint64)
    write_artifact(path, description, arrays)

    loaded = load_prepared_data(path, expected_role='test')

    assert loaded.dataset.tensors[1].dtype == torch.int64


def test_reading_and_tensor_edits_do_not_change_files(prepared_artifact):
    path, description, arrays = prepared_artifact
    write_artifact(path, description, arrays)
    before = {entry: entry.read_bytes() for entry in path.parent.iterdir()}
    loaded = load_prepared_data(path, expected_role='test')
    loaded.dataset.tensors[0].zero_()
    loaded.dataset.tensors[1].zero_()

    assert before == {entry: entry.read_bytes() for entry in path.parent.iterdir()}
    again = load_prepared_data(path, expected_role='test')
    assert again.dataset.tensors[0].tolist() == [[1., 2.], [-3., 4.]]
    assert again.dataset.tensors[1].tolist() == [1, 0]


def test_invalid_expected_role(prepared_artifact):
    path, _, _ = prepared_artifact

    with pytest.raises(ValueError):
        load_prepared_data(path, expected_role='unknown')


def test_invalid_expected_contract_type(prepared_artifact):
    path, _, _ = prepared_artifact

    with pytest.raises(TypeError):
        load_prepared_data(path, expected_role='test', expected_contract={})
