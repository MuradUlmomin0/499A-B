"""Load already-prepared arrays without fitting or applying preprocessing.

NPZ artifacts contain ``features`` (rows x input_dim), ``feature_names``
(ordered Unicode names), and ``class_names`` (ordered Unicode class names).
``labels`` contains one integer class ID per row; it is optional only for the
'predict' role. Unknown or duplicate archive keys are rejected.

Names and the manifest describe the producer's intended semantics. They cannot
prove that the feature values were computed correctly or that splits are disjoint.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from support.data.contract import ModelDataContract
from support.data.datasets import tabular_dataset
from support.data.manifest import PreparedDataManifest, load_data_manifest


@dataclass(frozen=True)
class PreparedData:
    """Loaded dataset plus its manifest, including role and testing-only status.

    Labeled datasets yield (features, label); unlabeled prediction datasets yield
    (features,). The frozen wrapper does not make the PyTorch tensors immutable.
    """

    manifest: PreparedDataManifest
    dataset: TensorDataset

    @property
    def has_labels(self) -> bool:
        return len(self.dataset.tensors) == 2


def _check_names(values, expected, field):
    if values.ndim != 1 or values.dtype.kind != 'U':
        raise ValueError(f'{field} must be a one-dimensional Unicode string array')
    if values.tolist() != list(expected):
        raise ValueError(f'{field} do not match the contract order and names')


def _feature_array(values, input_dim):
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != input_dim:
        raise ValueError('features must be a nonempty [rows, input_dim] matrix')
    if values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
        raise ValueError('features must contain finite real numeric values')
    # MLP inputs use float32. Check again after conversion: finite float64 values
    # can exceed float32 range and must not become infinite model inputs.
    with np.errstate(over='ignore', invalid='ignore'):
        features = values.astype(np.float32, copy=True)
    if not np.isfinite(features).all():
        raise ValueError('features must remain finite after float32 conversion')
    return features


def _label_array(values, sample_count, num_classes):
    if values.ndim != 1 or len(values) != sample_count or values.dtype.kind not in 'iu':
        raise ValueError('labels must contain one integer class ID per feature row')
    if (values < 0).any() or (values >= num_classes).any():
        raise ValueError('labels contain IDs outside the configured class range')
    # Range checking precedes conversion so large unsigned IDs cannot wrap.
    return values.astype(np.int64, copy=True)


def load_prepared_data(
    manifest_path: str | Path,
    *,
    expected_role: str,
    expected_contract: ModelDataContract | None = None,
) -> PreparedData:
    """Read a manifest and validated NPZ arrays into a CPU TensorDataset.

    Args:
        manifest_path: Version 1 JSON manifest accepted by load_data_manifest().
        expected_role: Explicitly requested use: train, validation, auxiliary,
            test, or predict. A different declared role is rejected before
            opening the data archive. No partition is repurposed automatically.
        expected_contract: Optional model-side contract to compare with the
            manifest. Supply it when a model has already been selected.

    Returns:
        PreparedData with the original validated manifest and a dataset of
        float32 features plus int64 labels, or features alone for unlabeled
        prediction. Row order is preserved; data is not shuffled or modified.

    Raises:
        ValueError: Roles, contracts, archive keys, names, or arrays are invalid.
        TypeError: expected_contract is not a ModelDataContract when supplied.
        OSError: A referenced file cannot be read. Manifest validation may also
            raise FileNotFoundError for absent files.

    Loading uses allow_pickle=False and does not deserialize Python objects.
    This is an in-memory adapter for prepared tabular artifacts, not a streaming
    raw-dataset reader. No scaling, label remapping, splitting, or training occurs.
    """
    if expected_role not in ('train', 'validation', 'auxiliary', 'test', 'predict'):
        raise ValueError('expected_role must be a supported data role')
    if expected_contract is not None and not isinstance(expected_contract, ModelDataContract):
        raise TypeError('expected_contract must be a ModelDataContract')

    manifest = load_data_manifest(manifest_path)
    if manifest.role != expected_role:
        raise ValueError(f'Expected role {expected_role!r}, but manifest declares {manifest.role!r}')
    if expected_contract is not None:
        expected_contract.assert_compatible(manifest.contract)

    archive = np.load(manifest.data_path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError('Prepared artifact must be an NPZ archive')
    with archive:
        names = archive.files
        required = {'features', 'feature_names', 'class_names'}
        if manifest.role != 'predict':
            required.add('labels')
        if len(names) != len(set(names)):
            raise ValueError('Prepared archive contains duplicate keys')
        if not required.issubset(names) or set(names) - (required | {'labels'}):
            raise ValueError('Prepared archive has missing or unsupported array keys')

        contract = manifest.contract
        _check_names(archive['feature_names'], contract.feature_names, 'feature_names')
        _check_names(archive['class_names'], contract.class_names, 'class_names')
        features = _feature_array(archive['features'], contract.input_dim)
        labels = None
        if 'labels' in names:
            labels = _label_array(archive['labels'], len(features), contract.num_classes)

    if labels is None:
        dataset = TensorDataset(torch.from_numpy(features).clone())
    else:
        dataset = tabular_dataset(features, labels, contract.input_dim, contract.num_classes)
    return PreparedData(manifest=manifest, dataset=dataset)
