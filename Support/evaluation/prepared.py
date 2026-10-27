"""Evaluate a loaded model artifact against explicitly selected prepared data.

This module coordinates compatibility checks, inference, and the existing
classification metrics. It does not load files, preprocess data, train a model,
select a checkpoint, or write experiment results.
"""

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from support.data.contract import ModelDataContract
from support.data.manifest import PreparedDataManifest
from support.data.prepared import PreparedData
from support.evaluation.metrics import classification_metrics
from support.evaluation.predict import _check_logits, _model_device
from support.models.artifact import ModelArtifact


@dataclass(frozen=True)
class PreparedEvaluationResult:
    """Metrics and source declarations captured for a later experiment logger.

    ``testing_only`` is true if either input declares testing-only use. False
    does not prove provenance, correct preprocessing, split isolation, or model
    quality. ``training_completed`` retains the model producer's declaration.

    The model fingerprint identifies the file originally read by its loader;
    this helper does not rehash a model that callers may have changed in memory.
    The frozen wrapper does not make the metrics dictionary immutable.
    """

    manifest: PreparedDataManifest
    model_name: str
    artifact_path: Path
    artifact_sha256: str
    training_completed: bool
    testing_only: bool
    sample_count: int
    metrics: dict


def evaluate_prepared_data(
    artifact: ModelArtifact,
    data: PreparedData,
    *,
    expected_role: str,
    batch_size: int = 256,
) -> PreparedEvaluationResult:
    """Evaluate every labeled row once without updating the model.

    Args:
        artifact: Model and its independently recorded contract, normally from
            load_model_artifact(). Use the intended loaded weights; the model
            remains on its current CPU/CUDA device.
        data: Labeled prepared data, normally from load_prepared_data(). Feature
            dtype must suit the model; the current loaders use float32 MLPs.
        expected_role: Explicit train, validation, auxiliary, test, or predict
            role, which must equal the manifest declaration. Train/auxiliary
            metrics are diagnostics; labeled predict data can also be scored.
            Evaluating a partition does not establish that it is held out.
        batch_size: Positive integer; rows are neither shuffled nor dropped.

    Returns:
        Source metadata, total sample count, and classification metrics using
        the existing missing-class policy. Per-class entries include declared
        class names. Loss is unweighted cross-entropy summed over samples and
        divided by the sample count, rather than averaging batch averages.

    Model execution shares device/logit checks with predict.py and uses logits
    argmax for identical class decisions. Loss uses float64 logits to avoid
    float32 overflow during accumulation. Every module's previous mode is
    restored even on failure. No optimizer is used and gradients are disabled.
    The loader's private generator does not consume the global PyTorch RNG.

    Raises:
        TypeError: Inputs are not the supported artifact/data/contract types.
        ValueError: Roles, purpose flags, contracts, labels, batches, features,
            devices, logits, or loss are invalid.
        RuntimeError: Model execution fails, for example on incompatible dtype.

    File loading and logging remain the runner's responsibility. Test-set
    metrics must not be used to choose a model or aggregation settings.
    """
    if not isinstance(artifact, ModelArtifact) or not isinstance(artifact.model, nn.Module):
        raise TypeError('artifact must be a ModelArtifact containing a torch.nn.Module')
    if not isinstance(artifact.contract, ModelDataContract):
        raise TypeError('artifact.contract must be a ModelDataContract')
    if not isinstance(data, PreparedData) or not isinstance(data.dataset, TensorDataset):
        raise TypeError('data must be PreparedData containing a TensorDataset')
    if not isinstance(data.manifest, PreparedDataManifest):
        raise TypeError('data.manifest must be a PreparedDataManifest')
    if expected_role not in ('train', 'validation', 'auxiliary', 'test', 'predict'):
        raise ValueError('expected_role must be a supported data role')
    if data.manifest.role != expected_role:
        raise ValueError(f'Expected role {expected_role!r}, but manifest declares {data.manifest.role!r}')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    flags = (artifact.training_completed, artifact.testing_only, data.manifest.testing_only)
    if any(type(value) is not bool for value in flags):
        raise ValueError('Model and data purpose flags must be explicit booleans')
    if not artifact.training_completed and not artifact.testing_only:
        raise ValueError('An untrained model artifact must be marked testing_only')

    contract = artifact.contract
    contract.assert_compatible(data.manifest.contract)
    if len(data.dataset.tensors) != 2 or not len(data.dataset):
        raise ValueError('Evaluation requires nonempty prepared features and labels')
    labels = data.dataset.tensors[1]
    if labels.ndim != 1 or labels.dtype != torch.int64:
        raise ValueError('Prepared labels must be a one-dimensional int64 tensor')
    if ((labels < 0) | (labels >= contract.num_classes)).any():
        raise ValueError('Prepared labels contain IDs outside the configured class range')

    model = artifact.model
    device = _model_device(model)
    loader = DataLoader(
        data.dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(0),
    )
    previous_modes = [(module, module.training) for module in model.modules()]
    truth, predictions, total_loss, count = [], [], 0.0, 0
    try:
        model.eval()
        with torch.inference_mode():
            for features, batch_labels in loader:
                if (features.ndim != 2 or features.shape[1] != contract.input_dim
                        or not features.is_floating_point() or not torch.isfinite(features).all()):
                    raise ValueError('Prepared features must be finite [batch, input_dim] floats')
                logits = model(features.to(device))
                _check_logits(logits, len(features), contract.num_classes)
                batch_loss = torch.nn.functional.cross_entropy(
                    logits.to(torch.float64), batch_labels.to(device), reduction='sum',
                ).item()
                if not math.isfinite(batch_loss):
                    raise ValueError('Model produced nonfinite cross-entropy loss')
                total_loss += batch_loss
                count += len(batch_labels)
                truth.extend(batch_labels.cpu().tolist())
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
    finally:
        # Restoring only the root flag would lose deliberately mixed modes.
        for module, was_training in previous_modes:
            module.training = was_training

    if not math.isfinite(total_loss):
        raise ValueError('Accumulated cross-entropy loss is nonfinite')
    metrics = classification_metrics(truth, predictions, contract.num_classes)
    metrics['loss'] = total_loss / count
    for entry in metrics['per_class']:
        entry['class_name'] = contract.class_names[entry['class_id']]

    return PreparedEvaluationResult(
        manifest=data.manifest,
        model_name=artifact.model_name,
        artifact_path=artifact.artifact_path,
        artifact_sha256=artifact.artifact_sha256,
        training_completed=artifact.training_completed,
        testing_only=artifact.testing_only or data.manifest.testing_only,
        sample_count=count,
        metrics=metrics,
    )
