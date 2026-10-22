"""Batched predictions from compatible PyTorch models without training."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from support.data.contract import ModelDataContract
from support.data.manifest import PreparedDataManifest
from support.data.prepared import PreparedData


@dataclass(frozen=True)
class PredictionResult:
    """CPU outputs in original row order, with their source declarations.

    ``probabilities[:, i]`` corresponds to ``manifest.contract.class_names[i]``.
    Values are softmax outputs, not calibrated estimates of correctness.
    ``manifest.testing_only`` retains the data's declared purpose; this helper
    cannot establish that a checkpoint was trained or that data is research-ready.
    The frozen wrapper does not make its output tensors immutable.
    """

    manifest: PreparedDataManifest
    class_ids: torch.Tensor
    probabilities: torch.Tensor

    @property
    def predicted_labels(self) -> tuple[str, ...]:
        """Translate each numeric prediction using the declared class order."""
        names = self.manifest.contract.class_names
        return tuple(names[index] for index in self.class_ids.tolist())


def _model_device(model):
    devices = {value.device for value in (*model.parameters(), *model.buffers())}
    if len(devices) > 1:
        raise ValueError('Prediction requires model parameters and buffers on one device')
    device = next(iter(devices), torch.device('cpu'))
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Prediction currently supports CPU or CUDA models')
    return device


def _check_logits(logits, rows, num_classes):
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise ValueError('Model must return a floating-point logits tensor')
    if logits.ndim != 2 or logits.shape != (rows, num_classes):
        raise ValueError('Model logits must have shape [batch, num_classes]')
    if not torch.isfinite(logits).all():
        raise ValueError('Model returned nonfinite logits')


def predict_prepared_data(
    model: nn.Module,
    data: PreparedData,
    *,
    model_contract: ModelDataContract,
    batch_size: int = 256,
) -> PredictionResult:
    """Generate class predictions from already-loaded, compatible prepared data.

    The caller supplies a model matching its own independently recorded
    contract, for example one constructed by create_model() and populated by
    load_model_checkpoint(). This function compares that contract with the data
    contract; it does not infer feature/class meanings from model tensors.

    Labeled and unlabeled prepared datasets are supported. Target labels are
    ignored: they cannot influence predictions, and no metrics are calculated.
    Roles are preserved in the returned manifest, including training-set use
    when deliberately requested for diagnostics by a later runner.

    The model stays on its current CPU/CUDA device. Features must be compatible
    with its input dtype (the current factory creates float32 MLPs). Prediction
    disables gradients, uses evaluation mode, and restores every module's prior
    training/evaluation flag even if a forward pass fails. No optimizer is used.

    Raises:
        TypeError: The model, data, or model contract has an unsupported type.
        ValueError: Batch size, contracts, features, devices, or logits are invalid.
        RuntimeError: Model execution fails, such as an incompatible input dtype.

    File loading, preprocessing, checkpoint validation, and experiment logging
    remain the caller's responsibilities. This helper never writes artifacts.
    """
    if not isinstance(model, nn.Module):
        raise TypeError('model must be a torch.nn.Module')
    if not isinstance(data, PreparedData) or not isinstance(data.dataset, TensorDataset):
        raise TypeError('data must be PreparedData containing a TensorDataset')
    if not isinstance(model_contract, ModelDataContract):
        raise TypeError('model_contract must be a ModelDataContract')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if not len(data.dataset) or len(data.dataset.tensors) not in (1, 2):
        raise ValueError('Prepared dataset must contain nonempty features and optional labels')
    model_contract.assert_compatible(data.manifest.contract)
    device = _model_device(model)

    # A private loader generator avoids consuming the experiment's global torch
    # RNG merely to create the iterator. Rows are neither shuffled nor dropped.
    loader = DataLoader(
        data.dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(0),
    )
    previous_modes = [(module, module.training) for module in model.modules()]
    ids, scores = [], []
    try:
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                features = batch[0]
                if (features.ndim != 2 or features.shape[1] != model_contract.input_dim
                        or not features.is_floating_point() or not torch.isfinite(features).all()):
                    raise ValueError('Prepared features must be finite [batch, input_dim] floats')
                logits = model(features.to(device))
                _check_logits(logits, len(features), model_contract.num_classes)
                probabilities = torch.softmax(logits, dim=1)
                if not torch.isfinite(probabilities).all():
                    raise ValueError('Model produced nonfinite class probabilities')
                # Select from logits, so softmax rounding cannot introduce a tie
                # that changes the model's original highest-scoring class.
                ids.append(logits.argmax(dim=1).detach().cpu().clone())
                scores.append(probabilities.detach().cpu().clone())
    finally:
        # Restoring only the root flag would lose intentionally mixed modes,
        # such as an evaluated submodule inside a model marked for training.
        for module, was_training in previous_modes:
            module.training = was_training

    return PredictionResult(
        manifest=data.manifest,
        class_ids=torch.cat(ids),
        probabilities=torch.cat(scores),
    )
