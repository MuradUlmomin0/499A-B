"""Load/save model weights with their own compatibility and purpose declarations.

Version 1 is a torch.save() mapping with these exact fields:
    schema_version, model_name, model_state, contract, training_completed,
    testing_only.

``contract`` is a ModelDataContract.to_dict() description recorded by the model
producer, not copied from whichever evaluation dataset happens to be supplied.
All metadata uses ordinary dictionaries, lists, strings, integers, and booleans;
the model state contains tensors. Full pickled model objects are not supported.
"""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path

import torch
from torch import nn

from support.data.contract import ModelDataContract
from support.models.factory import create_model
from support.utils.parameters import clone_state, validate_state


@dataclass(frozen=True)
class ModelArtifact:
    """A loaded CPU model and the declarations belonging to its saved weights.

    The model starts in evaluation mode. Callers may select a device or use its
    weights to initialize an explicitly requested future training run. Optimizer,
    RNG, and client-history state are not restored: this is not exact resume.

    Purpose flags describe producer claims, not proof of training, preprocessing
    correctness, or research validity. The frozen wrapper does not prevent later
    changes to the model object itself.
    """

    model: nn.Module
    contract: ModelDataContract
    artifact_path: Path
    artifact_sha256: str
    model_name: str
    training_completed: bool
    testing_only: bool


def load_model_artifact(path: str | Path) -> ModelArtifact:
    """Construct an MLP from its own saved contract and load validated weights.

    Reads one serialized byte snapshot so the returned SHA-256 identifies the
    same content used to obtain both metadata and weights. Loading uses the
    restricted weights-only reader, with tensors mapped to CPU. Model creation
    preserves the caller's global PyTorch RNG state.

    An explicitly untrained artifact must be marked testing-only. A future run
    must retain testing-only status if either the model or its data is testing-only.
    A declared completed training run does not establish scientific validity.

    Existing raw state dictionaries and Milestone A inspection checkpoints do
    not include this complete metadata. load_model_checkpoint() remains their
    lower-level API when an independently configured model is supplied. Both
    APIs use the same parameter validation utilities and model_state convention.

    Raises:
        FileNotFoundError: The artifact is missing or is a directory.
        ValueError: Schema, purpose declarations, contract, or weights are invalid.
        OSError or pickle.UnpicklingError or RuntimeError: The artifact cannot
            be read with PyTorch's restricted weights-only loader.

    This function does not train, read a dataset, run predictions, or write files.
    """
    artifact_path = Path(path).resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f'Model artifact not found: {artifact_path}')
    content = artifact_path.read_bytes()
    artifact_sha256 = hashlib.sha256(content).hexdigest()
    description = torch.load(io.BytesIO(content), map_location='cpu', weights_only=True)

    model, contract = _validated_model(description)
    return ModelArtifact(
        model=model, contract=contract, artifact_path=artifact_path,
        artifact_sha256=artifact_sha256, model_name=description['model_name'],
        training_completed=description['training_completed'], testing_only=description['testing_only'],
    )


def _validated_model(description):
    """Share the version 1 validation rules between loading and saving."""

    required = {
        'schema_version', 'model_name', 'model_state', 'contract',
        'training_completed', 'testing_only',
    }
    if not isinstance(description, Mapping) or set(description) != required:
        raise ValueError('Model artifact fields must exactly match the version 1 schema')
    version = description['schema_version']
    if type(version) is not int or version != 1:
        raise ValueError('Only model artifact schema version 1 is supported')
    model_name = description['model_name']
    if (not isinstance(model_name, str) or not model_name.strip()
            or model_name != model_name.strip()):
        raise ValueError('model_name must be a nonblank string without surrounding whitespace')
    for name in ('training_completed', 'testing_only'):
        if type(description[name]) is not bool:
            raise ValueError(f'{name} must be an explicit boolean')
    if not description['training_completed'] and not description['testing_only']:
        raise ValueError('An untrained model artifact must be marked testing_only')

    contract = ModelDataContract.from_dict(description['contract'])
    state = description['model_state']
    if not isinstance(state, Mapping) or not state:
        raise ValueError('model_state must be a nonempty tensor mapping')

    # The factory initializes temporary CPU parameters before saved weights are
    # loaded. Those unused random values must not advance the experiment's RNG.
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        model = create_model(contract.model_config())
        reference = model.state_dict()
        if set(state) != set(reference):
            raise ValueError('Artifact parameter names do not match its declared architecture')
        ordered_state = OrderedDict((name, state[name]) for name in reference)
        validate_state(ordered_state, reference)
        model.load_state_dict(clone_state(ordered_state), strict=True)
        model.eval()

    return model, contract


def save_model_artifact(
    model: nn.Module, path: str | Path, *, contract: ModelDataContract,
    model_name: str, training_completed: bool, testing_only: bool,
) -> Path:
    """Save a complete compatible MLP artifact without training or overwriting.

    The producer supplies the model's original contract and explicit purpose
    declarations. These declarations do not prove provenance or model quality.
    All validation and serialization precede filesystem writes. The caller's
    weights, modes, device, gradients and CPU RNG are preserved. Missing parent
    directories are created after validation. A newly created partial file is
    removed if writing fails; existing files are never replaced.
    """
    if not isinstance(model, nn.Module):
        raise TypeError('model must be a torch.nn.Module')
    if not isinstance(contract, ModelDataContract):
        raise TypeError('contract must be a ModelDataContract')
    state = model.state_dict()
    validate_state(state)
    description = {
        'schema_version': 1, 'model_name': model_name,
        'model_state': clone_state(state), 'contract': contract.to_dict(),
        'training_completed': training_completed, 'testing_only': testing_only,
    }
    _validated_model(description)
    buffer = io.BytesIO()
    torch.save(description, buffer)
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an existing artifact, including a collision.
    stream = destination.open('xb')
    try:
        with stream:
            stream.write(buffer.getvalue())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination
