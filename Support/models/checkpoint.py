"""Load compatible saved model weights without performing training.

Accepted files contain either a raw PyTorch state dictionary or the
``model_state`` entry used by PRIVATE SHIELD's existing checkpoints.
Architecture, feature order, class mapping, and preprocessing must be supplied
and checked by the caller; tensor compatibility alone cannot establish them.
"""

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from support.utils.parameters import clone_state, validate_state


def load_model_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> nn.Module:
    """Load validated weights into an already constructed, compatible model.

    Args:
        model: Model with the intended architecture, dtype, and device. As in
            Milestone A, supported states contain floating-point tensors only.
        checkpoint_path: Local file saved using ``torch.save(state_dict, path)``
            or ``torch.save({'model_state': state_dict, ...}, path)``.

    Returns:
        The same model object, with its saved weights loaded. Its device and
        training/evaluation mode are unchanged; the caller chooses its next use.

    Raises:
        TypeError: The model is not a PyTorch module.
        FileNotFoundError: The checkpoint path is not a file.
        ValueError: The saved state has incompatible keys, shapes, dtypes, or
            unsupported/nonfinite values. These checks precede model mutation.
        OSError or pickle.UnpicklingError or RuntimeError: PyTorch cannot read
            the file with its restricted weights-only loader.

    This restores weights only, not optimizer state, update history, or RNG
    state. It does not provide exact experiment resume or run evaluation.
    """
    if not isinstance(model, nn.Module):
        raise TypeError('model must be a torch.nn.Module')

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint file not found: {path}')

    # Read onto CPU even when the original checkpoint was saved on a GPU.
    # Full pickled model objects are deliberately outside this weights API.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Checkpoint must contain a state dictionary or model_state')
    state = checkpoint.get('model_state', checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise ValueError('Checkpoint model state must be a nonempty mapping')

    reference = model.state_dict()
    validate_state(reference)
    if set(state) != set(reference):
        raise ValueError('Checkpoint parameter names do not match the model')

    # Names determine parameter identity; dictionary insertion order need not
    # match. Normalize order before reusing the common state validation utility.
    ordered_state = OrderedDict((key, state[key]) for key in reference)
    validate_state(ordered_state, reference)
    model.load_state_dict(clone_state(ordered_state), strict=True)
    return model
