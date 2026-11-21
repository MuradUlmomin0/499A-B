"""True-class-conditioned probability matrices (FedMADE Eq. 1)."""
import numpy as np
import torch


def class_probability_matrix(model, loader, num_classes, device='cpu'):
    """Return float64 [true class, predicted class] mean probabilities.

    Every true class must have auxiliary Support. Actual class counts are used,
    so unequal support does not turn this into a frequency-weighted average.
    No model parameter, gradient, optimizer or module mode is left changed.
    """
    if type(num_classes) is not int or num_classes<2:
        raise ValueError('num_classes must be an integer >=2')
    sums=torch.zeros((num_classes,num_classes),dtype=torch.float64)
    counts=torch.zeros(num_classes,dtype=torch.int64)
    modes=[(module,module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            for features,labels in loader:
                if (features.ndim!=2 or not features.is_floating_point() or not torch.isfinite(features).all()
                        or labels.ndim!=1 or labels.dtype!=torch.int64 or len(labels)!=len(features)
                        or ((labels<0)|(labels>=num_classes)).any()):
                    raise ValueError('Invalid auxiliary batch')
                logits=model(features.to(device))
                if logits.shape!=(len(labels),num_classes) or not torch.isfinite(logits).all():
                    raise ValueError('Invalid/nonfinite local-model logits')
                probabilities=torch.softmax(logits.to(torch.float64),dim=1).cpu()
                sums.index_add_(0,labels.cpu(),probabilities)
                counts+=torch.bincount(labels.cpu(),minlength=num_classes)
    finally:
        for module,previous in modes:module.training=previous
    if (counts==0).any():raise ValueError('Every true auxiliary class needs samples')
    matrix=(sums/counts[:,None]).numpy().copy()
    if not np.isfinite(matrix).all() or not np.allclose(matrix.sum(axis=1),1,atol=1e-12,rtol=0):
        raise ValueError('Invalid class probability matrix')
    return matrix
