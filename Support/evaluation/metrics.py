import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def classification_metrics(y_true, y_pred, num_classes):
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    if true.ndim != 1 or pred.shape != true.shape or not len(true):
        raise ValueError('Expected nonempty matching label vectors')
    if not np.issubdtype(true.dtype,np.integer) or not np.issubdtype(pred.dtype,np.integer):
        raise ValueError('Labels must be integers')
    if ((true < 0) | (true >= num_classes) | (pred < 0) | (pred >= num_classes)).any():
        raise ValueError('Labels outside configured class range')
    labels = np.arange(num_classes)
    precision, recall, f1, support = precision_recall_fscore_support(true,pred,labels=labels,zero_division=0)
    return {'accuracy': float(accuracy_score(true,pred)),
            # Balanced accuracy averages recalls for true classes actually observed.
            'balanced_accuracy': float(recall[support > 0].mean()),
            'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean()),
            'macro_f1': float(f1.mean()), 'weighted_f1': float(np.average(f1,weights=support)),
            'per_class': [{'class_id': int(i), 'precision': float(precision[i]), 'recall': float(recall[i]),
                           'f1': float(f1[i]), 'support': int(support[i])} for i in labels],
            'confusion_matrix': confusion_matrix(true,pred,labels=labels).tolist()}


def evaluate_model(model, loader, device, num_classes):
    previous_mode = model.training
    model.eval()
    truth, predictions, total_loss, count = [], [], 0.0, 0
    try:
        with torch.no_grad():
            for features, labels in loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features)
                total_loss += torch.nn.functional.cross_entropy(logits,labels,reduction='sum').item()
                count += len(labels)
                truth.extend(labels.cpu().tolist())
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
    finally:
        model.train(previous_mode)
    metrics = classification_metrics(truth,predictions,num_classes)
    metrics['loss'] = total_loss / count
    return metrics
