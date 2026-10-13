"""Tiny synthetic fixtures ONLY for framework tests; never thesis evidence."""
import numpy as np
from .datasets import tabular_dataset


def make_dummy_data(input_dim, num_classes, client_samples, evaluation_samples, seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(num_classes, input_dim)).astype('float32')
    def draw(count):
        labels = rng.integers(0, num_classes, size=count)
        features = centers[labels] + rng.normal(scale=0.5, size=(count,input_dim)).astype('float32')
        return tabular_dataset(features, labels, input_dim, num_classes)
    # Each split is sampled independently; no dataset object or record is reused.
    return [draw(n) for n in client_samples], draw(evaluation_samples)
