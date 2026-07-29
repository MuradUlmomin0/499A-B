from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as functional


ModelState = Dict[str, torch.Tensor]


def min_max_normalize(values: torch.Tensor) -> torch.Tensor:
    """Normalize values between 0 and 1."""

    minimum = values.min()
    maximum = values.max()

    difference = maximum - minimum

    if difference.item() < 1e-12:
        return torch.zeros_like(values)

    return (values - minimum) / difference


def detect_free_riders(
    wef_matrices: List[torch.Tensor],
    expected_free_riders: int = 1,
) -> Tuple[List[int], torch.Tensor]:
    """
    Simplified WEF-based client detection.

    It uses:
    - Euclidean distance
    - cosine similarity
    - average WEF frequency

    For this first prototype, we assume that we know
    how many Free-Riders exist.
    """

    if not wef_matrices:
        raise ValueError("No WEF-Matrix was provided.")

    vectors = torch.stack(
        [
            matrix.flatten().to(torch.float32)
            for matrix in wef_matrices
        ]
    )

    # Median gives a reference pattern that is less affected
    # by one unusual client.
    reference = vectors.median(dim=0).values

    distances = torch.linalg.vector_norm(
        vectors - reference,
        dim=1,
    )

    cosine_similarities = functional.cosine_similarity(
        vectors,
        reference.unsqueeze(0),
        dim=1,
        eps=1e-8,
    )

    average_frequencies = vectors.mean(dim=1)
    median_frequency = average_frequencies.median()

    frequency_deviation = torch.abs(
        average_frequencies - median_frequency
    )

    distance_score = min_max_normalize(distances)

    cosine_score = min_max_normalize(
        1.0 - cosine_similarities
    )

    frequency_score = min_max_normalize(
        frequency_deviation
    )

    final_scores = (
        distance_score
        + cosine_score
        + frequency_score
    )

    expected_free_riders = min(
        expected_free_riders,
        len(wef_matrices),
    )

    detected_indices = torch.topk(
        final_scores,
        k=expected_free_riders,
    ).indices.tolist()

    return detected_indices, final_scores


def federated_average(
    model_states: List[ModelState],
) -> OrderedDict[str, torch.Tensor]:
    """
    Average model parameters from selected clients.
    """

    if not model_states:
        raise ValueError(
            "Cannot aggregate an empty model list."
        )

    averaged_state: OrderedDict[str, torch.Tensor] = (
        OrderedDict()
    )

    for key in model_states[0].keys():
        stacked_parameters = torch.stack(
            [
                state[key].to(torch.float32)
                for state in model_states
            ]
        )

        averaged_state[key] = stacked_parameters.mean(
            dim=0
        )

    return averaged_state