"""The single implemented aggregation strategy is sample-weighted FedAvg."""
from collections import OrderedDict
from typing import Protocol
import torch
from support.utils.parameters import validate_state


class Aggregator(Protocol):
    def aggregate(self, results): ...


class FedAvgAggregator:
    def aggregate(self, results):
        if not results:
            raise ValueError('Cannot aggregate no clients')
        if len({r.client_id for r in results}) != len(results):
            raise ValueError('Duplicate client contributions')
        if len({r.round_id for r in results}) != 1:
            raise ValueError('Mixed communication rounds')
        if any(type(r.sample_count) is not int or r.sample_count <= 0 for r in results):
            raise ValueError('Training sample counts must be positive integers')
        reference = results[0].state
        for result in results:
            validate_state(result.state, reference)
        total = sum(r.sample_count for r in results)
        weights = {r.client_id: r.sample_count / total for r in results}
        state = OrderedDict()
        # Accumulate in float64, then restore original tensor dtype and shape.
        for key, value in reference.items():
            accumulator = torch.zeros_like(value, dtype=torch.float64, device='cpu')
            for result in results:
                accumulator.add_(result.state[key].detach().cpu().to(torch.float64), alpha=weights[result.client_id])
            state[key] = accumulator.to(value.dtype)
        validate_state(state, reference)
        return state, weights
