from dataclasses import dataclass
from typing import Mapping
import torch


@dataclass
class ClientResult:
    client_id: str
    round_id: int
    sample_count: int
    state: Mapping[str, torch.Tensor]
    update: Mapping[str, torch.Tensor]
    metrics: dict
