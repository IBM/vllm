from enum import IntEnum

import torch
import torch.nn as nn

from vllm.model_executor.pooling_metadata import (PoolingMetadata,
                                                  PoolingTensors)
from vllm.sequence import EmbeddingSequenceGroupOutput, PoolerOutput


class PoolingType(IntEnum):
    """Enumeration for different types of pooling methods."""
    LAST = 0
    MEAN = 1


class Pooler(nn.Module):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.

    Attributes:
        pooling_type: The type of pooling to use (LAST, AVERAGE, MAX).
        normalize: Whether to normalize the pooled data.
    """

    def __init__(self, pooling_type: PoolingType, normalize: bool):
        super().__init__()
        self.pooling_type = pooling_type
        self.normalize = normalize

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        """Pools specific information from hidden states based on metadata."""
        prompt_lens = PoolingTensors.from_pooling_metadata(
            pooling_metadata, hidden_states.device).prompt_lens

        if self.pooling_type == PoolingType.LAST:
            last_token_flat_indices = torch.cumsum(prompt_lens, dim=0) - 1
            pooled_data = hidden_states[last_token_flat_indices]
        elif self.pooling_type == PoolingType.MEAN:
            # Calculate mean pooling
            cumsum = torch.cumsum(hidden_states, dim=0)
            start_indices = torch.cat([
                torch.tensor([0], device=hidden_states.device),
                torch.cumsum(prompt_lens[:-1], dim=0)
            ])
            end_indices = torch.cumsum(prompt_lens, dim=0)
            pooled_data = (
                cumsum[end_indices - 1] - cumsum[start_indices] +
                hidden_states[start_indices]) / prompt_lens.unsqueeze(1)
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling_type}")

        if self.normalize:
            pooled_data = nn.functional.normalize(pooled_data, p=2, dim=1)

        pooled_outputs = [
            EmbeddingSequenceGroupOutput(data.tolist()) for data in pooled_data
        ]

        return PoolerOutput(outputs=pooled_outputs)
