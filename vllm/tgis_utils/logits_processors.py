from typing import List, Tuple

import torch
from transformers.generation.logits_process import (
    ExponentialDecayLengthPenalty, TypicalLogitsWarper)


class TypicalLogitsWarperWrapper:

    def __init__(self, mass: float):
        self.warper = TypicalLogitsWarper(mass=mass)

    def __call__(self, token_ids: List[int],
                 logits: torch.tensor) -> torch.tensor:
        # transformers warpers assume tensors of shape (batch_size, vocab_size)
        # and the typical warper doesn't use input_ids
        return self.warper(input_ids=None, scores=logits.reshape((1, -1)))


class LengthPenaltyWarper:

    def __init__(self, length_penalty: Tuple[int, float], eos_token_id: int):
        self.length_penalty = length_penalty
        self.eos_token_id = eos_token_id
        self.warper = ExponentialDecayLengthPenalty(
            exponential_decay_length_penalty=length_penalty,
            eos_token_id=eos_token_id,
            input_ids_seq_length=0)

    def __call__(self, token_ids: List[int],
                 logits: torch.tensor) -> torch.tensor:
        return self.warper(input_ids=torch.Tensor(token_ids),
                           scores=logits.reshape((1, -1)))
