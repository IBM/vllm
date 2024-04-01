from typing import List, Tuple

import torch
from transformers.generation.logits_process import TypicalLogitsWarper


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

    def __call__(self, token_ids: List[int],
                 logits: torch.tensor) -> torch.tensor:
        tokens_past = len(token_ids) - self.length_penalty[0]
        if tokens_past > 0:
            logits[self.eos_token_id] *= pow(self.length_penalty[1],
                                             tokens_past)
        return logits
