from typing import List, Tuple

import torch
from transformers.generation.logits_process import TypicalLogitsWarper


class TypicalLogitsWarperWrapper:

    def __init__(self, mass: float):
        self.warper = TypicalLogitsWarper(mass=mass)

    def __call__(self, token_ids: List[int],
                 logits: torch.Tensor) -> torch.Tensor:
        # transformers warpers assume tensors of shape (batch_size, vocab_size)
        # and the typical warper doesn't use input_ids
        return self.warper(input_ids=None,
                           scores=logits.reshape(1, -1)).flatten()


class ExpDecayLengthPenaltyWarper:

    def __init__(self, length_penalty: Tuple[int, float], eos_token_id: int):
        self.start, self.penalty = length_penalty
        self.eos_token_id = eos_token_id

    def __call__(self, token_ids: List[int],
                 logits: torch.Tensor) -> torch.Tensor:
        tokens_past = len(token_ids) - self.start
        if tokens_past > 0:
            eos_logit = logits[self.eos_token_id]
            # To support negative logits we compute the penalty of the
            # absolute value and add to the original logit
            logits[self.eos_token_id] = eos_logit + torch.abs(eos_logit) * (
                pow(self.penalty, tokens_past) - 1)
        return logits
