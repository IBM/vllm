from typing import List, Union

import torch
from transformers.generation.logits_process import TypicalLogitsWarper


class MinTokensLogitsProcessor:

    def __init__(self, min_tokens: int, eos_token_id: Union[int, List[int]]):
        self.min_tokens = min_tokens
        self.eos_token_ids = torch.tensor(eos_token_id)

    def __call__(self, token_ids: List[int],
                 logits: torch.tensor) -> torch.tensor:
        # token_ids is only output tokens
        if len(token_ids) < self.min_tokens:
            logits[self.eos_token_ids] = -float("inf")
        return logits


class TypicalLogitsWarperWrapper:

    def __init__(self, mass: float):
        self.warper = TypicalLogitsWarper(mass=mass)

    def __call__(self, token_ids: List[int],
                 logits: torch.tensor) -> torch.tensor:
        # transformers warpers assume tensors of shape (batch_size, vocab_size)
        # and the typical warper doesn't use input_ids
        return self.warper(input_ids=None, scores=logits.reshape((1, -1)))
