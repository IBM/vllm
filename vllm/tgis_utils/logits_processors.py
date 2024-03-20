from typing import List

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
