"""Bits that get monkey patched in"""
import os

import torch
from peft import load_peft_weights

from vllm.prompt_adapter.models import PromptAdapterModel


@classmethod
def from_local_checkpoint(cls,
                          adapter_path,
                          prompt_adapter_id,
                          torch_device='cuda') -> "PromptAdapterModel":
    """Patch for PromptAdapterModel that supports decoder.pt models"""
    peft_config_path = os.path.join(adapter_path, "adapter_config.json")
    decoder_pt_path = os.path.join(adapter_path, "decoder.pt")

    print("\n\n ~~~~ PATCHED LOAD CODE ~~~ \n\n")

    if os.path.exists(peft_config_path):
        adapters_weights = load_peft_weights(adapter_path, torch_device)
        prompt_embedding = adapters_weights["prompt_embeddings"].half()
    elif os.path.exists(decoder_pt_path):
        # if no PEFT adapter found, load caikit-style adapter from path

        print("\n\n ~~~~ DECODER.PT LOADING ~~~ \n\n")

        prompt_embedding = torch.load(decoder_pt_path,
                                      weights_only=True,
                                      map_location=torch_device).half()
    else:
        raise ValueError(
            f"No supported adapter format found at path {adapter_path}")
    num_virtual_tokens = prompt_embedding.shape[0]
    return cls(prompt_adapter_id, num_virtual_tokens, prompt_embedding)


def monkey_patch_prompt_adapter():
    """Insert our own implementation to support decoder.pt prompts"""
    print("\n\n MONKEY PATCH!!!!! \n\n")
    PromptAdapterModel.from_local_checkpoint = from_local_checkpoint
