import torch
from vllm.adapter_commons.models import AdapterModel
from peft.utils import load_peft_weights
import os

class PromptAdapterModel(AdapterModel):

    def __init__(self,
                 prompt_adapter_id=None,
                 num_virtual_tokens=None,
                 prompt_embedding=None) -> None:
        self.id = prompt_adapter_id
        self.kv_cache = None
        self.prompt_embedding = prompt_embedding
        self.num_virtual_tokens = num_virtual_tokens
        
    @classmethod
    def from_local_checkpoint(cls,
                              adapter_path,
                              prompt_adapter_id,
                              torch_device='cuda') -> "PromptAdapterModel":
        try:
            adapters_weights = load_peft_weights(adapter_path,
                                                torch_device)
            prompt_embedding = adapters_weights["prompt_embeddings"].half()
        except Exception as e: 
            # if no PEFT adapter found, load caikit-style adapter from path
            prompt_embedding = torch.load(adapter_path+'/decoder.pt', 
                                          weights_only=True, 
                                          map_location=torch_device).half()
        num_virtual_tokens = prompt_embedding.shape[0]
        return cls(prompt_adapter_id, num_virtual_tokens, prompt_embedding)