"""Contains code to map api requests for adapters (e.g. peft prefixes, LoRA)
into valid LLM engine requests"""
import dataclasses
import os
from typing import Dict, Optional, Tuple, Union

from vllm.entrypoints.grpc.pb.generation_pb2 import (BatchedGenerationRequest,
                                                     SingleGenerationRequest)
from vllm.entrypoints.grpc.validation import TGISValidationError
from vllm.lora.request import LoRARequest


@dataclasses.dataclass
class AdapterStore:
    cache_path: str  # Path to local store of adapters to load from
    unique_id_map: Dict[str, int]  # maps adapter names to unique integer IDs
    next_unique_id: int = 1


def validate_adapters(
    request: Union[SingleGenerationRequest, BatchedGenerationRequest],
    lora_adapter_store: Optional[AdapterStore]
) -> Tuple[Optional[LoRARequest], None]:
    """Takes the adapter names from the request and constructs a valid
        engine request if one is set. Raises if the requested adapter
        does not exist"""
    lora_id = request.lora_id
    if lora_id:
        if not lora_adapter_store:
            # using raise/format instead of .error so mypy knows this raises
            raise ValueError(TGISValidationError.LoraDisabled.value.format())

        local_lora_path = os.path.join(lora_adapter_store.cache_path, lora_id)

        # Do a bit of up-front validation so that we don't ask the engine
        # to try to load an invalid adapter
        if not os.path.exists(local_lora_path):
            TGISValidationError.LoraAdapterNotFound.error(
                lora_id, "directory does not exist")
        if not os.path.exists(
                os.path.join(local_lora_path, "adapter_config.json")):
            TGISValidationError.LoraAdapterNotFound.error(
                lora_id, "invalid adapter: no adapter_config.json found")

        # We need to track a unique integer for vLLM to identify the lora
        # adapters
        if lora_id not in lora_adapter_store.unique_id_map:
            lora_adapter_store.unique_id_map[
                lora_id] = lora_adapter_store.next_unique_id
            lora_adapter_store.next_unique_id += 1
        unique_id = lora_adapter_store.unique_id_map[lora_id]
        lora_request = LoRARequest(lora_name=lora_id,
                                   lora_int_id=unique_id,
                                   lora_local_path=local_lora_path)
    else:
        lora_request = None

    if request.prefix_id:
        # TODO: hook up PromptAdapterRequest once implemented in the engine
        raise ValueError("prefix_id not implemented yet")

    # Second return slot left here for the incoming PromptAdapterRequest
    # See https://github.com/vllm-project/vllm/pull/4645/files
    return lora_request, None
