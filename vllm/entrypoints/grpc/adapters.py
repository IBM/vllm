"""Contains code to map api requests for adapters (e.g. peft prefixes, LoRA)
into valid LLM engine requests"""
import dataclasses
import json
import os
from typing import Dict, Optional, Union

from vllm.entrypoints.grpc.pb.generation_pb2 import (BatchedGenerationRequest,
                                                     SingleGenerationRequest)
from vllm.entrypoints.grpc.validation import TGISValidationError
from vllm.lora.request import LoRARequest


@dataclasses.dataclass
class AdapterMetadata:
    unique_id: int  # Unique integer for vllm to identify the adapter
    adapter_type: str  # The string name of the peft adapter type, e.g. LORA
    full_path: str


@dataclasses.dataclass
class AdapterStore:
    cache_path: str  # Path to local store of adapters to load from
    adapters: Dict[str, AdapterMetadata]
    next_unique_id: int = 1


def validate_adapters(
        request: Union[SingleGenerationRequest, BatchedGenerationRequest],
        adapter_store: Optional[AdapterStore]) -> Dict[str, LoRARequest]:
    """Takes the adapter name from the request and constructs a valid
        engine request if one is set. Raises if the requested adapter
        does not exist or adapter type is unsupported

        Returns the kwarg dictionary to add to an engine.generate() call.
        """
    adapter_id = request.adapter_id

    if adapter_id and not adapter_store:
        TGISValidationError.AdaptersDisabled.error()

    if not adapter_id or not adapter_store:
        return {}

    # If not already cached, we need to validate that files exist and
    # grab the type out of the adapter_config.json file
    if (adapter_metadata := adapter_store.adapters.get(adapter_id)) is None:
        local_adapter_path = os.path.join(adapter_store.cache_path, adapter_id)

        if not os.path.exists(local_adapter_path):
            TGISValidationError.AdapterNotFound.error(
                adapter_id, "directory does not exist")

        adapter_config_path = os.path.join(local_adapter_path,
                                           "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            TGISValidationError.AdapterNotFound.error(
                adapter_id, "invalid adapter: no adapter_config.json found")

        # NB: blocks event loop
        with open(adapter_config_path) as adapter_config_file:
            adapter_config = json.load(adapter_config_file)

        adapter_type = adapter_config.get("peft_type", None)

        # Add to cache
        adapter_metadata = AdapterMetadata(
            unique_id=adapter_store.next_unique_id,
            adapter_type=adapter_type,
            full_path=local_adapter_path)
        adapter_store.adapters[adapter_id] = adapter_metadata

    # Build the proper vllm request object
    if adapter_metadata.adapter_type == "LORA":
        lora_request = LoRARequest(lora_name=adapter_id,
                                   lora_int_id=adapter_metadata.unique_id,
                                   lora_local_path=adapter_metadata.full_path)
        return {"lora_request": lora_request}

    # All other types unsupported
    TGISValidationError.AdapterUnsupported.error(adapter_metadata.adapter_type)
