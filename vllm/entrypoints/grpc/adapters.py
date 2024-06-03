"""Contains code to map api requests for adapters (e.g. peft prefixes, LoRA)
into valid LLM engine requests"""
import asyncio
import concurrent.futures
import dataclasses
import json
import os
from typing import Dict, Optional, Union

from vllm.entrypoints.grpc.pb.generation_pb2 import (BatchedGenerationRequest,
                                                     SingleGenerationRequest)
from vllm.entrypoints.grpc.validation import TGISValidationError
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest

global_thread_pool = None  # used for loading adapter files from disk


@dataclasses.dataclass
class AdapterMetadata:
    unique_id: int  # Unique integer for vllm to identify the adapter
    adapter_type: str  # The string name of the peft adapter type, e.g. LORA
    full_path: str
    full_config: Dict  # The loaded adapter_config.json dict


@dataclasses.dataclass
class AdapterStore:
    cache_path: str  # Path to local store of adapters to load from
    adapters: Dict[str, AdapterMetadata]
    next_unique_id: int = 1


async def validate_adapters(
    request: Union[SingleGenerationRequest, BatchedGenerationRequest],
    adapter_store: Optional[AdapterStore]
) -> Dict[str, Union[LoRARequest, PromptAdapterRequest]]:
    """Takes the adapter name from the request and constructs a valid
        engine request if one is set. Raises if the requested adapter
        does not exist or adapter type is unsupported

        Returns the kwarg dictionary to add to an engine.generate() call.
        """
    global global_thread_pool
    adapter_id = request.adapter_id
    # Backwards compatibility for `prefix_id` arg
    if not adapter_id and request.prefix_id:
        adapter_id = request.prefix_id

    if adapter_id and not adapter_store:
        TGISValidationError.AdaptersDisabled.error()

    if not adapter_id or not adapter_store:
        return {}

    # If not already cached, we need to validate that files exist and
    # grab the type out of the adapter_config.json file
    if (adapter_metadata := adapter_store.adapters.get(adapter_id)) is None:
        local_adapter_path = os.path.join(adapter_store.cache_path, adapter_id)

        loop = asyncio.get_running_loop()
        if global_thread_pool is None:
            global_thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=2)

        adapter_config = await (loop.run_in_executor(
            global_thread_pool, _load_adapter_config_from_file, adapter_id,
            local_adapter_path))
        adapter_type = adapter_config.get("peft_type", None)

        # Add to cache
        adapter_metadata = AdapterMetadata(
            unique_id=adapter_store.next_unique_id,
            adapter_type=adapter_type,
            full_path=local_adapter_path,
            full_config=adapter_config)
        adapter_store.adapters[adapter_id] = adapter_metadata

    # Build the proper vllm request object
    if adapter_metadata.adapter_type == "LORA":
        lora_request = LoRARequest(lora_name=adapter_id,
                                   lora_int_id=adapter_metadata.unique_id,
                                   lora_local_path=adapter_metadata.full_path)
        return {"lora_request": lora_request}
    elif adapter_metadata.adapter_type == "PROMPT_TUNING":
        prompt_adapter_request = PromptAdapterRequest(
            prompt_adapter_id=adapter_metadata.unique_id,
            prompt_adapter_name=adapter_id,
            prompt_adapter_local_path=adapter_metadata.full_path,
            prompt_adapter_num_virtual_tokens=adapter_metadata.full_config.get(
                "num_virtual_tokens", 0))
        return {"prompt_adapter_request": prompt_adapter_request}

    # All other types unsupported
    TGISValidationError.AdapterUnsupported.error(adapter_metadata.adapter_type)


def _load_adapter_config_from_file(adapter_id: str, adapter_path: str) -> Dict:
    """This function does all the filesystem access required to deduce the type
     of the adapter. It's run in a separate thread pool executor so that file
     access does not block the main event loop."""
    if not os.path.exists(adapter_path):
        TGISValidationError.AdapterNotFound.error(adapter_id,
                                                  "directory does not exist")

    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        TGISValidationError.AdapterNotFound.error(
            adapter_id, "invalid adapter: no adapter_config.json found")

    # NB: blocks event loop
    with open(adapter_config_path) as adapter_config_file:
        adapter_config = json.load(adapter_config_file)

    return adapter_config
