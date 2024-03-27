import logging
from typing import (Any, Callable, Dict, Hashable, List, Optional, Type)
from torch import nn

from vllm.utils import LRUCache

from peft import (PeftConfig, PromptLearningConfig, mapping)
from peft.utils import load_peft_weights

logger = logging.getLogger(__name__)

_GLOBAL_PROMPT_ADAPTER_ID = 0


def get_prompt_adapter_id():
    global _GLOBAL_PROMPT_ADAPTER_ID
    _GLOBAL_PROMPT_ADAPTER_ID += 1
    return _GLOBAL_PROMPT_ADAPTER_ID


class PromptAdapterModel(object):

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
                              adapter_model_and_path,
                              prompt_adapter_id,
                              torch_device='cuda') -> "PromptAdapterModel":
        peft_config = mapping.PEFT_TYPE_TO_CONFIG_MAPPING[
            PeftConfig._get_peft_type(
                adapter_model_and_path, )].from_pretrained(
                    adapter_model_and_path, )
        assert isinstance(peft_config, PromptLearningConfig)
        num_virtual_tokens = peft_config.num_virtual_tokens
        adapters_weights = load_peft_weights(adapter_model_and_path,
                                             torch_device)
        prompt_embedding = adapters_weights["prompt_embeddings"].half()
        return cls(prompt_adapter_id, num_virtual_tokens, prompt_embedding)


class PromptAdapterModelManager:
    """A manager that manages multiple Prompt Adapter models."""

    def __init__(self, model: nn.Module):
        """Create a PromptAdapterModel and adapter for a given model.

        Args:
            model: the model to be adapted.
        """
        self.model: nn.Module = model
        self._registered_prompt_adapters: Dict[int, PromptAdapterModel] = {}
        # Dict instead of a Set for compatibility with LRUCache.
        self.prompt_adapter_index_to_id: List[
            Optional[int]] = [None] * self.prompt_adapter_slots
        self._active_prompt_adapters: Dict[int, None] = {}
        self._last_mapping = None
        self.model.prompt_adapter_manager = self

    @property
    def prompt_adapter_slots(self) -> int:
        return 1  # from a prompt_adapter_config?

    @property
    def capacity(self) -> int:
        return 1  # from a prompt_adapter_config?

    def __len__(self) -> int:
        return len(self._registered_prompt_adapters)

    def activate_prompt_adapter(
        self,
        prompt_adapter_id: int,
    ) -> bool:
        """Move PromptAdapter into a GPU buffer to be used in the forward pass."""
        if prompt_adapter_id in self._active_prompt_adapters:
            return False
        first_free_slot = next(
            ((i, prompt_adapter_id) for i, prompt_adapter_id in enumerate(
                self.prompt_adapter_index_to_id) if prompt_adapter_id is None),
            None)
        if first_free_slot is None:
            raise ValueError("No free prompt_adapter slots")
        index, _ = first_free_slot
        self._active_prompt_adapters[prompt_adapter_id] = None
        prompt_adapter_model = self._registered_prompt_adapters[
            prompt_adapter_id]
        logger.debug(
            f"Activating prompt_adapter. int id: {prompt_adapter_model.id}, slot index: {index}"
        )
        self.prompt_adapter_index_to_id[index] = prompt_adapter_model.id
        for module_name, module in self.model.named_modules():
            if 'Model' in (module.__class__.__name__):
                module.prefix_encoder = prompt_adapter_model
                break
        return True

    def _deactivate_prompt_adapter(self, prompt_adapter_id: int):
        try:
            index = self.prompt_adapter_index_to_id.index(prompt_adapter_id)
            self.prompt_adapter_index_to_id[index] = None
        except ValueError:
            pass

    def deactivate_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        """Remove a prompt_adapter from a GPU buffer."""
        if prompt_adapter_id in self._active_prompt_adapters:
            self._deactivate_prompt_adapter(prompt_adapter_id)
            self._active_prompt_adapters.pop(prompt_adapter_id)
            return True
        return False

    def _add_prompt_adapter(self, prompt_adapter: PromptAdapterModel) -> bool:
        self._registered_prompt_adapters[prompt_adapter.id] = prompt_adapter

    def add_prompt_adapter(self, prompt_adapter: PromptAdapterModel) -> bool:
        """Add a PromptAdapterModel to the manager CPU cache."""
        if prompt_adapter.id not in self._registered_prompt_adapters:
            if len(self._registered_prompt_adapters) >= self.capacity:
                raise RuntimeError("No free prompt_adapter slots.")
            self._add_prompt_adapter(prompt_adapter)
            return True
        return False

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        """Remove a PromptAdapterModel from the manager CPU cache."""
        # TODO: should we check active prompt_adapter?
        self.deactivate_prompt_adapter(prompt_adapter_id)
        return bool(
            self._registered_prompt_adapters.pop(prompt_adapter_id, None))

    def list_prompt_adapters(self) -> Dict[int, PromptAdapterModel]:
        """List all registered PromptAdapterModels."""
        return dict(self._registered_prompt_adapters)

    def get_prompt_adapter(
            self, prompt_adapter_id: int) -> Optional[PromptAdapterModel]:
        return self._registered_prompt_adapters.get(prompt_adapter_id, None)

    def remove_all_prompt_adapters(self) -> bool:
        """Remove all PromptAdapterModel from the manager."""
        self._registered_prompt_adapters.clear()
        self.prompt_adapter_index_to_id = [None] * self.prompt_adapter_slots
        self._active_prompt_adapters.clear()


class PromptAdapterLRUCache(LRUCache):

    def __init__(self, capacity: int,
                 deactivate_prompt_adapter_fn: Callable[[Hashable], None]):
        super().__init__(capacity)
        self.deactivate_prompt_adapter_fn = deactivate_prompt_adapter_fn

    def _on_remove(self, key: Hashable, value: Any):
        logger.debug(f"Removing prompt_adapter. int id: {key}")
        self.deactivate_prompt_adapter_fn(key)
        return super()._on_remove(key, value)


class LRUCachePromptAdapterModelManager(PromptAdapterModelManager):
    """A model manager that manages multiple prompt_adapters with LRU cache."""

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self._registered_prompt_adapters: PromptAdapterLRUCache = PromptAdapterLRUCache(
            self.capacity, self.deactivate_prompt_adapter)
        self._active_prompt_adapters: PromptAdapterLRUCache = PromptAdapterLRUCache(
            self.prompt_adapter_slots, self._deactivate_prompt_adapter)

    def list_prompt_adapters(self) -> Dict[int, PromptAdapterModel]:
        """List all registered PromptAdapterModel."""
        return dict(self._registered_prompt_adapters.cache)

    def add_prompt_adapter(self, prompt_adapter: PromptAdapterModel) -> bool:
        """Add a PromptAdapterModel to the manager."""
        if prompt_adapter.id not in self._registered_prompt_adapters:
            self._add_prompt_adapter(prompt_adapter)
            was_added = True
        else:
            # We always touch to update the LRU cache order
            self._registered_prompt_adapters.touch(prompt_adapter.id)
            was_added = False
        return was_added

    def activate_prompt_adapter(
        self,
        prompt_adapter_id: int,
    ) -> bool:
        if prompt_adapter_id not in self._active_prompt_adapters and len(
                self._active_prompt_adapters) >= self.prompt_adapter_slots:
            self._active_prompt_adapters.remove_oldest()
        result = super().activate_prompt_adapter(prompt_adapter_id)
        # We always touch to update the LRU cache order
        self._active_prompt_adapters.touch(prompt_adapter_id)
        return result

    def remove_oldest_prompt_adapter(self) -> bool:
        if len(self._registered_prompt_adapters) > 0:
            self._registered_prompt_adapters.remove_oldest()
            return True
        return False


def create_prompt_adapter_manager(
        model: nn.Module,
        prompt_adapter_manager_cls: Type[
            PromptAdapterModelManager] = PromptAdapterModelManager,
        **kwargs) -> PromptAdapterModelManager:
    """Create a PromptAdapterModel for a given model."""
    prompt_adapter_manager = prompt_adapter_manager_cls(model=model, **kwargs)
    return prompt_adapter_manager
