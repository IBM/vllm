import logging
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, List, Optional, Set, Type
import torch
from vllm.prompt_adapter.models import (PromptAdapterModel,
                                        PromptAdapterModelManager,
                                        LRUCachePromptAdapterModelManager,
                                        create_prompt_adapter_manager)
from vllm.prompt_adapter.request import PromptAdapterRequest

logger = logging.getLogger(__name__)


class AbstractWorkerPromptAdapterManager(ABC):
    """Abstract class for managing Prompt Adapters on the worker side."""

    def __init__(self, device: torch.device):
        self.device = device

    @abstractproperty
    def is_enabled(self) -> bool:
        ...

    @abstractmethod
    def create_prompt_adapter_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        ...

    @abstractmethod
    def set_active_prompt_adapters(
            self, prompt_adapter_requests: List[PromptAdapterRequest]) -> None:
        ...

    @abstractmethod
    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        ...

    @abstractmethod
    def add_dummy_prompt_adapter(self,
                                 prompt_adapter_request: PromptAdapterRequest,
                                 rank: int) -> bool:
        ...

    @abstractmethod
    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        ...

    @abstractmethod
    def remove_all_prompt_adapters(self) -> bool:
        ...

    @abstractmethod
    def list_prompt_adapters(self) -> Set[int]:
        ...


class WorkerPromptAdapterManager(AbstractWorkerPromptAdapterManager):
    """WorkerPromptAdapterManager that manages prompt_adapter models on the worker side.

    Every request, the requested prompt_adapters will be loaded (unless they are already
    loaded), and every other prompt_adapter will be unloaded."""

    _prompt_adapter_manager_cls: Type[
        PromptAdapterModelManager] = PromptAdapterModelManager

    def __init__(
        self,
        device: torch.device,
        prompt_adapter_model_cls: Type[
            PromptAdapterModel] = PromptAdapterModel,
    ):
        self._prompt_adapter_manager: Optional[
            PromptAdapterModelManager] = None
        self._prompt_adapter_model_cls = prompt_adapter_model_cls
        self.device = device
        super().__init__(device)

    @property
    def is_enabled(self) -> bool:
        return True

    def create_prompt_adapter_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        prompt_adapter_manager = create_prompt_adapter_manager(
            model,
            prompt_adapter_manager_cls=self._prompt_adapter_manager_cls,
        )
        self._prompt_adapter_manager: PromptAdapterModelManager = prompt_adapter_manager
        return prompt_adapter_manager.model

    def set_active_prompt_adapters(
            self, prompt_adapter_requests: List[PromptAdapterRequest]) -> None:
        self._apply_prompt_adapters(prompt_adapter_requests)

    def _apply_prompt_adapters(
            self, prompt_adapter_requests: List[PromptAdapterRequest]) -> None:
        prompt_adapters_that_exist = self.list_prompt_adapters()
        prompt_adapters_map = {
            prompt_adapter_request.prompt_adapter_id: prompt_adapter_request
            for prompt_adapter_request in prompt_adapter_requests
            if prompt_adapter_request
        }
        if len(prompt_adapters_map
               ) > self._prompt_adapter_manager.prompt_adapter_slots:
            raise RuntimeError(
                f"Number of requested prompt_adapters ({len(prompt_adapters_map)}) is greater "
                "than the number of GPU prompt_adapter slots "
                f"({self._prompt_adapter_manager.prompt_adapter_slots}).")

        new_prompt_adapters = set(prompt_adapters_map)
        prompt_adapters_to_add = new_prompt_adapters - prompt_adapters_that_exist
        prompt_adapters_to_remove = prompt_adapters_that_exist - new_prompt_adapters

        for prompt_adapter_id in prompt_adapters_to_remove:
            self.remove_prompt_adapter(prompt_adapter_id)

        for prompt_adapter_id in prompt_adapters_to_add:
            self.add_prompt_adapter(prompt_adapters_map[prompt_adapter_id])

    def _load_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest
    ) -> PromptAdapterModel:
        # try:
        prompt_adapter = self._prompt_adapter_model_cls.from_local_checkpoint(
            prompt_adapter_request.prompt_adapter_local_path,
            prompt_adapter_id=prompt_adapter_request.prompt_adapter_id,
            torch_device=str(self.device))
        # except Exception as e:
        #     raise RuntimeError(
        #         f"Loading prompt_adapter {prompt_adapter_request.prompt_adapter_local_path} failed") from e
        return prompt_adapter

    def add_dummy_prompt_adapter(self,
                                 prompt_adapter_request: PromptAdapterRequest,
                                 rank: int) -> bool:
        pass
        # if prompt_adapter_request.prompt_adapter_int_id in self.list_prompt_adapters():
        #     return False
        # return self._prompt_adapter_manager.add_prompt_adapter(
        #     self._prompt_adapter_manager.create_dummy_prompt_adapter(prompt_adapter_request.prompt_adapter_int_id,
        #                                          rank, self.embedding_modules))

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        if prompt_adapter_request.prompt_adapter_id in self.list_prompt_adapters(
        ):
            return False
        prompt_adapter = self._load_prompt_adapter(prompt_adapter_request)
        loaded = self._prompt_adapter_manager.add_prompt_adapter(
            prompt_adapter)
        self._prompt_adapter_manager.activate_prompt_adapter(prompt_adapter.id)
        return loaded

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return self._prompt_adapter_manager.remove_prompt_adapter(
            prompt_adapter_id)

    def remove_all_prompt_adapters(self) -> bool:
        self._prompt_adapter_manager.remove_all_prompt_adapters()

    def list_prompt_adapters(self) -> Set[int]:
        return set(self._prompt_adapter_manager.list_prompt_adapters())


class LRUCacheWorkerPromptAdapterManager(WorkerPromptAdapterManager):
    """WorkerPromptAdapterManager that manages prompt_adapter models on the worker side.

    Uses an LRU Cache. Every request, the requested prompt_adapters will be loaded
    (unless they are already loaded) and least recently used prompt_adapters will
    be unloaded if the cache is above capacity."""

    _prompt_adapter_manager_cls: Type[
        LRUCachePromptAdapterModelManager] = LRUCachePromptAdapterModelManager

    def create_prompt_adapter_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        prompt_adapter_manager = create_prompt_adapter_manager(
            model, prompt_adapter_manager_cls=self._prompt_adapter_manager_cls)
        self._prompt_adapter_manager: LRUCachePromptAdapterModelManager = prompt_adapter_manager
        return prompt_adapter_manager.model

    def _apply_prompt_adapters(
            self, prompt_adapter_requests: List[PromptAdapterRequest]) -> None:
        prompt_adapters_map = {
            prompt_adapter_request.prompt_adapter_id: prompt_adapter_request
            for prompt_adapter_request in prompt_adapter_requests
            if prompt_adapter_request
        }
        if len(prompt_adapters_map
               ) > self._prompt_adapter_manager.prompt_adapter_slots:
            raise RuntimeError(
                f"Number of requested prompt_adapters ({len(prompt_adapters_map)}) is greater "
                "than the number of GPU prompt_adapter slots "
                f"({self._prompt_adapter_manager.prompt_adapter_slots}).")
        for prompt_adapter in prompt_adapters_map.values():
            self.add_prompt_adapter(prompt_adapter)

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        if prompt_adapter_request.prompt_adapter_id not in self.list_prompt_adapters(
        ):
            # Remove before we load the new prompt_adapter to save memory
            if len(self._prompt_adapter_manager
                   ) + 1 > self._prompt_adapter_manager.capacity:
                self._prompt_adapter_manager.remove_oldest_prompt_adapter()
            prompt_adapter = self._load_prompt_adapter(prompt_adapter_request)
            loaded = self._prompt_adapter_manager.add_prompt_adapter(
                prompt_adapter)
        else:
            # If the prompt_adapter is already loaded, just touch it to
            # update its position in the caches
            loaded = self._prompt_adapter_manager.get_prompt_adapter(
                prompt_adapter_request.prompt_adapter_id)
        self._prompt_adapter_manager.activate_prompt_adapter(
            prompt_adapter_request.prompt_adapter_id)
        return loaded
