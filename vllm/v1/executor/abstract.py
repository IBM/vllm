from abc import ABC, abstractmethod
from typing import Type

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput


class Executor(ABC):
    """Abstract class for executors."""

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> Type["Executor"]:
        executor_class: Type[Executor]
        distributed_executor_backend = (
            vllm_config.parallel_config.distributed_executor_backend)
        if distributed_executor_backend == "ray":
            from vllm.executor.ray_distributed_executor import (  # noqa
                RayDistributedExecutor)
            executor_class = RayDistributedExecutor
        elif distributed_executor_backend == "mp":
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor
            executor_class = MultiprocExecutor
        else:
            assert (distributed_executor_backend is None)
            from vllm.v1.executor.uniproc_executor import UniprocExecutor
            executor_class = UniprocExecutor
        return executor_class

    @abstractmethod
    def __init__(self, vllm_config: VllmConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    def initialize(self, kv_cache_config: KVCacheConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    def determine_available_memory(self) -> int:  # in bytes
        raise NotImplementedError

    @abstractmethod
    def get_kv_cache_spec(self) -> KVCacheSpec:
        raise NotImplementedError

    @abstractmethod
    def execute_model(
        self,
        scheduler_output,
    ) -> ModelRunnerOutput:
        raise NotImplementedError

    @abstractmethod
    def profile(self, is_start: bool = True):
        raise NotImplementedError

    @abstractmethod
    def shutdown(self):
        pass

    @abstractmethod
    def check_health(self) -> None:
        raise NotImplementedError