import os
from typing import Any, Dict, List, Optional, Set, Tuple

from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async)
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class SpyreExecutor(ExecutorBase):

    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert (self.lora_config is
                None), "LoRA is not supported for Spyre backend."
        assert (not self.speculative_config
                ), "Speculative decoding not yet supported for Spyre backend."

        #assert self.parallel_config.world_size == 1, (
        #    "SpyreExecutor only supports single Spyre card.")

        if os.getenv(key='SPYRE_PYTEST_DEBUG', default='0') == '1':
            import debugpy
            host_addr = os.getenv(key='SPYRE_PYTEST_DBG_ADDR',
                                  default='0.0.0.0')
            debugpy.listen((host_addr, 5678))
            print(f"[debugpy] {host_addr}: wait for client...\n")
            debugpy.wait_for_client()

        self.driver_worker = self._create_worker()
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def _get_worker_kwargs(
            self,
            local_rank: int = 0,
            rank: int = 0,
            distributed_init_method: Optional[str] = None) -> Dict[str, Any]:
        """Return worker init args for a given rank."""
        if distributed_init_method is None:
            distributed_init_method = get_distributed_init_method(
                get_ip(), get_open_port())
        return dict(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=(not self.parallel_config)
            or (rank % self.parallel_config.tensor_parallel_size == 0),
        )

    def _create_worker(self,
                       local_rank: int = 0,
                       rank: int = 0,
                       distributed_init_method: Optional[str] = None):

        wrapper = WorkerWrapperBase(vllm_config=self.vllm_config)

        wrapper.init_worker(**self._get_worker_kwargs(
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method))

        return wrapper.worker

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        return self.driver_worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:

        assert execute_model_req.num_lookahead_slots == 0, (
            "lookahead not supported for Spyre backend.")

        output = self.driver_worker.execute_model(execute_model_req)

        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.driver_worker.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.driver_worker.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self.driver_worker.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.driver_worker.list_loras()

    def add_prompt_adapter(self, prompt_adapter_request) -> bool:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the Spyre backend.")

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the Spyre backend.")

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the Spyre backend.")

    def list_prompt_adapters(self) -> Set[int]:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the Spyre backend.")

    def check_health(self) -> None:
        # SpyreExecutor will always be healthy as long as
        # it's running.
        return


class SpyreExecutorAsync(SpyreExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[SamplerOutput]:
        output = await make_async(
            self.driver_worker.execute_model
        )(seq_group_metadata_list=execute_model_req.seq_group_metadata_list, )
        return output

    async def check_health_async(self) -> None:
        # SpyreExecutor will always be healthy as long as
        # it's running.
        return
