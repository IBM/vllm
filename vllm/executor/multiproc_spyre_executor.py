import json
import os
import platform
import signal
import threading
import weakref
from functools import partial
from typing import Any, List, Set, Tuple

import torch

from vllm.executor.multiproc_worker_utils import (ProcessWorkerWrapper,
                                                  ResultHandler, WorkerMonitor)
from vllm.executor.spyre_executor import SpyreExecutor, create_worker
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.utils import (get_distributed_init_method, get_open_port,
                        get_vllm_instance_id)

logger = init_logger(__name__)


class MultiprocessingSpyreExecutor(SpyreExecutor):
    """Python multiprocessing-based multi-Spyre executor"""

    uses_ray: bool = False

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks.

        This invokes `determine_num_available_blocks` on each worker and takes
        the min of the results, guaranteeing that the selected cache sizes are
        compatible with all workers.

        Returns:
            - tuple[num_gpu_blocks, num_cpu_blocks]
        """
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self._run_workers("determine_num_available_blocks", )

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)

        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache in all workers.
        """

        # NOTE: We log here to avoid multiple logs when number of workers is
        # greater than one. We could log in the engine, but not all executors
        # have GPUs.
        logger.info("# GPU blocks: %d, # CPU blocks: %d", num_gpu_blocks,
                    num_cpu_blocks)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._run_workers("initialize_cache",
                          num_gpu_blocks=num_gpu_blocks,
                          num_cpu_blocks=num_cpu_blocks)

    def _init_executor(self) -> None:
        self._check_executor_parameters()

        # Create the parallel GPU workers.
        world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size

        # Ensure that VLLM_INSTANCE_ID is set, to be inherited by workers
        os.environ["VLLM_INSTANCE_ID"] = get_vllm_instance_id()

        # Disable torch async compiling which won't work with daemonic processes
        # [tom] it doesn't seme to work setting this from the code, we need to
        # set at command line. hopefully will be fixed by upgrading torch.
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        # Configure thread parallelism if OMP_NUM_THREADS isn't set
        #
        # Helps to avoid CPU contention. The default of spawning a thread per
        # core combined with multiprocessing for each GPU can have a negative
        # impact on performance. The contention is amplified when running in a
        # container where CPU limits can cause throttling.
        default_omp_num_threads = 1
        if "OMP_NUM_THREADS" not in os.environ and (
                current_parallelism :=
                torch.get_num_threads()) > default_omp_num_threads:
            logger.warning(
                "Reducing Torch parallelism from %d threads to %d to avoid "
                "unnecessary CPU contention. Set OMP_NUM_THREADS in the "
                "external environment to tune this value as needed.",
                current_parallelism, default_omp_num_threads)
            os.environ["OMP_NUM_THREADS"] = str(default_omp_num_threads)
            torch.set_num_threads(default_omp_num_threads)

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # 127.0.0.1 for communication.
        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port())

        self.workers: List[ProcessWorkerWrapper] = []
        # This is the list of workers that are rank 0 of each TP group EXCEPT
        # global rank 0. These are the workers that will broadcast to the
        # rest of the workers.
        self.tp_driver_workers: List[ProcessWorkerWrapper] = []
        # This is the list of workers that are not drivers and not the first
        # worker in a TP group. These are the workers that will be
        # broadcasted to.
        self.non_driver_workers: List[ProcessWorkerWrapper] = []

        if world_size == 1:
            self.worker_monitor = None
        else:
            result_handler = ResultHandler()
            for rank in range(1, world_size):
                worker = ProcessWorkerWrapper(
                    result_handler,
                    partial(
                        create_worker,
                        **self._get_create_worker_kwargs(
                            rank=rank,
                            local_rank=rank,
                            distributed_init_method=distributed_init_method,
                        )))
                self.workers.append(worker)
                if rank % tensor_parallel_size == 0:
                    self.tp_driver_workers.append(worker)
                else:
                    self.non_driver_workers.append(worker)

            self.worker_monitor = WorkerMonitor(self.workers, result_handler)
            result_handler.start()
            self.worker_monitor.start()

        # Set up signal handlers to shutdown the executor cleanly
        # sometimes gc does not work well

        # Use weakref to avoid holding a reference to self
        ref = weakref.ref(self)

        def shutdown(signum, frame):
            if executor := ref():
                executor.shutdown()

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, shutdown)
            signal.signal(signal.SIGTERM, shutdown)

        self.driver_worker = self._create_worker(
            distributed_init_method=distributed_init_method)
        self._run_workers("init_device")
        self._run_workers("load_model")

    def _check_executor_parameters(self):
        world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size

        # Read number of Spyre cards from senlib config file
        if platform.machine() == "s390x":
            spyre_device_count = int(os.getenv("AIU_WORLD_SIZE", 1))
        else:
            with open("/etc/aiu/senlib_config.json", 'rb') as f:
                config = json.load(f)
            spyre_device_count = len(config["GENERAL"]["sen_bus_id"])

        # Use confusing message for more common TP-only case.
        assert tensor_parallel_size <= spyre_device_count, (
            f"please set tensor_parallel_size ({tensor_parallel_size}) "
            f"to less than max local gpu count ({spyre_device_count})")

        assert world_size <= spyre_device_count, (
            f"please ensure that world_size ({world_size}) "
            f"is less than than max local gpu count ({spyre_device_count})")

    def shutdown(self):
        if (worker_monitor := getattr(self, "worker_monitor",
                                      None)) is not None:
            worker_monitor.close()

    def execute_model(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[SamplerOutput]:

        output = self._run_workers("execute_model",
                                   execute_model_req=execute_model_req)
        return output[0]

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return self._run_workers(
            "add_lora",
            lora_request=lora_request,
        )

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self._run_workers(
            "remove_lora",
            lora_id=lora_id,
        )

    def pin_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self._run_workers(
            "pin_lora",
            lora_id=lora_id,
        )

    def list_loras(self) -> Set[int]:
        return self._run_workers("list_loras")

    def _run_workers(
        self,
        method: str,
        *args,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers.
        """

        # Start all remote workers first.
        worker_outputs = [
            worker.execute_method(method, *args, **kwargs)
            for worker in self.workers
        ]

        driver_worker_method = getattr(self.driver_worker, method)
        driver_worker_output = driver_worker_method(*args, **kwargs)

        # Get the results of the workers.
        return [driver_worker_output
                ] + [output.get() for output in worker_outputs]

    def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        if self.worker_monitor is not None and not self.worker_monitor.is_alive(
        ):
            raise RuntimeError("Worker processes are not running")


class MultiprocessingSpyreExecutorAsync(MultiprocessingSpyreExecutor):

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[SamplerOutput]:

        # this is not really async, rather this a blocking call.
        # there may well be perf implications, not sure.
        # However, Spyre does not seem to play nice with asyncio.
        # I think vLLM is moving away from the AsyncLLMEngine,
        # so let's not waste time here for now.
        output = self._run_workers("execute_model",
                                   execute_model_req=execute_model_req)

        return output[0]

    async def stop_remote_worker_execution_loop_async(self) -> None:
        return
