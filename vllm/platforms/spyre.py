from typing import TYPE_CHECKING, Optional

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None
import vllm.envs as envs

from .interface import Platform, PlatformEnum

logger = init_logger(__name__)


class SpyrePlatform(Platform):
    _enum = PlatformEnum.SPYRE
    device_name: str = "spyre"
    device_type: str = "spyre"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "spyre"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        """
        Check if the current platform supports async output.
        """
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config

        if scheduler_config.is_multi_step:
            raise NotImplementedError

        if parallel_config.worker_cls == "auto":
            if envs.VLLM_USE_V1:
                raise NotImplementedError
            else:
                parallel_config.worker_cls = \
                    "vllm.worker.spyre_worker.SpyreWorker"

        cache_config = vllm_config.cache_config
        if cache_config:
            # spyre needs block_size = max_model_len
            vllm_config.cache_config.block_size = \
                vllm_config.model_config.max_model_len

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on Spyre.")
        return False