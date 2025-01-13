from typing import Optional

from .interface import Platform, PlatformEnum


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
