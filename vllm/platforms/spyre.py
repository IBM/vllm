from .interface import Platform, PlatformEnum


class SpyrePlatform(Platform):
    _enum = PlatformEnum.SPYRE

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "spyre"
