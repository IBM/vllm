from dataclasses import dataclass


@dataclass
class PromptAdapterRequest:
    """
    Request for a Prompt adapter.

    Note that this class should be be used internally. For online
    serving, it is recommended to not allow users to use this class but
    instead provide another layer of abstraction to prevent users from
    accessing unauthorized Prompt adapters.

    prompt_adapter_id must be globally unique for a given adapter.
    This is currently not enforced in vLLM.
    """

    prompt_adapter_name: str
    prompt_adapter_id: int
    prompt_adapter_local_path: str
    prompt_adapter_num_virtual_tokens: int

    def __post_init__(self):
        if self.prompt_adapter_id < 1:
            raise ValueError(
                f"lora_int_id must be > 0, got {self.prompt_adapter_id}")

    def __eq__(self, value: object) -> bool:
        return isinstance(
            value, PromptAdapterRequest
        ) and self.prompt_adapter_id == value.prompt_adapter_id

    def __hash__(self) -> int:
        return self.prompt_adapter_id
