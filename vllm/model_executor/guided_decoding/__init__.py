import asyncio
from typing import Optional, Union

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest)
from vllm.model_executor.guided_decoding.lm_format_enforcer_decoding import (
    get_lm_format_enforcer_guided_decoding_logits_processor)
from vllm.model_executor.guided_decoding.lm_format_enforcer_decoding import (
    validate_request as lm_format_validate_request)
from vllm.model_executor.guided_decoding.outlines_decoding import (
    get_outlines_guided_decoding_logits_processor)
from vllm.model_executor.guided_decoding.outlines_decoding import (
    validate_request as outlines_validate_request)
from vllm.sampling_params import LogitsProcessor, LogitsProcessorFactory


async def get_guided_decoding_logits_processor(
        guided_decoding_backend: str, request: Union[CompletionRequest,
                                                     ChatCompletionRequest],
        tokenizer) -> Optional[LogitsProcessor]:
    if guided_decoding_backend == 'outlines':
        return await get_outlines_guided_decoding_logits_processor(
            request, tokenizer)
    if guided_decoding_backend == 'lm-format-enforcer':
        return await get_lm_format_enforcer_guided_decoding_logits_processor(
            request, tokenizer)

    raise ValueError(
        f"Unknown guided decoding backend '{guided_decoding_backend}'. "
        "Must be one of 'outlines, 'lm-format-enforcer'")


class GuidedDecodingLogitsProcessorFactory(LogitsProcessorFactory):

    def __init__(self, guided_decoding_backend: str,
                 request: Union[CompletionRequest,
                                ChatCompletionRequest], tokenizer):
        self.guided_decoding_backend = guided_decoding_backend
        self.request = request
        self.tokenizer = tokenizer

    def _adapter(self):
        try:
            asyncio.get_running_loop()
            task = asyncio.create_task(
                get_guided_decoding_logits_processor(
                    self.guided_decoding_backend, self.request,
                    self.tokenizer))
            yield from task
            return task.result()
        except RuntimeError:
            yield asyncio.run(
                get_guided_decoding_logits_processor(
                    self.guided_decoding_backend, self.request,
                    self.tokenizer))

    def get_processor(self) -> LogitsProcessor:
        return next(self._adapter())

    async def get_processor_async(self) -> LogitsProcessor:
        return await asyncio.create_task(self._adapter())


def get_guided_decoding_logits_processor_factory(
        guided_decoding_backend: str, request: Union[CompletionRequest,
                                                     ChatCompletionRequest],
        tokenizer) -> Optional[LogitsProcessorFactory]:

    if guided_decoding_backend == 'lm-format-enforcer':
        first = lm_format_validate_request(request, tokenizer)
        if first is None and request.guided_grammar:
            first = outlines_validate_request(request, tokenizer)

    elif guided_decoding_backend == 'outlines':
        first = outlines_validate_request(request, tokenizer)

    else:
        raise ValueError(
            f"Unknown guided decoding backend '{guided_decoding_backend}'. "
            "Must be one of 'outlines, 'lm-format-enforcer'")

    if first is not None:
        return GuidedDecodingLogitsProcessorFactory(guided_decoding_backend,
                                                    request, tokenizer)
    else:
        return None
