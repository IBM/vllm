import asyncio
import concurrent.futures
from copy import copy
from re import escape as regex_escape
from typing import Optional, Tuple, Union

import vllm.model_executor.guided_decoding.outlines_decoding as outlines_decoding  # noqa: E501
from vllm.entrypoints.grpc.pb.generation_pb2 import DecodingParameters
from vllm.model_executor.guided_decoding.outlines_decoding import (
    GuidedDecodingMode, _get_cached_logits_processor)
from vllm.model_executor.guided_decoding.outlines_logits_processors import (
    JSONLogitsProcessor, RegexLogitsProcessor)
from vllm.sampling_params import LogitsProcessor, LogitsProcessorFactory


async def get_outlines_guided_decoding_logits_processor(
        decoding_params: DecodingParameters,
        tokenizer) -> Union[JSONLogitsProcessor, RegexLogitsProcessor, None]:
    """
    Check for guided decoding parameters
    and get the necessary logits processor for the given guide.
    We cache logit processors by (guide, tokenizer), and on cache hit
    we make a shallow copy to reuse the same underlying FSM.
    """
    guide, mode = _get_guide_and_mode(decoding_params)
    if not guide:
        return None

    if outlines_decoding.global_thread_pool is None:
        outlines_decoding.global_thread_pool = (
            concurrent.futures.ThreadPoolExecutor(max_workers=2))
    loop = asyncio.get_running_loop()

    result = await loop.run_in_executor(
        outlines_decoding.global_thread_pool,
        _get_cached_logits_processor,
        guide,
        tokenizer,
        mode,
        None,  # guided_whitespace_pattern - TBD
    )

    logits_processor = copy(result)
    # reset logits processor's internal state
    logits_processor.init_state()
    return logits_processor


def _get_guide_and_mode(
    decoding_params: DecodingParameters,
) -> Union[Tuple[str, GuidedDecodingMode], Tuple[None, None]]:
    guided = decoding_params.WhichOneof("guided")
    if guided is not None:
        if guided == "json_schema":
            return decoding_params.json_schema, GuidedDecodingMode.JSON
        if guided == "regex":
            return decoding_params.regex, GuidedDecodingMode.REGEX
        if guided == "choice":
            choice_list = decoding_params.choice.choices
            if len(choice_list) < 2:
                raise ValueError("Must provide at least two choices")
            # choice just uses regex
            choices = [regex_escape(str(choice)) for choice in choice_list]
            choices_regex = "(" + "|".join(choices) + ")"
            return choices_regex, GuidedDecodingMode.CHOICE
        if guided == "grammar":
            return decoding_params.grammar, GuidedDecodingMode.GRAMMAR
        if decoding_params.format == DecodingParameters.JSON:
            return outlines_decoding.JSON_GRAMMAR, GuidedDecodingMode.GRAMMAR
    return None, None


class GuidedDecodingLogitsProcessorFactory(LogitsProcessorFactory):

    def __init__(self, decoding_params: DecodingParameters, tokenizer):
        self.decoding_params = decoding_params
        self.tokenizer = tokenizer

    def _adapter(self):
        try:
            asyncio.get_running_loop()
            task = asyncio.create_task(
                get_outlines_guided_decoding_logits_processor(
                    self.decoding_params, self.tokenizer))
            yield from task
            return task.result()
        except RuntimeError:
            yield asyncio.run(
                get_outlines_guided_decoding_logits_processor(
                    self.decoding_params, self.tokenizer))

    def get_processor(self) -> LogitsProcessor:
        return next(self._adapter())

    async def get_processor_async(self) -> LogitsProcessor:
        return await asyncio.create_task(self._adapter())


def get_outlines_guided_decoding_logits_processor_factory(
        decoding_params: DecodingParameters,
        tokenizer) -> Optional[LogitsProcessorFactory]:

    guide, _ = _get_guide_and_mode(decoding_params)
    if not guide:
        return None

    return GuidedDecodingLogitsProcessorFactory(decoding_params, tokenizer)
