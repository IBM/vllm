"""Some methods for producing logs similar to TGIS"""
import logging
from typing import List, Optional, Union

from google.protobuf import text_format

from vllm.entrypoints.grpc.pb.generation_pb2 import (BatchedGenerationRequest,
                                                     GenerationResponse,
                                                     Parameters,
                                                     SingleGenerationRequest,
                                                     StopReason)
from vllm.sequence import RequestMetrics


def log_response(
    request: Union[BatchedGenerationRequest, SingleGenerationRequest],
    response: GenerationResponse,
    engine_metrics: Optional[RequestMetrics],
    start_time: float,
    logger: logging.Logger,
    sub_request_num: int = 0,
):
    if isinstance(request, BatchedGenerationRequest):
        # unary case
        request_count = len(request.requests)
        if request_count == 1:
            kind_log = "Request"
        else:
            kind_log = (f"Sub-request {sub_request_num} from batch of "
                        f"{request_count}")
        _log_response(inputs=[r.text for r in request.requests],
                      response=response,
                      params=request.params,
                      prefix_id=request.prefix_id,
                      engine_metrics=engine_metrics,
                      start_time=start_time,
                      kind_log=kind_log,
                      method_str="generate",
                      logger=logger)
    else:
        # streaming case
        _log_response(inputs=[request.request.text],
                      response=response,
                      params=request.params,
                      prefix_id=request.prefix_id,
                      engine_metrics=engine_metrics,
                      start_time=start_time,
                      kind_log="Streaming response",
                      method_str="generate_stream",
                      logger=logger)


def log_error(request: Union[BatchedGenerationRequest,
                             SingleGenerationRequest], exception: Exception,
              logger: logging.Logger):
    """Logs errors similar to how the TGIS server does"""
    params = request.params
    paramstr = text_format.MessageToString(params, as_one_line=True)
    prefix_id = request.prefix_id

    if isinstance(request, BatchedGenerationRequest):
        method_str = "generate"
        inputs = [r.text for r in request.requests]
    else:
        method_str = "generate_stream"
        inputs = [request.request.text]

    short_input = [_truncate(input_, 32) for input_ in inputs]
    input_chars = sum(len(input_) for input_ in inputs)

    span_str = (f"{method_str}{{input={short_input} prefix_id={prefix_id} "
                f"input_chars=[{input_chars}] params={paramstr} ")

    # Using %s to format the exception to only print the exception's message
    # like TGIS does. (This is intentionally not using exc_info=True)
    logger.error("%s: %s", span_str, exception)


def _log_response(inputs: List[str], params: Parameters, prefix_id: str,
                  response: GenerationResponse,
                  engine_metrics: Optional[RequestMetrics], start_time: float,
                  kind_log: str, method_str: str, logger: logging.Logger):
    """Logs responses similar to how the TGIS server does"""
    # This time contains both request validation and tokenization
    if engine_metrics is not None:
        tokenization_time = engine_metrics.arrival_time - start_time
        inference_time = (engine_metrics.last_token_time -
                          engine_metrics.first_scheduled_time)
        queue_time = engine_metrics.time_in_queue
        time_per_token = _safe_div(inference_time,
                                   response.generated_token_count)
        total_time = engine_metrics.last_token_time - start_time
    else:
        logger.warning("No engine metrics for request, cannot log timing info")
        tokenization_time = inference_time = queue_time = time_per_token =\
            total_time = 0
    output_len = len(response.text)
    short_output = _truncate(response.text, 32)
    short_input = [_truncate(input_, 32) for input_ in inputs]
    input_chars = sum(len(input_) for input_ in inputs)

    paramstr = text_format.MessageToString(params, as_one_line=True)
    span_str = (f"{method_str}{{input={short_input} prefix_id={prefix_id} "
                f"input_chars=[{input_chars}] params={paramstr} "
                f"tokenization_time={tokenization_time * 1e3:.2f}ms "
                f"queue_time={queue_time * 1e3:.2f}ms "
                f"inference_time={inference_time * 1e3:.2f}ms "
                f"time_per_token={time_per_token * 1e3:.2f}ms "
                f"total_time={total_time * 1e3:.2f}ms "
                f"input_toks={response.input_token_count}}}")
    stop_reason_str = StopReason.Name(response.stop_reason)

    if response.stop_reason == StopReason.ERROR:
        level = logging.ERROR
    elif response.stop_reason in {
            StopReason.CANCELLED, StopReason.TOKEN_LIMIT
    }:
        level = logging.WARN
    else:
        level = logging.INFO
    logger.log(level,
               "%s: %s generated %d tokens before %s, output %d chars: %s",
               span_str, kind_log, response.generated_token_count,
               stop_reason_str, output_len, short_output)


def _truncate(text: str, len_: int) -> bytes:
    """Truncates a string and escapes control characters"""
    text = f"{text:.{len_}}..." if len(text) > len_ else text
    return text.encode("unicode_escape")


def _safe_div(a: float, b: float, *, default: float = 0.0) -> float:
    """Simple safe division with a default answer for divide-by-zero.
    """
    try:
        return a / b
    except ZeroDivisionError:
        return default
