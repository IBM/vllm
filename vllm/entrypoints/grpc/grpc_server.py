import argparse
import dataclasses
import inspect
import logging
import time
import uuid
from typing import (Any, AsyncIterator, Dict, List, MutableSequence, Optional,
                    Tuple, Union)

import grpc
from google.protobuf import text_format
from grpc import StatusCode, aio
from grpc._cython.cygrpc import AbortError
from grpc.aio import ServicerContext
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm import (AsyncLLMEngine, CompletionOutput, RequestOutput,
                  SamplingParams)
from vllm.config import ModelConfig
from vllm.entrypoints.grpc.pb import generation_pb2_grpc
# yapf: disable
from vllm.entrypoints.grpc.pb.generation_pb2 import (BatchedGenerationRequest,
                                                     BatchedGenerationResponse,
                                                     BatchedTokenizeRequest,
                                                     BatchedTokenizeResponse,
                                                     DecodingMethod,
                                                     GenerationResponse,
                                                     ModelInfoRequest,
                                                     ModelInfoResponse,
                                                     Parameters,
                                                     ResponseOptions,
                                                     SingleGenerationRequest,
                                                     StopReason, TokenInfo,
                                                     TokenizeResponse)
from vllm.entrypoints.grpc.validation import validate_input, validate_params
from vllm.entrypoints.openai.serving_completion import merge_async_iterators
from vllm.logger import init_logger
from vllm.sequence import Logprob
from vllm.tgis_utils.logits_processors import (LengthPenaltyWarper,
                                               TypicalLogitsWarperWrapper)
from vllm.transformers_utils.tokenizer_group import BaseTokenizerGroup

logger = init_logger(__name__)

@dataclasses.dataclass
class Times:
    """Container tracking times (in seconds) when requests start and finish """
    # When control enters Generate or GenerateStream
    request_start: float
    # When the request is sent to the vLLM engine
    engine_start: float = 0
    # When the stream from the vLLM engine closes
    end: float = 0


def with_default(value: Any, default: Any) -> Any:
    return value if value else default


async def _handle_exception(e: Exception, func, *args, **kwargs):
    # We don't log AbortErrors since these correspond to gRPC errors
    # intentionally raised during handling of requests.
    if not isinstance(e, AbortError):
        if type(e).__name__ == "torch.cuda.OutOfMemoryError":  #TODO check
            context = kwargs.get("context", None) or args[-1]
            logger.exception(f"{func.__name__} caused GPU OOM error")
            await context.abort(StatusCode.RESOURCE_EXHAUSTED, str(e))
        logger.exception(f"{func.__name__} failed")
    raise e


def log_rpc_handler_errors(func):
    if inspect.isasyncgenfunction(func):

        async def func_with_log(*args, **kwargs):
            try:
                async for val in func(*args, **kwargs):
                    yield val
            except Exception as e:
                await _handle_exception(e, func, *args, **kwargs)
    else:

        async def func_with_log(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                await _handle_exception(e, func, *args, **kwargs)

    return func_with_log


class TextGenerationService(generation_pb2_grpc.GenerationServiceServicer):

    def __init__(self, engine: AsyncLLMEngine, args: argparse.Namespace):
        self.engine: AsyncLLMEngine = engine

        # These set in _post_init()
        self.tokenizer_group: BaseTokenizerGroup = None
        self.tokenizer: Union[PreTrainedTokenizer,
                              PreTrainedTokenizerFast] = None
        self.config: ModelConfig = None

        self.max_max_new_tokens = args.max_new_tokens
        self.skip_special_tokens = not args.output_special_tokens
        self.default_include_stop_seqs = args.default_include_stop_seqs

    async def _post_init(self):
        self.config = await self.engine.get_model_config()
        self.tokenizer_group = await self.engine.get_tokenizer_group()
        self.tokenizer = await self.engine.get_tokenizer()

    @log_rpc_handler_errors
    async def Generate(self, request: BatchedGenerationRequest,
                       context: ServicerContext) -> BatchedGenerationResponse:
        start_time = time.time()
        request_id = self.request_id(context)
        sampling_params, deadline = await self._validate_and_convert_params(
            request.params, context)
        truncate_input_tokens = with_default(
            request.params.truncate_input_tokens, None)
        request_count = len(request.requests)

        generators = []
        timing_infos = []
        max_is_token_limit = [False] * request_count
        for i, req in enumerate(request.requests):
            input_ids, max_is_token_limit[i]\
                = await self._validate_prompt_and_tokenize(
                    sampling_params, truncate_input_tokens, req.text, context)
            timing_info = Times(request_start=start_time)
            timing_infos.append(timing_info)
            generators.append(
                self.timed_generator(
                    # prompt is supplied for observability, the text is not
                    # re-tokenized when `prompt_token_ids` is supplied
                    self.engine.generate(prompt=req.text,
                                         sampling_params=sampling_params,
                                         request_id=f"{request_id}-{i}",
                                         prompt_token_ids=input_ids),
                    timing_info))

        # TODO handle cancellation
        result_generator: AsyncIterator[Tuple[
            int, RequestOutput]] = merge_async_iterators(*generators)

        resp_options = request.params.response
        responses: List = [None] * request_count
        time_limit_reached = False
        async for i, res in result_generator:
            # if await raw_request.is_disconnected():
            #     # Abort the request if the client disconnects.
            #     await self.engine.abort(f"{request_id}-{i}")
            #     return self.create_error_response("Client disconnected")
            responses[i] = res

            if deadline is not None and time.time(
            ) >= deadline and None not in responses:
                for j in range(request_count):
                    await self.engine.abort(f"{request_id}-{j}")
                time_limit_reached = True
                break

        for i, res in enumerate(responses):
            # Text prompt is not returned if only token_ids are passed
            res.prompt = request.requests[i].text
            response = self._convert_output(res.outputs[0], resp_options,
                                            max_is_token_limit[i],
                                            time_limit_reached)
            response = self._convert_input_details(res, resp_options,
                                                       sampling_params,
                                                       response)
            if request_count == 1:
                kind_log = "Request"
            else:
                kind_log = f"Sub-request {i} from batch of {request_count}"
            self.log_response(req=request,
                              res=response,
                              times=timing_infos[i],
                              kind_log=kind_log,
                              method_str="generate")
            responses[i] = response

        return BatchedGenerationResponse(responses=responses)

    @log_rpc_handler_errors
    async def GenerateStream(
            self, request: SingleGenerationRequest,
            context: ServicerContext) -> AsyncIterator[GenerationResponse]:
        timing_info = Times(request_start=time.time())
        request_id = self.request_id(context)
        sampling_params, deadline = await self._validate_and_convert_params(
            request.params, context)
        truncate_input_tokens = with_default(
            request.params.truncate_input_tokens, None)

        input_ids, max_is_tok_limit = await self._validate_prompt_and_tokenize(
            sampling_params, truncate_input_tokens, request.request.text,
            context)

        result_generator = self.timed_generator(
            self.engine.generate(
                # prompt is supplied for observability, the text is not
                # re-tokenized when `prompt_token_ids` is supplied
                prompt=request.request.text,
                sampling_params=sampling_params,
                request_id=request_id,
                prompt_token_ids=input_ids,
            ),
            timing_info
        )

        resp_options = request.params.response

        first = True
        first_response = None
        last_output_length = 0
        last_token_count = 0
        time_limit_reached = False
        full_output = ""
        full_output_token_count = 0
        #TODO handle cancellation
        #TODO: Time and log
        async for result in result_generator:
            if first:
                # Text prompt is not returned if only token_ids are passed
                result.prompt = request.request.text
                first_response = self._convert_input_details(
                    result, resp_options, sampling_params,
                    GenerationResponse())
                yield first_response
                first = False

            output = result.outputs[0]

            if deadline is not None and time.time() >= deadline:
                await self.engine.abort(request_id)
                time_limit_reached = True

            # Convert output text and token_ids to deltas
            yield self._convert_output(output, resp_options, max_is_tok_limit,
                                       time_limit_reached, last_output_length,
                                       last_token_count)
            if time_limit_reached:
                break

            last_output_length = len(output.text)
            last_token_count = len(output.token_ids)
            # Accumulate full output for logging
            full_output += output.text
            full_output_token_count += last_token_count

        # Edit up the first_response for logging purposes only
        if first_response is None:
            # We didn't output anything!
            return
        first_response.text = full_output
        first_response.generated_token_count = full_output_token_count
        self.log_response(req=request,
                          res=first_response,
                          times=timing_info,
                          kind_log="Streaming response",
                          method_str="generate_stream")

    def _convert_input_details(
            self, result: RequestOutput, resp_options: ResponseOptions,
            sampling_params: SamplingParams,
            response: GenerationResponse) -> GenerationResponse:

        response.input_token_count = len(result.prompt_token_ids)
        if resp_options.input_tokens:
            self._convert_tokens(
                result.prompt_token_ids,
                result.prompt_logprobs,
                resp_options.token_logprobs,
                resp_options.token_ranks,
                resp_options.top_n_tokens,
                response.input_tokens,
            )

        if resp_options.input_text:
            response.text = result.prompt if not response.text \
                else result.prompt + response.text

        if sampling_params.seed is not None:
            response.seed = sampling_params.seed
        return response

    def _convert_output(self,
                        output: CompletionOutput,
                        resp_options: ResponseOptions,
                        max_is_token_limit: bool,
                        time_limit_reached: bool = False,
                        text_start_offset: int = 0,
                        token_start_offset: int = 0) -> GenerationResponse:

        stop_reason, stop_sequence = self._convert_reason(
            output, max_is_token_limit, time_limit_reached)
        response = GenerationResponse(
            text=output.text[text_start_offset:],
            generated_token_count=len(output.token_ids),
            stop_reason=stop_reason,
            stop_sequence=stop_sequence,
        )

        if resp_options.generated_tokens:
            self._convert_tokens(
                output.token_ids,
                output.logprobs,
                resp_options.token_logprobs,
                resp_options.token_ranks,
                resp_options.top_n_tokens,
                response.tokens,
                token_start_offset,
            )
        return response

    @staticmethod
    def request_id(context: ServicerContext) -> str:
        return uuid.uuid4().hex

    async def _validate_and_convert_params(
            self, params: Parameters, context: ServicerContext
    ) -> Tuple[SamplingParams, Optional[float]]:
        """ Returns (sampling_params, deadline) """
        # First run TGIS validation to raise errors that match the TGIS api
        try:
            validate_params(params, self.max_max_new_tokens)
        except ValueError as tgis_validation_error:
            await context.abort(StatusCode.INVALID_ARGUMENT,
                                str(tgis_validation_error))

        resp_options = params.response
        sampling = params.sampling
        stopping = params.stopping
        greedy = params.method == DecodingMethod.GREEDY

        max_new_tokens: Optional[int] = None
        if stopping.max_new_tokens > 0:
            max_new_tokens = stopping.max_new_tokens
        min_new_tokens = max(0, stopping.min_new_tokens)

        logprobs = 1 if (resp_options.token_logprobs
                         or resp_options.token_ranks) else 0
        top_n_tokens = resp_options.top_n_tokens
        if top_n_tokens:
            # vLLM will currently return logprobs for n+1 tokens
            # (selected token plus top_n excluding selected)
            logprobs += top_n_tokens
            if greedy and resp_options.token_logprobs:
                logprobs -= 1

        logprobs = with_default(logprobs, None)

        # GAPS:
        # - exp_decay_length_penalty

        # NEW FUNCTION TO ADD (later)
        # - presence penalty, freq penalty
        # - min_p
        # - beam search (with length_penalty, stop_early, n)

        # TBD (investigate more)
        # - best_of / n
        # - spaces_between_special_tokens
        # - skip_special_tokens (per request)
        # - stop_token_ids

        # to match TGIS, only including typical_p processing
        # when using sampling
        logits_processors = []

        if not greedy and 0.0 < sampling.typical_p < 1.0:
            logits_processors.append(
                TypicalLogitsWarperWrapper(mass=sampling.typical_p))
        if params.decoding.length_penalty is not None:
            length_penalty = (
                params.decoding.length_penalty.start_index,
                params.decoding.length_penalty.decay_factor,
            )
            logits_processors.append(
                LengthPenaltyWarper(length_penalty=length_penalty,
                                    eos_token_id=self.tokenizer.eos_token_id))

        time_limit_millis = stopping.time_limit_millis
        deadline = time.time(
        ) + time_limit_millis / 1000.0 if time_limit_millis > 0 else None

        try:
            sampling_params = SamplingParams(
                logprobs=logprobs,
                prompt_logprobs=logprobs
                if resp_options.input_tokens else None,
                max_tokens=max_new_tokens,
                min_tokens=min_new_tokens,
                temperature=with_default(sampling.temperature, 1.0)
                if not greedy else 0.0,
                top_k=with_default(sampling.top_k, -1),
                top_p=with_default(sampling.top_p, 1.0),
                seed=sampling.seed if sampling.HasField("seed") else None,
                repetition_penalty=with_default(
                    params.decoding.repetition_penalty, 1.0),
                logits_processors=logits_processors,
                stop=with_default(stopping.stop_sequences, None),
                include_stop_str_in_output=stopping.include_stop_sequence
                if stopping.HasField("include_stop_sequence") else
                self.default_include_stop_seqs,
                skip_special_tokens=self.skip_special_tokens,
            )
        except ValueError as vllm_validation_error:
            # There may be validation cases caught by vLLM that are not covered
            # by the TGIS api validation
            await context.abort(StatusCode.INVALID_ARGUMENT,
                                str(vllm_validation_error))

        return sampling_params, deadline

    @staticmethod
    def _convert_reason(output: CompletionOutput, max_is_token_limit: bool,
                        time_limit_reached: bool) -> Tuple['StopReason', str]:
        finish_reason = output.finish_reason
        stop_sequence = None
        if finish_reason is None:
            stop_reason = StopReason.TIME_LIMIT if (
                time_limit_reached) else StopReason.NOT_FINISHED
        elif finish_reason == "length":
            stop_reason = StopReason.TOKEN_LIMIT if (
                max_is_token_limit) else StopReason.MAX_TOKENS
        elif finish_reason == "stop":
            stop_reason = StopReason.STOP_SEQUENCE
            # TODO depends on https://github.com/vllm-project/vllm/pull/2976
            if hasattr(output, "stop_reason"):
                stop_str_or_tok = output.stop_reason
                if stop_str_or_tok is None:
                    stop_reason = StopReason.EOS_TOKEN
                elif isinstance(stop_str_or_tok, str):
                    stop_sequence = stop_str_or_tok
                else:
                    logger.warning(
                        f"Unexpected stop_reason type: {type(stop_str_or_tok)}"
                    )
        elif finish_reason == "abort":
            stop_reason = StopReason.CANCELLED
        else:
            logger.warning(f"Unrecognized finish_reason: {finish_reason}")
            stop_reason = StopReason.CANCELLED

        return stop_reason, stop_sequence

    def _convert_tokens(
        self,
        token_ids: list[int],
        logprobs_list: Optional[list[Dict[int, Logprob]]],
        include_logprobs: bool,
        include_ranks: bool,
        top_n_tokens: int,
        token_infos: MutableSequence[TokenInfo],  # OUT
        token_start_offset: int = 0,
    ):
        if token_start_offset:
            token_ids = token_ids[token_start_offset:]
            if logprobs_list is not None:
                logprobs_list = logprobs_list[token_start_offset:]
        #TODO later use get_lora_tokenizer here
        token_texts = self.tokenizer.convert_ids_to_tokens(token_ids)
        for i, text in enumerate(token_texts):
            token_info = TokenInfo(text=text)
            if logprobs_list is not None:
                logprobs = logprobs_list[i]
                # Logprobs entry will be None for first prompt token
                if logprobs is not None:
                    if include_logprobs or include_ranks:
                        logprob = logprobs[token_ids[i]]
                        if include_logprobs:
                            token_info.logprob = logprob.logprob
                        if include_ranks:
                            token_info.rank = logprob.rank
                    if top_n_tokens:
                        items = sorted(logprobs.items(),
                                       key=lambda item: item[1].logprob,
                                       reverse=True)[:top_n_tokens]
                        #TODO later use get_lora_tokenizer here
                        tt_texts = self.tokenizer.convert_ids_to_tokens(
                            [tid for tid, _ in items])
                        token_info.top_tokens.extend(
                            TokenInfo.TopToken(
                                text=tt_text,
                                logprob=(logprob.logprob
                                         if include_logprobs else None),
                            )
                            for tt_text, (_, logprob) in zip(tt_texts, items))
            token_infos.append(token_info)

    async def _validate_prompt_and_tokenize(
        self,
        sampling_params: SamplingParams,
        truncate_input_tokens: Optional[int],
        prompt: Optional[str],
        context: ServicerContext,
    ) -> Tuple[List[int], bool]:
        max_model_len = self.config.max_model_len
        # tokenize_kwargs = {"truncation": True,
        #                    "max_length": truncate_input_tokens} \
        #     if truncate_input_tokens is not None else {
        #       "truncation": True, "max_length": max_model_len + 1}
        tokenize_kwargs = {}

        input_ids = await self.tokenizer_group.encode_async(
            prompt, **tokenize_kwargs)

        #TODO this is temporary until truncation option is added
        # to the TokenizerGroup encode methods
        if truncate_input_tokens and truncate_input_tokens < len(input_ids):
            input_ids = input_ids[-truncate_input_tokens:]
            if not sampling_params.skip_special_tokens:
                add_bos_token = getattr(self.tokenizer, "add_bos_token", False)
                if add_bos_token:
                    input_ids[0] = self.tokenizer.bos_token_id
        # -----------------------------------------------

        token_num = len(input_ids)

        try:
            validate_input(sampling_params, token_num, max_model_len)
        except ValueError as tgis_validation_error:
            await context.abort(StatusCode.INVALID_ARGUMENT,
                                str(tgis_validation_error))

        max_new_tokens: Optional[int] = sampling_params.max_tokens
        max_is_token_limit = False
        if max_new_tokens is None:
            # TGIS has fixed default (of 20 I think), but I think fine to keep
            # default as effective max here, given paged attention
            sampling_params.max_tokens = min(self.max_max_new_tokens,
                                             max_model_len - token_num)
            max_is_token_limit = True
        elif token_num + max_new_tokens > max_model_len:
            sampling_params.max_tokens = max_model_len - token_num
            max_is_token_limit = True

        return input_ids, max_is_token_limit

    @staticmethod
    def truncate(text: str, len_: int) -> bytes:
        """Truncates a string and escapes control characters, for logging"""
        text = f"{text:.{len_}}..." if len(text) > len_ else text
        return text.encode("unicode_escape")

    @staticmethod
    async def timed_generator(generator: AsyncIterator[RequestOutput],
                              times: Times) -> AsyncIterator[RequestOutput]:
        """Injects some timing data around each result generator from the
        LLMEngine"""
        times.engine_start = time.time()
        async for val in generator:
            yield val
        times.end = time.time()

    def log_response(self,
                     req: BatchedGenerationRequest | SingleGenerationRequest,
                     res: GenerationResponse,
                     times: Times,
                     kind_log: str,
                     method_str: str):
        """Logs responses similar to how the TGIS server does"""
        # This contains both request validation and tokenization
        tokenization_time = times.engine_start - times.request_start
        llm_engine_time = times.end - times.engine_start
        time_per_token = self.safe_div(llm_engine_time,
                                       res.generated_token_count, default=0)
        total_time = times.request_start - times.end
        output_len = len(res.text)
        short_output = self.truncate(res.text, 32)
        short_input = [self.truncate(r.text, 32) for r in req.requests]
        input_chars = sum(len(r.text) for r in req.requests)

        paramstr = text_format.MessageToString(req.params, as_one_line=True)
        span_str = (f"{method_str}{{input={short_input} prefix_id={req.prefix_id} "
                    f"input_chars=[{input_chars}] params={paramstr} "
                    f"tokenization_time={tokenization_time * 1e3:.2f}ms "
                    f"queue_and_inference_time={llm_engine_time * 1e3:.2f}ms "
                    f"time_per_token={time_per_token * 1e3:.2f}ms "
                    f"total_time={total_time * 1e3:.2f}ms "
                    f"input_toks={res.input_token_count}}}")
        stop_reason_str = StopReason.Name(res.stop_reason)

        if res.stop_reason == StopReason.ERROR:
            level = logging.ERROR
        elif res.stop_reason in {StopReason.CANCELLED, StopReason.TOKEN_LIMIT}:
            level = logging.WARN
        else:
            level = logging.INFO
        logger.log(level,
                   f"{span_str}: {kind_log} generated "
                   f"{res.generated_token_count} tokens before "
                   f"{stop_reason_str}, output {output_len} chars: "
                   f"{short_output}")

    @staticmethod
    def safe_div(a: float, b: float, *, default: float) -> float:
        """Simple safe division with a default answer for divide-by-zero.
        Used for logging where we don't mind incorrect answers.
        """
        try:
            return a / b
        except ZeroDivisionError:
            return default

    @log_rpc_handler_errors
    async def Tokenize(self, request: BatchedTokenizeRequest,
                       context: ServicerContext) -> BatchedTokenizeResponse:
        #TODO implement these
        if request.return_offsets:
            await context.abort(StatusCode.INVALID_ARGUMENT,
                                "return_offsets not yet supported")
        if request.truncate_input_tokens:
            await context.abort(StatusCode.INVALID_ARGUMENT,
                                "truncate_input_tokens not yet supported")

        responses: List[TokenizeResponse] = []

        #TODO maybe parallelize, also move convert_ids_to_tokens
        # into the other threads
        for req in request.requests:
            token_ids = await self.tokenizer_group.encode_async(req.text)
            responses.append(
                TokenizeResponse(
                    token_count=len(token_ids),
                    tokens=None if not request.return_tokens else
                    self.tokenizer.convert_ids_to_tokens(token_ids)))

        return BatchedTokenizeResponse(responses=responses)

    @log_rpc_handler_errors
    async def ModelInfo(self, request: ModelInfoRequest,
                        context: ServicerContext) -> ModelInfoResponse:
        return ModelInfoResponse(
            # vLLM currently only supports decoder models
            model_kind=ModelInfoResponse.ModelKind.DECODER_ONLY,
            max_sequence_length=self.config.max_model_len,
            max_new_tokens=self.max_max_new_tokens,
        )


async def start_grpc_server(engine: AsyncLLMEngine,
                            args: argparse.Namespace) -> aio.Server:

    # Log memory summary after model is loaded
    from torch.cuda import memory_summary
    logger.info(memory_summary(engine.engine.device_config.device))

    server = aio.server()
    service = TextGenerationService(engine, args)
    await service._post_init()

    generation_pb2_grpc.add_GenerationServiceServicer_to_server(
        service, server)

    #TODO add reflection

    # SERVICE_NAMES = (
    #     generation_pb2.DESCRIPTOR.services_by_name["GenerationService"]
    #     .full_name,
    #     reflection.SERVICE_NAME,
    # )
    # reflection.enable_server_reflection(SERVICE_NAMES, server)

    host = "0.0.0.0" if args.host is None else args.host
    listen_on = f"{host}:{args.grpc_port}"
    ssl_keyfile = args.ssl_keyfile
    ssl_certfile = args.ssl_certfile
    ssl_ca_certs = args.ssl_ca_certs

    if ssl_keyfile and ssl_certfile:
        require_client_auth = False
        try:
            with open(ssl_keyfile, "rb") as f:
                ssl_key = f.read()
        except Exception as e:
            raise ValueError(
                f"Error reading `ssl_keyfile` file: {ssl_keyfile}") from e
        try:
            with open(ssl_certfile, "rb") as f:
                ssl_cert = f.read()
        except Exception as e:
            raise ValueError(
                f"Error reading `ssl_certfile` file: {ssl_certfile}") from e
        if ssl_ca_certs:
            require_client_auth = True
            try:
                with open(ssl_ca_certs, "rb") as f:
                    root_certificates = f.read()
            except Exception as e:
                raise ValueError(
                    f"Error reading `ssl_ca_certs` file: {ssl_ca_certs}"
                ) from e
        else:
            root_certificates = None
        server_credentials = grpc.ssl_server_credentials([(ssl_key, ssl_cert)],
                                                         root_certificates,
                                                         require_client_auth)
        server.add_secure_port(listen_on, server_credentials)
    else:
        server.add_insecure_port(listen_on)

    await server.start()
    logger.info(f"gRPC Server started at {listen_on}")

    return server
