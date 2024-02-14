import argparse
import inspect
import time
import uuid

from grpc import aio, StatusCode

from typing import Optional, AsyncIterator, Dict, MutableSequence, Any, Union, Tuple, List

from grpc._cython.cygrpc import AbortError
from grpc.aio import ServicerContext
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.transformers_utils.tokenizer import TokenizerGroup
from vllm.logger import init_logger
from vllm.config import ModelConfig
from vllm.entrypoints.grpc.pb import generation_pb2_grpc
from vllm.entrypoints.grpc.pb.generation_pb2 import BatchedTokenizeRequest, BatchedGenerationRequest, \
    SingleGenerationRequest, ModelInfoRequest, BatchedTokenizeResponse, TokenizeResponse, ModelInfoResponse, \
    GenerationResponse, BatchedGenerationResponse, StopReason, TokenInfo, Parameters, DecodingMethod, ResponseOptions
from vllm.entrypoints.openai.serving_completion import merge_async_iterators
from vllm.sampling_params import LogitsProcessor
from vllm.tgis_utils.logits_processors import MinTokensLogitsProcessor, TypicalLogitsWarperWrapper
from vllm import AsyncLLMEngine, SamplingParams, RequestOutput, CompletionOutput

logger = init_logger(__name__)

MAX_TOP_N_TOKENS = 10

MAX_STOP_SEQS = 6
MAX_STOP_SEQ_LENGTH = 240


def with_default(value: Any, default: Any) -> Any:
    return value if value else default


async def _handle_exception(e: Exception, func, *args, **kwargs):
    # We don't log AbortErrors since these correspond to gRPC errors intentionally
    # raised during handling of requests.
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
        self.tokenizer_group: TokenizerGroup = engine.get_tokenizer_group()
        self.tokenizer: Union[
            PreTrainedTokenizer,
            PreTrainedTokenizerFast] = self.tokenizer_group.tokenizer
        self.config: ModelConfig = None

        self.max_max_new_tokens = args.max_new_tokens
        self.skip_special_tokens = not args.output_special_tokens
        self.default_include_stop_seqs = args.default_include_stop_seqs

    async def _post_init(self):
        self.config = await self.engine.get_model_config()

    @log_rpc_handler_errors
    async def Generate(self, request: BatchedGenerationRequest,
                       context: ServicerContext) -> BatchedGenerationResponse:
        request_id = self.request_id(context)
        sampling_params, deadline = await self._validate_and_convert_params(
            request.params, context)
        truncate_input_tokens = with_default(
            request.params.truncate_input_tokens, None)
        request_count = len(request.requests)

        generators = []
        max_is_token_limit = [False] * request_count
        for i, req in enumerate(request.requests):
            input_ids, max_is_token_limit[i]\
                = await self._validate_prompt_and_tokenize(
                    sampling_params, truncate_input_tokens, req.text, context)
            generators.append(
                self.engine.generate(None,
                                     sampling_params,
                                     f"{request_id}-{i}",
                                     prompt_token_ids=input_ids))

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
            responses[i] = self._convert_input_details(res, resp_options,
                                                       sampling_params,
                                                       response)

        return BatchedGenerationResponse(responses=responses)

    @log_rpc_handler_errors
    async def GenerateStream(
            self, request: SingleGenerationRequest,
            context: ServicerContext) -> AsyncIterator[GenerationResponse]:
        request_id = self.request_id(context)
        sampling_params, deadline = await self._validate_and_convert_params(
            request.params, context)
        truncate_input_tokens = with_default(
            request.params.truncate_input_tokens, None)

        input_ids, max_is_token_limit = await self._validate_prompt_and_tokenize(
            sampling_params, truncate_input_tokens, request.request.text,
            context)

        result_generator = self.engine.generate(
            prompt=None,
            sampling_params=sampling_params,
            request_id=request_id,
            prompt_token_ids=input_ids,
        )

        resp_options = request.params.response

        first = True
        last_output_length = 0
        last_token_count = 0
        time_limit_reached = False
        #TODO handle cancellation
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
            yield self._convert_output(output, resp_options,
                                       max_is_token_limit, time_limit_reached,
                                       last_output_length, last_token_count)
            if time_limit_reached:
                break

            last_output_length = len(output.text)
            last_token_count = len(output.token_ids)

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

        resp_options = params.response
        sampling = params.sampling
        stopping = params.stopping
        greedy = params.method == DecodingMethod.GREEDY

        try:
            if params.decoding.HasField("length_penalty"):
                raise ValueError(
                    "decoding.length_penalty parameter not yet supported")
            if resp_options.token_ranks:
                raise ValueError(
                    "response.token_ranks option not yet supported")

            # default max may be limited further in later processing
            max_new_tokens: Optional[int] = None
            if stopping.max_new_tokens > 0:
                max_new_tokens = stopping.max_new_tokens
                if max_new_tokens > self.max_max_new_tokens:
                    raise ValueError(f"max_new_tokens ({max_new_tokens}) "
                                     f"must be <= {self.max_max_new_tokens}")

            min_new_tokens = -1
            if stopping.min_new_tokens > 0:
                min_new_tokens = stopping.min_new_tokens
                if max_new_tokens is not None:
                    if min_new_tokens > max_new_tokens:
                        raise ValueError(
                            f"min_new_tokens ({min_new_tokens}) "
                            f"must be <= max_new_tokens ({max_new_tokens})")
                elif min_new_tokens > self.max_max_new_tokens:
                    raise ValueError(f"min_new_tokens ({min_new_tokens}) "
                                     f"must be <= {self.max_max_new_tokens}")

            if stopping.stop_sequences and len(stopping.stop_sequences) > MAX_STOP_SEQS or \
                    not all(0 < len(ss) <= MAX_STOP_SEQ_LENGTH for ss in stopping.stop_sequences):
                raise ValueError(
                    f"can specify at most {MAX_STOP_SEQS} non-empty stop sequences, "
                    f"each not more than {MAX_STOP_SEQ_LENGTH} UTF8 bytes")

            # TODO more parameter validation

            logprobs = 1 if resp_options.token_logprobs else 0
            top_n_tokens = resp_options.top_n_tokens
            if top_n_tokens:
                if top_n_tokens > MAX_TOP_N_TOKENS:
                    raise ValueError(f"top_n_tokens ({top_n_tokens}) "
                                     f"must be <= {MAX_TOP_N_TOKENS}")

                # vLLM will currently return logprobs for n+1 tokens
                # (selected token plus top_n excluding selected)
                logprobs += top_n_tokens
                if greedy and resp_options.token_logprobs:
                    logprobs -= 1

            logprobs = with_default(logprobs, None)

            # GAPS:
            # - exp_decay_length_penalty
            # - return ranks

            # NEW FUNCTION TO ADD (later)
            # - presence penalty, freq penalty
            # - min_p
            # - beam search (with length_penalty, stop_early, n)

            # TBD (investigate more)
            # - best_of / n
            # - spaces_between_special_tokens
            # - skip_special_tokens (per request)
            # - stop_token_ids

            # use logits processors to extend the sampling methods
            logits_processors: List[LogitsProcessor] = []
            if min_new_tokens > 0:
                min_tokens_processor = MinTokensLogitsProcessor(
                    min_tokens=min_new_tokens,
                    # TODO: will eos_tokens_ids need to be adjusted to use the LoRA tokenizer?
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                logits_processors.append(min_tokens_processor)

            # to match TGIS, only including typical_p processing when using sampling
            if not greedy and 0.0 < sampling.typical_p < 1.0:
                logits_processors.append(
                    TypicalLogitsWarperWrapper(mass=sampling.typical_p))

            time_limit_millis = stopping.time_limit_millis
            deadline = time.time(
            ) + time_limit_millis / 1000.0 if time_limit_millis > 0 else None

            sampling_params = SamplingParams(
                logprobs=logprobs,
                prompt_logprobs=logprobs if resp_options.input_tokens else None,
                max_tokens=max_new_tokens,
                min_tokens=min_new_tokens,
                temperature=sampling.temperature if not greedy else 0.0,
                top_k=with_default(sampling.top_k, -1),
                top_p=with_default(sampling.top_p, 1.0),
                seed=sampling.seed if sampling.HasField("seed") else None,
                repetition_penalty=with_default(params.decoding.repetition_penalty, 1.0),
                logits_processors=logits_processors,
                stop=with_default(stopping.stop_sequences, None),
                include_stop_str_in_output=stopping.include_stop_sequence \
                   if stopping.HasField("include_stop_sequence") else self.default_include_stop_seqs,
                skip_special_tokens=self.skip_special_tokens,
            )
        except ValueError as e:
            #TODO run TGIS param validation here to match TGIS error messages
            await context.abort(StatusCode.INVALID_ARGUMENT, str(e))

        return sampling_params, deadline

    @staticmethod
    def _convert_reason(output: CompletionOutput, max_is_token_limit: bool,
                        time_limit_reached: bool) -> Tuple['StopReason', str]:
        finish_reason = output.finish_reason
        stop_sequence = None
        if finish_reason is None:
            stop_reason = StopReason.TIME_LIMIT if time_limit_reached else StopReason.NOT_FINISHED
        elif finish_reason == "length":
            stop_reason = StopReason.TOKEN_LIMIT if max_is_token_limit else StopReason.MAX_TOKENS
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
        logprobs_list: Optional[list[Dict[int, float]]],
        include_logprobs: bool,
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
                if include_logprobs:
                    token_info.logprob = logprobs[token_ids[i]]
                if top_n_tokens:
                    items = sorted(logprobs.items(),
                                   key=lambda item: item[1],
                                   reverse=True)[:top_n_tokens]
                    #TODO later use get_lora_tokenizer here
                    tt_texts = self.tokenizer.convert_ids_to_tokens(
                        [tid for tid, _ in items])
                    token_info.top_tokens.extend(
                        TokenInfo.TopToken(
                            text=tt_text,
                            logprob=logprob,
                        ) for tt_text, (_, logprob) in zip(tt_texts, items))
            token_infos.append(token_info)

    async def _validate_prompt_and_tokenize(
        self,
        sampling_params: SamplingParams,
        truncate_input_tokens: Optional[int],
        prompt: Optional[str],
        context: ServicerContext,
    ) -> Tuple[List[int], bool]:
        tokenize_kwargs = {"truncation": True, "max_length": truncate_input_tokens} \
            if truncate_input_tokens is not None else {}

        max_model_len = self.config.max_model_len
        input_ids = await self.tokenizer_group.encode_async(
            prompt, **tokenize_kwargs)
        token_num = len(input_ids)

        if token_num >= max_model_len:
            await context.abort(
                StatusCode.INVALID_ARGUMENT,
                f"input tokens ({token_num}) must be < {max_model_len}")
        min_new_tokens = sampling_params.min_tokens
        if token_num + min_new_tokens > max_model_len:
            await context.abort(
                StatusCode.INVALID_ARGUMENT,
                f"input tokens ({token_num}) plus min_new_tokens ({min_new_tokens}) must be <= {max_model_len}"
            )

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

    @log_rpc_handler_errors
    async def Tokenize(self, request: BatchedTokenizeRequest,
                       context: ServicerContext) -> BatchedTokenizeResponse:
        responses: List[TokenizeResponse] = []

        #TODO maybe parallelize, also move convert_ids_to_tokens into the other threads
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
    #     generation_pb2.DESCRIPTOR.services_by_name["GenerationService"].full_name,
    #     reflection.SERVICE_NAME,
    # )
    # reflection.enable_server_reflection(SERVICE_NAMES, server)

    host = "0.0.0.0" if args.host is None else args.host
    listen_on = f"{host}:{args.grpc_port}"

    #TODO add TLS
    server.add_insecure_port(listen_on)
    await server.start()
    logger.info(f"gRPC Server started at {listen_on}")

    return server
