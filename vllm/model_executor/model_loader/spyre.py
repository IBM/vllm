"""Utilities for selecting and loading Spyre models."""
import sys
from typing import Optional

import torch
import torch._inductor.config
import torch.distributed as dist
import torch.nn as nn
from fms.models import get_model
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm.config import ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata

try:
    from torch_sendnn import torch_sendnn  # noqa: F401
except ImportError:
    print("WARNING: Disabled: torch_sendnn")
    pass
try:
    import backends.dynamo_tracer  # noqa: F401
except ImportError:
    print("WARNING: Disabled: dynamo_tracer")
    pass

BACKEND_LIST = ['sendnn_decoder', 'inductor']

logger = init_logger(__name__)


class SpyreCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                logits_as_input=True)
        self.sampler = Sampler()
        self.past_key_value_states = None
        self.dtype = torch.float16 if envs.VLLM_SPYRE_DYNAMO_BACKEND == \
            'sendnn_decoder' else torch.float32
        # number of added padding sequences to fill
        # batch to warmed up batch size
        self.num_padded_sequences = 0

        # Lazy initialized
        self.model: nn.Module

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        masks: torch.Tensor,
        is_prompt: bool,
    ) -> torch.Tensor:

        if is_prompt:
            self.past_key_value_states = None

        extra_kwargs = {}
        if envs.VLLM_SPYRE_DYNAMO_BACKEND != "sendnn_decoder":
            # Bug in 2.3.1 fixed in 2.4.1 for SDPA flash
            # cpu impl when padding too much
            extra_kwargs["attn_algorithm"] = "math"

        output = self.model(
            input_ids,
            position_ids=positions,
            mask=masks,
            past_key_value_states=self.past_key_value_states,
            use_cache=True,
            only_last_token=True,
            **extra_kwargs,
        )

        logits, past_key_value_states = output
        self.past_key_value_states = past_key_value_states

        # mark dynamic
        if self.past_key_value_states is not None:
            for layer in self.past_key_value_states:
                for tensor in layer:
                    torch._dynamo.mark_dynamic(tensor, 2)

        # removing batch padding sequences to compute logits
        batch_size = input_ids.shape[0]

        logits = logits[:batch_size - self.num_padded_sequences]

        return logits

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, model_config: ModelConfig, max_prompt_length: int,
                     max_decode_length: int,
                     distributed_strategy: Optional[str], **kwargs):

        if self.dtype is not model_config.dtype:
            logger.info(
                "Ignoring user-provided dtype=%s and using dtype=%s instead.",
                model_config.dtype, self.dtype)

        if model_config.quantization == "gptq":

            # note, we have to find a better way to package this
            # shouldn't it be part of FMS?
            sys.path.append("/home/senuser/aiu-fms")

            if envs.VLLM_SPYRE_DYNAMO_BACKEND == "sendnn_decoder":
                from aiu_as_addon import aiu_adapter, aiu_linear  # noqa: F401
                linear_type = "gptq_aiu"
                print("Loaded `aiu_as_addon` functionalities")
            else:
                from cpu_addon import cpu_linear  # noqa: F401
                linear_type = "gptq_cpu"
                print("Loaded `cpu_addon` functionalities")

            quant_cfg = model_config._parse_quant_hf_config()

            linear_config = {
                "linear_type": linear_type,
                "group_size": quant_cfg['group_size'],
                "desc_act": quant_cfg['desc_act'],
            }
            data_type = None
            model_source = "hf_gptq_aiu"
        else:
            linear_config = {"linear_type": "torch_linear"}
            data_type = self.dtype
            model_source = "hf"

        # we can use fused weights unless running on Spyre
        fused_weights = envs.VLLM_SPYRE_DYNAMO_BACKEND != "sendnn_decoder"

        self.model = get_model(architecture="hf_configured",
                               variant=model_config.model,
                               model_path=model_config.model,
                               source=model_source,
                               data_type=data_type,
                               distributed_strategy=distributed_strategy,
                               group=dist.group.WORLD,
                               fused_weights=fused_weights,
                               linear_config=linear_config)

        compile_mode = "default"

        self.model.eval()
        torch.set_grad_enabled(False)

        _target_cache_size = max(int(max_decode_length * 2),
                                 int(max_prompt_length * 2.5))
        if hasattr(torch._dynamo.config, "accumulated_cache_size_limit") and \
            _target_cache_size > torch._dynamo.config.\
            accumulated_cache_size_limit:
            _prev = torch._dynamo.config.accumulated_cache_size_limit
            torch._dynamo.config.accumulated_cache_size_limit = \
                _target_cache_size
            print("NOTICE: Adjusting "
                  "torch._dynamo.config.accumulated_cache_size_limit"
                  f" from {_prev} to "
                  f"{torch._dynamo.config.accumulated_cache_size_limit} "
                  f"to accommodate prompt size of {max_prompt_length} "
                  f"and decode tokens of {max_decode_length}")

        if _target_cache_size > torch._dynamo.config.cache_size_limit:
            _prev = torch._dynamo.config.cache_size_limit
            torch._dynamo.config.cache_size_limit = _target_cache_size
            print(
                "NOTICE: Adjusting torch._dynamo.config.cache_size_limit from"
                f" {_prev} to {torch._dynamo.config.cache_size_limit} to "
                f"accommodate prompt size of {max_prompt_length} and "
                f"decode tokens of {max_decode_length}")

        if envs.VLLM_SPYRE_DYNAMO_BACKEND in BACKEND_LIST:
            self.model = torch.compile(self.model,
                                       mode=compile_mode,
                                       backend=envs.VLLM_SPYRE_DYNAMO_BACKEND)


def get_spyre_model(model_config: ModelConfig, parallel_config: ParallelConfig,
                    max_prompt_length, max_decode_length) -> nn.Module:

    # Create a model instance.
    model = SpyreCausalLM(model_config.hf_config)

    # Load the weights from the cached or downloaded files.
    model.load_weights(
        model_config,
        max_prompt_length=max_prompt_length,
        max_decode_length=max_decode_length,
        distributed_strategy="tp" if parallel_config.world_size > 1 else None)

    return model
