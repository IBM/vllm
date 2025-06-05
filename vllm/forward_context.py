# SPDX-License-Identifier: Apache-2.0

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union, List

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group,
                                          is_v1_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata

logger = init_logger(__name__)

track_batchsize: bool = envs.VLLM_LOG_BATCHSIZE_INTERVAL >= 0
last_logging_time: float = 0
forward_start_time: float = 0
batchsize_logging_interval: float = envs.VLLM_LOG_BATCHSIZE_INTERVAL
batchsize_forward_time: defaultdict = defaultdict(list)


@dataclass
class DPMetadata:
    cu_tokens_across_dp_cpu: torch.Tensor

@dataclass
class ALoRAMetadata:
    k_offsets: torch.Tensor
    query_start_locs: List[int]
   # num_reqs: int
    #mask: torch.Tensor 
    
def make_alora_mask(k_offsets,query_start_locs):
       
    #k_offsets = self.k_offsets
    #query_start_locs = self.query_start_locs
    #num_reqs = self.num_reqs

    # (C) Build the 1D “save‐prefix” mask:
    T = torch.max(query_start_locs)                                           # total rows
    #row_ids = torch.arange(T, device=output.device)              # [T]
    starts  = query_start_locs[:-1]                              # [N]
    ends    = query_start_locs[1:]                               # [N]
    lengths = ends - starts                                      # [N]
    kept_lens = lengths - k_offsets                              # [N]
    kept_lens  = torch.clamp(kept_lens, min=0)      # any negative → 0

    #ge       = row_ids.unsqueeze(0) >= starts.unsqueeze(1)       # [N×T]
    #lt       = row_ids.unsqueeze(0) < (starts.unsqueeze(1) + kept_lens.unsqueeze(1))  # [N×T]
    #cond2d   = ge & lt                                           # [N×T]
    #mask1d   = cond2d.any(dim=0)                                 # [T], dtype=bool
    device = query_start_locs.device
    delta = torch.zeros(T + 1, device=device, dtype=torch.bfloat16)
    starts_clamped = starts #torch.clamp(starts, min=0, max=T)
    ends_for_scatter = starts + kept_lens
   # ends_for_scatter = torch.clamp(ends_for_scatter, min=0, max=T)
    #ones = torch.ones_like(starts_clamped, dtype=output.dtype)  # [N], float
    #neg_ones = -ones
    pos_vals = kept_lens.sign().to(torch.bfloat16) #(kept_lens > 0).to(output.dtype)
    neg_vals = - pos_vals
    delta.scatter_add_(0, starts, pos_vals)       # delta[start_i] += +1
    #delta.clamp(min=0,max=1)
    delta.scatter_add_(0, ends_for_scatter, neg_vals)  # delta[end_i]   += -1
    #delta.clamp(min=-1,max=1)
    #delta[0] = 1
    # 6) Now take cumsum over delta[:-1] to get a “coverage count” per row:
    cums = torch.cumsum(delta[:-1], dim=0)  # shape [T]; dtype float
    # Wherever cums[r] > 0, that row was in at least one interval.
#     print(query_start_locs[:num_reqs+1])


    mask1d = cums > 0                       # shape [T], bool
    #mask2d = mask1d.unsqueeze(1).to(torch.bfloat16)
       # (D) Save original prefix rows:
    #self.mask = mask2d
    return mask1d
@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[str, Any]
    """
    Type AttentionMetadata for v0, 
    Type Dict[str, AttentionMetadata] for v1, map from layer_name of each 
    attention layer to its attention metadata
    set dynamically for each forward pass
    """
    attn_metadata: Union["AttentionMetadata", dict[str, "AttentionMetadata"]]
    # TODO: remove after making all virtual_engines share the same kv cache
    virtual_engine: int  # set dynamically for each forward pass
    # set dynamically for each forward pass
    dp_metadata: Optional[DPMetadata] = None
    alora_metadata: Optional[ALoRAMetadata] = None


_forward_context: Optional[ForwardContext] = None


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context.")
    return _forward_context


@contextmanager
def set_forward_context(attn_metadata: Any,
                        vllm_config: VllmConfig,
                        virtual_engine: int = 0,
                        num_tokens: int = 0,
                        alora_metadata: Optional[ALoRAMetadata] = None):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    global forward_start_time
    need_to_track_batchsize = track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        forward_start_time = time.perf_counter()
    dp_metadata: Optional[DPMetadata] = None
    if vllm_config.parallel_config.data_parallel_size > 1:
        dp_size = vllm_config.parallel_config.data_parallel_size
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        if attn_metadata is not None and hasattr(attn_metadata,
                                                 "num_prefill_tokens"):
            # for v0 attention backends
            batchsize = attn_metadata.num_prefill_tokens + \
                attn_metadata.num_decode_tokens
        else:
            # for v1 attention backends or no attn_metadata
            batchsize = num_tokens
        num_tokens_across_dp = [0] * dp_size
        num_tokens_across_dp[dp_rank] = batchsize
        num_tokens_tensor = torch.tensor(num_tokens_across_dp,
                                         device="cpu",
                                         dtype=torch.int32)
        from vllm.distributed.parallel_state import get_dp_group
        dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
        cu_tokens_across_dp_cpu = torch.cumsum(num_tokens_tensor, dim=0)
        dp_metadata = DPMetadata(cu_tokens_across_dp_cpu)

    global _forward_context
    prev_context = _forward_context
    _forward_context = ForwardContext(
        no_compile_layers=vllm_config.compilation_config.
        static_forward_context,
        virtual_engine=virtual_engine,
        attn_metadata=attn_metadata,
        dp_metadata=dp_metadata,
        alora_metadata=alora_metadata,
        )

    # KVConnector: trigger (possibly async) load before forward.
    # Each attn layer will block until the reading is complete.
    trigger_kv_transfer = (attn_metadata is not None
                           and has_kv_transfer_group()
                           and is_v1_kv_transfer_group())
    if trigger_kv_transfer:
        kv_connector = get_kv_transfer_group()
        assert isinstance(kv_connector, KVConnectorBase_V1)
        kv_connector.start_load_kv(_forward_context)

    try:
        yield
    finally:
        global last_logging_time, batchsize_logging_interval
        if need_to_track_batchsize:
            if hasattr(attn_metadata, "num_prefill_tokens"):
                # for v0 attention backends
                batchsize = attn_metadata.num_prefill_tokens + \
                    attn_metadata.num_decode_tokens
            else:
                # for v1 attention backends
                batchsize = num_tokens
            # we use synchronous scheduling right now,
            # adding a sync point here should not affect
            # scheduling of the next batch
            torch.cuda.synchronize()
            now = time.perf_counter()
            # time measurement is in milliseconds
            batchsize_forward_time[batchsize].append(
                (now - forward_start_time) * 1000)
            if now - last_logging_time > batchsize_logging_interval:
                last_logging_time = now
                forward_stats = []
                for bs, times in batchsize_forward_time.items():
                    if len(times) <= 1:
                        # can be cudagraph / profiling run
                        continue
                    medium = torch.quantile(torch.tensor(times), q=0.5).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    logger.info(("Batchsize forward time stats "
                                 "(batchsize, count, median_time(ms)): %s"),
                                forward_stats)

        # KVConnector: each attn layer triggers (possibly async) save.
        # Ensure all those operations complete before forward() is done.
        if trigger_kv_transfer:
            kv_connector = get_kv_transfer_group()
            assert isinstance(kv_connector, KVConnectorBase_V1)
            kv_connector.wait_for_save()

        _forward_context = prev_context
