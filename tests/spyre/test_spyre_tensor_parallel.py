"""Verification of vLLM output by comparing with HF

Run `pytest tests/spyre/test_spyre_tensor_parallel.py`.
"""

from typing import List, Tuple

import pytest
from spyre_util import (compare_results, generate_hf_output,
                        generate_spyre_vllm_output)

from vllm import SamplingParams


@pytest.mark.skip("Skip until failure is resolved.")
@pytest.mark.parametrize("model", ["/models/llama-194m"])
@pytest.mark.parametrize("prompts", [[
    "Provide a list of instructions for preparing"
    " chicken soup for a family of four.", "Hello",
    "What is the weather today like?", "Who are you?"
]])
@pytest.mark.parametrize("warmup_shapes", [[(64, 20, 4)]]
                         )  #,[(64,20,8)],[(128,20,4)],[(128,20,8)]])
# (prompt_length/new_tokens/batch_size)
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("backend",
                         ["eager"])  #, "inductor", "sendnn_decoder"])
def test_output(
    model: str,
    prompts: List[str],
    warmup_shapes: List[Tuple[int, int, int]],
    tp_size: int,
    backend: str,
) -> None:
    '''
    The warmup is based on one or multiple shapes. After the warmup,
    one request with the provided prompts is input to vLLM which
    is executed in tensor-parallel fashion on <tp_size> Spyres.
    The same prompts are also input to HF. The generated output
    including text, token ids, and logprobs, is verified to be
    identical for vLLM and HF.

    If errors occur, these can be analyzed/debugged by setting
    'DISABLE_ASSERTS = True' in spyre_util.py and by rerunning the
    test using 'pytest --capture=no tests/spyre/test_spyre_tensore_parallel.py'
    After debugging, DISABLE_ASSERTS should be reset to 'False'.
    '''

    max_new_tokens = max([t[1] for t in warmup_shapes])

    vllm_sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=True)

    vllm_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        warmup_shapes=warmup_shapes,
        max_model_len=2048,
        block_size=2048,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=tp_size,
        backend=backend)

    hf_results = generate_hf_output(model=model,
                                    prompts=prompts,
                                    max_new_tokens=max_new_tokens)

    compare_results(model=model,
                    prompts=prompts,
                    warmup_shapes=warmup_shapes,
                    tensor_parallel_size=tp_size,
                    backend=backend,
                    vllm_results=vllm_results,
                    hf_results=hf_results)
