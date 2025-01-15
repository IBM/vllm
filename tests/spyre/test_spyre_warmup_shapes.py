"""Verification of Spyre warmup shapes

Run `pytest tests/spyre/test_spyre_warmup_shapes.py`.
"""

from typing import List, Tuple

import pytest
from spyre_util import (compare_results, generate_hf_output,
                        generate_spyre_vllm_output)

from vllm import SamplingParams


@pytest.mark.parametrize("model", ["/models/llama-194m"])
@pytest.mark.parametrize("prompts", [
    7 * [
        "Hello",
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request. Be polite in your response to "
        "the user. Provide a list of instructions for preparing chicken soup"
        " for a family of four. Indicate if the weather forecast looks good "
        "for today. Explain in a brief summary comprised of at most 50 words"
        " what you are."
    ]
])
@pytest.mark.parametrize("warmup_shapes", [[(64, 20, 8), (128, 20, 4)]]
                         )  # (prompt_length/new_tokens/batch_size)
@pytest.mark.parametrize("backend",
                         ["eager"])  #, "inductor", "sendnn_decoder"])
def test_output(
    model: str,
    prompts: List[str],
    warmup_shapes: List[Tuple[int, int, int]],
    backend: str,
) -> None:
    '''
    The warmup is based on two shapes, that 'overlap' each
    other. After the warmup, one request with the provided
    prompts is input to vLLM. There should be at least one
    prompt corresponding to each of the two warmup shapes.
    It is useful to define enough prompts to fill multiple
    batches entirely and partially, in order to test the
    handling of overlapping warmup shapes also in relation
    with the position of a prompt within a batch (not
    likely that this will be an issue, but just to be sure).
    The same prompts are also input to HF. The generated
    output including text, token ids, and logprobs, is
    verified to be identical for vLLM and HF.

    If errors occur, these can be analyzed/debugged by setting
    'DISABLE_ASSERTS = True' in spyre_util.py and by rerunning the
    test using 'pytest --capture=no tests/spyre/test_spyre_warmup_shapes.py'
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
        tensor_parallel_size=1,
        backend=backend)

    hf_results = generate_hf_output(model=model,
                                    prompts=prompts,
                                    max_new_tokens=max_new_tokens)

    compare_results(model=model,
                    prompts=prompts,
                    warmup_shapes=warmup_shapes,
                    tensor_parallel_size=1,
                    backend=backend,
                    vllm_results=vllm_results,
                    hf_results=hf_results)