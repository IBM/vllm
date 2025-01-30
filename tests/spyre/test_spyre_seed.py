"""Verification of seeded random sampling to be deterministic

Run `python -m pytest tests/spyre/test_spyre_seed.py`.
"""

import math
from typing import Tuple

import pytest
from spyre_util import generate_spyre_vllm_output

from vllm import SamplingParams
import os

# get model directory path from env, if not set then default to "/models". 
model_dir_path = os.environ.get("SPYRE_TEST_MODEL_DIR", "/models")
# get model backend from env, if not set then default to "eager" 
# For multiple values, export SPYRE_TEST_MODEL_DIR="eager,inductor"
backend_type = os.environ.get("SYPRE_TEST_BACKEND_TYPE", "eager")
# get model names from env, if not set then default to "llama-194m" 
# For multiple values, export SPYRE_TEST_MODEL_DIR="llama-194m,all-roberta-large-v1"
user_test_model_list = os.environ.get("SPYRE_TEST_MODEL_LIST","llama-194m")
test_model_list, test_backend_list = [],[]

for model in user_test_model_list.split(','):
    test_model_list.append(f"{model_dir_path}/{model.strip()}")

for backend in backend_type.split(','):
    test_backend_list.append(backend.strip())

@pytest.mark.parametrize("model", test_model_list)
@pytest.mark.parametrize("prompt", [
    "Provide a list of instructions for preparing"
    " chicken soup for a family of four."
])
@pytest.mark.parametrize("temperature", [0.1, 1.0])
@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("warmup_shape", [(64, 20, 4), (64, 20, 8),
                                          (128, 20, 4), (128, 20, 8)]
                         )  # (prompt_length/new_tokens/batch_size)
@pytest.mark.parametrize("backend",
                         test_backend_list)  #, "inductor", "sendnn_decoder"])
def test_seed(
    model: str,
    prompt: str,
    temperature: float,
    seed: int,
    warmup_shape: Tuple[int, int, int],
    backend: str,
) -> None:
    '''
    The warmup is based on a single shape. After the warmup,
    output is generated for one request with 16 identical prompts
    using random sampling (non-zero temperature) in combination
    with a seed. The generated output, including text, token ids,
    logprobs is verified to be identical for all 16 sequences.
    '''

    max_new_tokens = warmup_shape[1]

    prompts = [prompt] * 16

    vllm_sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=True,
        seed=seed)

    vllm_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        warmup_shapes=[warmup_shape],
        max_model_len=2048,
        block_size=2048,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend)

    # compare all generated outputs against the first generated output
    for vllm_result in vllm_results:
        assert vllm_result['text'] == vllm_results[0]['text']

        # compare logprobs for all tokens between
        # the current and the first sequence
        assert len(vllm_result['logprobs']) == len(vllm_results[0]['logprobs'])
        for token_id, logprob, token_id_0, logprob_0 in zip(
                vllm_result['token_ids'], vllm_result['logprobs'],
                vllm_results[0]['token_ids'], vllm_results[0]['logprobs']):
            assert token_id == token_id_0
            assert math.isclose(logprob, logprob_0, rel_tol=0.1)
