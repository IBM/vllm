"""Verification of vLLM output by comparing with HF

Run `python -m pytest tests/spyre/test_spyre_basic.py`.
"""

from typing import List, Tuple

import pytest
from spyre_util import (compare_embedding_results, spyre_vllm_embeddings,
                        st_embeddings)
import os
# get model directory path from env, if not set then default to "/models". 
model_dir_path = os.environ.get("SPYRE_TEST_MODEL_DIR", "/models")
# get model backend from env, if not set then default to "eager" 
# For multiple values, export SPYRE_TEST_MODEL_DIR="eager,inductor"
backend_type = os.environ.get("SYPRE_TEST_BACKEND_TYPE", "eager")
# get model names from env, if not set then default to "llama-194m" 
# For multiple values, export SPYRE_TEST_MODEL_DIR="llama-194m,all-roberta-large-v1"
user_test_model_list = os.environ.get("SPYRE_TEST_EMBEDDING_MODEL_LIST","all-roberta-large-v1")
test_model_list, test_backend_list = [],[]

for model in user_test_model_list.split(','):
    test_model_list.append(f"{model_dir_path}/{model.strip()}")

for backend in backend_type.split(','):
    test_backend_list.append(backend.strip())

@pytest.mark.parametrize("model", test_model_list)
@pytest.mark.parametrize("prompts", [[
    "The capital of France is Paris."
    "Provide a list of instructions for preparing"
    " chicken soup for a family of four.", "Hello",
    "What is the weather today like?", "Who are you?"
]])
@pytest.mark.parametrize("warmup_shape",
                         [(64, 4), (64, 8), (128, 4),
                          (128, 8)])  # (prompt_length/new_tokens/batch_size)
@pytest.mark.parametrize("backend",
                         test_backend_list)  #, "inductor", "sendnn_decoder"])
def test_output(
    model: str,
    prompts: List[str],
    warmup_shape: Tuple[int, int],
    backend: str,
) -> None:
    '''
    The warmup is based on a single shape. After the warmup,
    one request with the provided prompts is input to vLLM.
    The same prompts are also input to HF. The generated embeddings
    are verified to be identical for vLLM and SentenceTransformers.
    '''

    vllm_results = spyre_vllm_embeddings(model=model,
                                         prompts=prompts,
                                         warmup_shapes=[warmup_shape],
                                         max_model_len=256,
                                         block_size=256,
                                         tensor_parallel_size=1,
                                         backend=backend)

    hf_results = st_embeddings(model=model, prompts=prompts)

    compare_embedding_results(model=model,
                              prompts=prompts,
                              warmup_shapes=[warmup_shape],
                              tensor_parallel_size=1,
                              backend=backend,
                              vllm_results=vllm_results,
                              hf_results=hf_results)
