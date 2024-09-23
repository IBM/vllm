"""Compare the outputs of HF and vLLM for Mistral models using greedy sampling.

Run `pytest tests/models/test_llama_embedding.py`.
"""
import pytest
import torch
import torch.nn.functional as F

from vllm.inputs import build_decoder_prompts

MODELS = [
    {
        "name": "intfloat/e5-mistral-7b-instruct",
        "is_decoder_only": True
    },
    {
        "name": "bert-base-uncased",
        "is_decoder_only": False,
        "max_model_len": 512
    },
]


def compare_embeddings(embeddings1, embeddings2):
    similarities = [
        F.cosine_similarity(torch.tensor(e1), torch.tensor(e2), dim=0)
        for e1, e2 in zip(embeddings1, embeddings2)
    ]
    return similarities


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: dict,
    dtype: str,
) -> None:
    model_name = model["name"]
    is_decoder_only = model["is_decoder_only"]
    max_model_len = model.get("max_model_len", 1024)
    with hf_runner(model_name, dtype=dtype,
                   is_embedding_model=True) as hf_model:
        hf_outputs = hf_model.encode(example_prompts)

    with vllm_runner(
            model_name,
            dtype=dtype,
            disable_sliding_window=True,
            enforce_eager=True,
            # NOTE: Uncomment this line if runs out of GPU memory.
            # gpu_memory_utilization=0.95,
            max_model_len=max_model_len,
    ) as vllm_model:
        prompt_inputs = build_decoder_prompts(
            example_prompts) if is_decoder_only else example_prompts
        vllm_outputs = vllm_model.encode(prompt_inputs)

    similarities = compare_embeddings(hf_outputs, vllm_outputs)
    all_similarities = torch.stack(similarities)
    tolerance = 1e-2
    assert torch.all((all_similarities <= 1.0 + tolerance)
                     & (all_similarities >= 1.0 - tolerance)
                     ), f"Not all values are within {tolerance} of 1.0"
