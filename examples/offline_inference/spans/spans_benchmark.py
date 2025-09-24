# SPDX-License-Identifier: Apache-2.0
import os
import time
import random

# necessary for spans to work
os.environ["VLLM_USE_V1"] = "1"
# to ensure deterministic behaviour
os.environ["TOKENIZERS_PARALLELISM"] = "False"

# in case you need it
os.environ['VLLM_ATTENTION_BACKEND'] = "TRITON_ATTN_VLLM_V1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = '0'

# standard imports
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


# helper functions
def pad(toklist):
    padtok = int(os.environ.get("VLLM_V1_SPANS_TOKEN_PAD", None))
    return toklist[:-1] + [padtok] * ((16 - len(toklist)) % 16) + toklist[-1:]


def avg(list_of_numbers):
    return sum(list_of_numbers) / max(len(list_of_numbers), 1)


def wrap(prompt):
    if isinstance(prompt[0], list):
        return [TokensPrompt(prompt_token_ids=p) for p in prompt]
    return TokensPrompt(prompt_token_ids=prompt)

def initialize_vllm(model,
                    temp=0.6,
                    logprobs=None,
                    max_toks=131072,
                    max_generated_toks=1):
    # boot up vLLM
    samp_params_preload = SamplingParams(temperature=temp, max_tokens=1)
    samp_params_generate = SamplingParams(temperature=temp,
                                          max_tokens=max_generated_toks,
                                          logprobs=logprobs)
    llm = LLM(
        model=model,
        gpu_memory_utilization=0.9,
        enforce_eager=True,  # <- so it boots faster
        block_size=16,
        max_model_len=max_toks,
        max_num_seqs=4,
    )
    tok = llm.get_tokenizer()
    tok_fun = lambda x: tok.convert_tokens_to_ids(tok.tokenize(x))
    return samp_params_preload, samp_params_generate, tok_fun, llm


def main():
    model_names = [
        "ldsjmdy/Tulu3-Block-FT",  # <- finetuned to handle block-attention
        "ldsjmdy/Tulu3-RAG",  #      <- baseline
    ]
    model_name = model_names[0]

    # tokens that need to be set to perform block-attention
    PAD_TOK = 27  # <-  "<"
    SPAN_TOK = 10  # <- "+"
    SPAN_RECOMP_TOK = 31  # <- "@"

    # vLLM-specific env vars

    # enables block attention
    # -> when this line is not commented, we expect a speedup
    #    in the execution of the last two .generate calls
    os.environ['VLLM_V1_SPANS_ENABLED'] = 'True'

    # the token that tells vLLM "this is the beginning of a span"
    os.environ['VLLM_V1_SPANS_TOKEN_PLUS'] = str(SPAN_TOK)

    # token that tells vLLM:
    # "from here on, recompute KV vectors if any previous tokens differ"
    os.environ['VLLM_V1_SPANS_TOKEN_CROSS'] = str(SPAN_RECOMP_TOK)

    # will print every step of the span process if set to true
    # os.environ['VLLM_V1_SPANS_DEBUG'] = 'True'

    # will disable the adjustment of positional encodings when a KV cache
    # block is loaded to a different position than it was stored
    # -> when this line is not commented,
    #    spans overlap in their positional encodings
    os.environ['VLLM_V1_SPANS_DISABLE_REPOSITION'] = 'True'

    # general env vars

    # our helper function uses this token to pad spans
    os.environ['VLLM_V1_SPANS_TOKEN_PAD'] = str(PAD_TOK)

    # now we instantiate the model
    samp_params_preload, samp_params_generate, tok, llm = initialize_vllm(
        model_name, max_generated_toks=1)
        # model_name, max_generated_toks=1, max_toks=2048)

    # components of the prompt template
    prefix = pad(
        [SPAN_RECOMP_TOK] + tok("<|system|>\nYou are an intelligent AI assistant. " \
            "Please answer questions based on the user's instructions. " \
            "Below are some reference documents that may help you in " \
            "answering the user's question."
            ))
    midfx = [SPAN_RECOMP_TOK] + tok(
        "<|user|>\nPlease write a high-quality answer for the " \
        "given question using only the provided search documents " \
        "(some of which might be irrelevant).\nQuestion: "
    )
    postfx = tok('''\n<|assistant|>\n''')

    print("---->", postfx)

    times = []
    for ndocs in [1, 2, 4, 8]:
        for dlen in [512, 1024, 2048, 4096, 8192]:
            print(f"<!> DOCLENGTH {dlen} NUMDOCS {ndocs}")

            doc_toks = tok(
                "Sequence Transduction Models and Template-Assisted Selective Epitaxy")
            docs = [pad([SPAN_TOK] +
                        random.choices(doc_toks, k=dlen))
                        for _ in range(ndocs)]

            # user query
            query = midfx + tok(
                "Tell me which one concerns deep learning. " \
                    "Indicate your answer with a number in brackets."
            ) + postfx

            for i in range(3):
                print(f"<!> ITERATION {i}")

                # preload documents
                ts_pre = time.time()
                llm.generate(
                    [wrap(d) for d in docs] + [wrap(prefix)],
                    sampling_params=samp_params_preload, use_tqdm=False)
                te_pre = time.time() - ts_pre

                ts_gen = time.time()

                # this now will load prefix, doc_a, doc_b,
                # from the KV cache regardless of the order
                random.shuffle(docs)
                llm.generate(wrap(prefix + \
                        sum(docs, []) + \
                        query),
                    sampling_params=samp_params_generate, use_tqdm=False)

                # this should also run faster:
                random.shuffle(docs)
                llm.generate(wrap(prefix + \
                        sum(docs, []) + \
                        query),
                    sampling_params=samp_params_generate, use_tqdm=False)

                te_gen = time.time() - ts_gen

                print(f"doc preload time / TTFT : {te_pre:.4f} / {te_gen:.4f} (s)")
                times.append(dict(
                    preload_time=te_pre,
                    gen_time=te_gen,
                    it=i,
                    doc_len=dlen,
                    num_docs=ndocs,
                ))


if __name__ == '__main__':
    main()
