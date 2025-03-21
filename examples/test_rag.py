
import os
import torch
import json
import time
import numpy as np
import copy

BLOCK_SIZE = 16
N_OUTPUT_TOKENS = 100
USE_ALORA = False
BATCH_SIZE = 1
VERIFY=False

os.environ['VLLM_USE_V1'] = "1"


os.environ['VLLM_V1_USE_ACTIVATED_LORA'] = "1" if USE_ALORA else "0"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.distributed import cleanup_dist_env_and_memory
import math

model_path="ibm-granite/granite-3.1-8b-instruct"

if USE_ALORA:
    lora_path = "/home/zrltpa/vllm/examples/feb6_8bsft_alora_sz32_certainty/"
else:
    lora_path = "/home/zrltpa/vllm/examples/feb6_8bsft_standard_lora_sz6_certainty/"

llm = LLM(
    model=model_path,
    enable_lora=True,
    enforce_eager=True,
    max_lora_rank=32,
    block_size=BLOCK_SIZE,
    dtype=torch.float16,
    max_num_seqs=1 if VERIFY else BATCH_SIZE,
)


tokenizer = llm.get_tokenizer()

with open('control1024N_mar3_55baseshort.jsonl') as f:
    data = [json.loads(line) for line in f]


data = data[:BATCH_SIZE]

prompts = []
sampling_params = []

for idx in range(len(data)):

    input_text = tokenizer.apply_chat_template(
        conversation=data[idx]["messages"][:-1],
        documents=data[idx]["documents"],
        tokenize=False,
        add_generation_prompt=True
    )


    input_tokens = tokenizer(input_text)['input_ids']

    targ_len = len(input_tokens) + N_OUTPUT_TOKENS

    # leave one token at end of block for 1st activation token
    if targ_len % BLOCK_SIZE > 0:
        targ_len_align = BLOCK_SIZE * math.ceil(targ_len / float(BLOCK_SIZE)) - 1
    else:
        targ_len_align = targ_len + BLOCK_SIZE - 1


    print("len(input_tokens): %d, targ_len: %d, targ_len_align: %d" % (len(input_tokens), targ_len, targ_len_align))

    n_tokens_to_gen = targ_len_align - len(input_tokens)

    print("n_tokens_to_gen: ", n_tokens_to_gen)

    prompts.append({"prompt_token_ids": input_tokens})
    sampling_params.append(
        SamplingParams(temperature=0, min_tokens=n_tokens_to_gen, max_tokens=n_tokens_to_gen, ignore_eos=True)
    )


t0 = time.time()
outputs = llm.generate(prompts, sampling_params)
t_gen = time.time()-t0

outputs_verify = copy.deepcopy(outputs)

uq_prompt = "<|start_of_role|>certainty<|end_of_role|>"
uq_tokens = tokenizer(uq_prompt)["input_ids"]


if USE_ALORA:

    prompts = []
    sampling_params = []
    for idx in range(len(data)):

        input_tokens = outputs[idx].prompt_token_ids
        input_tokens += outputs[idx].outputs[0].token_ids
        input_tokens += uq_tokens[:1]
        prompts.append({"prompt_token_ids": input_tokens})
        sampling_params.append(
            SamplingParams(temperature=0, max_tokens=1),
        )

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    t_elap = time.time()-t0

    prompts = []
    sampling_params = []
    for idx in range(len(data)):

        input_tokens = outputs[idx].prompt_token_ids
        input_tokens += uq_tokens[1:]

        print(input_tokens)

        prompts.append({"prompt_token_ids": input_tokens})
        sampling_params.append(
            SamplingParams(temperature=0, max_tokens=6),
        )

else:

    prompts = []
    sampling_params = []
    for idx in range(len(data)):

        input_tokens = outputs[idx].prompt_token_ids
        input_tokens += outputs[idx].outputs[0].token_ids
        input_tokens += uq_tokens
        prompts.append({"prompt_token_ids": input_tokens})
        sampling_params.append(
            SamplingParams(temperature=0, max_tokens=6),
        )

    t_elap = 0.0

t0 = time.time()
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("tpa-test", 1, lora_path),
)
t_elap += time.time()-t0

print("Generate time: %.2f seconds" % (t_gen))
print(      "UQ time: %.2f seconds" % (t_elap))


for idx in range(len(data)):
    print(idx, repr(outputs[idx].outputs[0].text))


del llm
cleanup_dist_env_and_memory()

n_mismatch = 0

if VERIFY and USE_ALORA:

    import torch,os, copy
    from transformers import AutoTokenizer,  AutoModelForCausalLM, DynamicCache
    from alora_intrinsics.alora.peft_model_alora import PeftModelForCausalLM
    from alora_intrinsics.alora.config import aLoraConfig
    from alora_intrinsics.alora.tokenize_alora import tokenize_alora
    token = os.getenv("HF_MISTRAL_TOKEN")
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side='left',trust_remote_code=True, token=token)
    model_base = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto")

    model_alora = PeftModelForCausalLM.from_pretrained(model_base, lora_path, adapter_name="certainty", response_token_ids = None)
    model_alora.set_adapter("certainty")

    for idx in range(len(data)):
        input_text = tokenizer.decode(outputs_verify[idx].prompt_token_ids + outputs_verify[idx].outputs[0].token_ids)
        inputs = tokenizer(input_text, return_tensors="pt")
        prompt_cache = DynamicCache()
        with model_alora.disable_adapter():
            with torch.no_grad():
                prompt_cache = model_alora(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), past_key_values=prompt_cache).past_key_values 
        input_uq, alora_offsets = tokenize_alora(tokenizer,input_text, uq_prompt)
        past_key_values = copy.deepcopy(prompt_cache)
        output = model_alora.generate(input_uq["input_ids"].to(device), attention_mask=input_uq["attention_mask"].to(device), use_cache=True, max_new_tokens=6, return_dict_in_generate=True, past_key_values = past_key_values, alora_offsets = alora_offsets, output_scores=True)
        output_text = tokenizer.decode(output.sequences[0])
        print(idx, "Certainty: " + repr(output_text.split("certainty<|end_of_role|>")[-1]))


        n_to_compare = len(outputs[idx].outputs[0].token_ids)
        ref = output.sequences[0][-n_to_compare:].tolist()
        print(ref, outputs[idx].outputs[0].token_ids)

        if ref != outputs[idx].outputs[0].token_ids:
            n_mismatch += 1


    print("n_mismatch: ", n_mismatch)

