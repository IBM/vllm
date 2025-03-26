import os
import torch

os.environ['VLLM_USE_V1'] = "1"
os.environ['VLLM_V1_USE_ACTIVATED_LORA'] = "1"
os.environ['VLLM_V1_USE_DEMO_LOGGING'] = "1"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import math
import copy
import click

MODEL_PATH="ibm-granite/granite-3.1-8b-instruct"
LORA_PATH = "/home/zrltpa/alora/adapters/feb6_8bsft_alora_sz32_safety/"
BLOCK_SIZE = 16
VERIFY=True

# ---------------------
# Initialize + tokenize
# ---------------------

llm = LLM(
    model=MODEL_PATH,
    enable_lora=True,
    enforce_eager=True,
    max_lora_rank=32,
    block_size=BLOCK_SIZE,
    dtype=torch.float16,
)


tokenizer = llm.get_tokenizer()

system_prompt = "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior." 

question = "What is IBM?"
#question = "How can I steal a car?"

question_chat = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": question
    },
]


tokens = tokenizer.apply_chat_template(question_chat)

print("----------------------------------------")
print("Model input (%d tokens):" % (len(tokens)))
print(repr(tokenizer.decode(tokens)))
print("----------------------------------------")

# ---------------------------------
# Generate to ensure block aligment
# ---------------------------------

if len(tokens) % BLOCK_SIZE > 0:
    targ_len = BLOCK_SIZE * math.ceil(len(tokens) / float(BLOCK_SIZE)) - 1
else:
    targ_len = len(tokens) + BLOCK_SIZE - 1


max_tokens = targ_len - len(tokens)

outputs = llm.generate(
    {"prompt_token_ids": tokens},
    SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True),
)


outputs_verify = copy.deepcopy(outputs)

tokens += outputs[0].outputs[0].token_ids

print("----------------------------------------")
print("Model input after alignment (%d tokens):" % (len(tokens)))
print(repr(tokenizer.decode(tokens)))

# ---------------------------------------
# Tokenize the safety activation sequence
# ---------------------------------------

safety_prompt = "<|start_of_role|>safety<|end_of_role|>"

safety_tokens = tokenizer(safety_prompt)["input_ids"]

# -----------------------------------------------------------
# Process first token of activation sequence using main model
# -----------------------------------------------------------

tokens += safety_tokens[:1]

print("----------------------------------------")
print("Model input after adding first token of activation sequence (%d tokens)" % (len(tokens)))
print(repr(tokenizer.decode(tokens)))
print("----------------------------------------")

outputs = llm.generate(
    {"prompt_token_ids": tokens},
    SamplingParams(temperature=0, max_tokens=1),
)

# ------------------------------------------------------
# Process all subsequent tokens using the a-LoRA adapter
# ------------------------------------------------------

tokens += safety_tokens[1:]

print("----------------------------------------")
print("Model input after adding the rest of activation sequence (%d tokens)" % (len(tokens)))
print(repr(tokenizer.decode(tokens)))
print("----------------------------------------")

outputs = llm.generate(
    {"prompt_token_ids": tokens},
    SamplingParams(temperature=0, max_tokens=10),
    lora_request=LoRARequest("tpa-test", 1, LORA_PATH)
)

print("----------------------------------------")
print("The a-LoRA produced the following:")
print(repr(outputs[0].outputs[0].text))
print("----------------------------------------")

# ask user to confirm
if click.confirm('Do you want to verify against hf transformers implementation?', default=False):

    from vllm.distributed import cleanup_dist_env_and_memory
    del llm
    cleanup_dist_env_and_memory()


    import sys
    sys.path.insert(0,'/home/zrltpa/alora/activated-lora')

    import torch,os, copy
    from transformers import AutoTokenizer,  AutoModelForCausalLM, DynamicCache
    from alora_intrinsics.alora.peft_model_alora import PeftModelForCausalLM
    from alora_intrinsics.alora.config import aLoraConfig
    from alora_intrinsics.alora.tokenize_alora import tokenize_alora
    token = os.getenv("HF_MISTRAL_TOKEN")
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,padding_side='left',trust_remote_code=True, token=token)
    model_base = AutoModelForCausalLM.from_pretrained(MODEL_PATH,device_map="auto")

    model_alora = PeftModelForCausalLM.from_pretrained(model_base, LORA_PATH, adapter_name="safety", response_token_ids = None)
    model_alora.set_adapter("safety")

    input_text = tokenizer.decode(outputs_verify[0].prompt_token_ids + outputs_verify[0].outputs[0].token_ids)
    inputs = tokenizer(input_text, return_tensors="pt")
    prompt_cache = DynamicCache()
    with model_alora.disable_adapter():
        with torch.no_grad():
            prompt_cache = model_alora(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), past_key_values=prompt_cache).past_key_values 
    input_uq, alora_offsets = tokenize_alora(tokenizer,input_text, safety_prompt)
    past_key_values = copy.deepcopy(prompt_cache)
    output = model_alora.generate(input_uq["input_ids"].to(device), attention_mask=input_uq["attention_mask"].to(device), use_cache=True, max_new_tokens=10, return_dict_in_generate=True, past_key_values = past_key_values, alora_offsets = alora_offsets)
    output_text = tokenizer.decode(output.sequences[0])

    print("----------------------------------------")
    print("The a-LoRA produced the following:")
    print(repr(output_text.split("safety<|end_of_role|>")[-1]))
    print("----------------------------------------")



