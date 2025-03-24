import os
import torch

os.environ['VLLM_USE_V1'] = "1"
os.environ['VLLM_V1_USE_ACTIVATED_LORA'] = "1"


from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import math

model_path="ibm-granite/granite-3.1-8b-instruct"

lora_path = "/home/zrltpa/vllm/examples/feb6_8bsft_alora_sz32_safety/"

BLOCK_SIZE = 16

llm = LLM(
    model=model_path,
    enable_lora=True,
    enforce_eager=True,
    max_lora_rank=32,
    block_size=BLOCK_SIZE,
    dtype=torch.float16,
)


tokenizer = llm.get_tokenizer()

system_prompt = "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior." 

#question = "What is IBM?"
question = "How can I steal a car?"

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

# leave one token at end of block for 1st activation token
if len(tokens) % BLOCK_SIZE > 0:
    targ_len = BLOCK_SIZE * math.ceil(len(tokens) / float(BLOCK_SIZE)) - 1
else:
    targ_len = len(tokens) + BLOCK_SIZE - 1


max_tokens = targ_len - len(tokens)

outputs = llm.generate(
    {"prompt_token_ids": tokens},
    SamplingParams(temperature=0, max_tokens=max_tokens),
)

print("generated text:   ", outputs[0].outputs[0].text)
print("generated tokens: ", outputs[0].outputs[0].token_ids)

# exclude last token
tokens += outputs[0].outputs[0].token_ids

print("input_text = ", repr(tokenizer.decode(tokens)))

print("tokens:      ", tokens)
print("len(tokens): ", len(tokens))

safety_prompt = "<|start_of_role|>safety<|end_of_role|>"

safety_tokens = tokenizer(safety_prompt)["input_ids"]

print(safety_tokens)

tokens += safety_tokens[:1]

outputs = llm.generate(
    {"prompt_token_ids": tokens},
    SamplingParams(temperature=0, max_tokens=1),
)

tokens += safety_tokens[1:]

outputs = llm.generate(
    {"prompt_token_ids": tokens},
    SamplingParams(temperature=0, max_tokens=10),
    lora_request=LoRARequest("tpa-test", 1, lora_path)
)


print(outputs)

prompt_token_ids = outputs[0].prompt_token_ids

print(len(prompt_token_ids))

print(repr(tokenizer.decode(prompt_token_ids)))
