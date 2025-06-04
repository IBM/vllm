from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch
from huggingface_hub import snapshot_download
#from transformers import LlamaForCausalLM
import os
from peft import LoraConfig, get_peft_model, TaskType
#import torch
import time
#import gc
from alora.config import aLoraConfig

BASE_NAME = "ibm-granite/granite-3.2-8b-instruct"
ALORA_NAME = "ibm-granite/granite-3.2-8b-alora-uncertainty"
invocation_string = "<|start_of_role|>certainty<|end_of_role|>"

USE_ALORA = True
os.environ['VLLM_V1_USE_ACTIVATED_LORA'] = "1" if USE_ALORA else "0"
os.environ['VLLM_USE_V1'] = "1"
os.environ['VLLM_V1_USE_DEMO_LOGGING'] = "1"
from huggingface_hub import snapshot_download

# download your LoRA adapter to ~/.cache/huggingface/…
alora_path = snapshot_download(repo_id=ALORA_NAME) #, local_dir="~/tmpcert",local_dir_use_symlinks=False)

print(alora_path)
#######################################



llm = LLM(model=BASE_NAME,
          enable_lora=True,
          enforce_eager=True,
          dtype=torch.bfloat16,
          enable_prefix_caching=False, # enable APC
          max_lora_rank=64, 
          enable_chunked_prefill=False,
         )

prompts = [
    "<|start_of_role|>user<|end_of_role|>What is MIT?<|end_of_text|>",#\n<|start_of_role|>assistant<|end_of_role|>",
    "What is MIT?",
    "<|start_of_role|>user<|end_of_role|>What is the capital of Massachusetts?<|end_of_text|>\n",#<|start_of_role|>assistant<|end_of_role|>",
    "<|start_of_role|>user<|end_of_role|>What is MIT?<|end_of_text|>",
    "<|start_of_role|>user<|end_of_role|>What is the capital of Massachusetts?<|end_of_text|>\n",#<|start_of_role|>assistant<|end_of_role|>",
    "<|start_of_role|>user<|end_of_role|>What is MIT?<|end_of_text|>",# + invocation_string,
]

sampling_params = SamplingParams(temperature=0, max_tokens=600)
if 1:
    outputsBase = llm.generate(prompts,
                       sampling_params,
                      )
    generated_text = []
    for output in outputsBase:
        prompt = output.prompt
        generated_text += [output.outputs[0].text]
        print(f"Prompt: {prompt!r}, Generated text: {generated_text[-1]!r}")
else:
    generated_text = "1. MIT, or Massachusetts Institute of Technology, is a prestigious private research university located in Cambridge, Massachusetts, USA.\n2. It was founded in 1861 and is known for its strong programs in science, technology, engineering, and mathematics (STEM).\n3. MIT is often ranked as one of the world's top universities and is a member of the Ivy League.\n4. It is"
prompts_alora = [x + y + "<|end_of_text|>\n"+ invocation_string for x,y in zip(prompts, generated_text)] 
sampling_params = SamplingParams(temperature=0, max_tokens=10)

t0 = time.time()
outputs = llm.generate(prompts_alora,
                       sampling_params,
                       lora_request = LoRARequest("UQ_adapter", 1, alora_path),
                      )
t = time.time() -t0
print(f"Time: {t}")
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")




