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

from huggingface_hub import snapshot_download

# download your LoRA adapter to ~/.cache/huggingface/…
alora_path = snapshot_download(repo_id=ALORA_NAME)





#######################################



llm = LLM(model=BASE_NAME,
          #dtype='float16', # float16 for v100i
          enable_lora=True,
          enforce_eager=True,
          dtype=torch.bfloat16,
          enable_prefix_caching=False, # enable APC
          max_lora_rank=64,
          enable_chunked_prefill=False,
          tokenizer="no_chat_template_tokenizer",
         )

prompts = [
    #"What is MIT?",
    "<|start_of_role|>user<|end_of_role|>What is MIT?<|end_of_text|>\n",#<|start_of_role|>assistant<|end_of_role|>",# + invocation_string,
    "What is MIT?",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
#prompts_alora = [
#        "<|start_of_role|>user<|end_of_role|>What is MIT?<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>" +  + invocation_string,
#]
sampling_params = SamplingParams(temperature=0, max_tokens=600)
if 1:
    outputsBase = llm.generate(prompts,
                       sampling_params,
  #                     # need to increment global ID below if recomputing lora values without restarting kernel
   #                    k_offsets = [4],
                      )
    generated_text = []
    for output in outputsBase:
        prompt = output.prompt
        generated_text += [output.outputs[0].text]
        print(f"Prompt: {prompt!r}, Generated text: {generated_text[-1]!r}")
else:
    generated_text = "1. MIT, or Massachusetts Institute of Technology, is a prestigious private research university located in Cambridge, Massachusetts, USA.\n2. It was founded in 1861 and is known for its strong programs in science, technology, engineering, and mathematics (STEM).\n3. MIT is often ranked as one of the world's top universities and is a member of the Ivy League.\n4. It is"
prompts_alora = [x + y + "<|end_of_text|>\n"+ invocation_string for x,y in zip(prompts, generated_text)] #[
        #"<|start_of_role|>user<|end_of_role|>What is MIT?<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>" + generated_text[0] + "<|end_of_text|>\n" + invocation_string,
        #"<|start_of_role|>user<|end_of_role|>What is MIT?<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>" + generated_text[1] + "<|end_of_text|>\n" + invocation_string,
        #prompts[0] + generated_text[0] + "<|end_of_text|>\n" + invocation_string,
        #prompts[1] + generated_text[1] + "<|end_of_text|>\n" + invocation_string,
#]
sampling_params = SamplingParams(temperature=0, max_tokens=10)
tok_prompts = []
for prompt in prompts_alora:
    input_tokens = tokenizer(prompt)['input_ids']
    #print(prompt)
    #print(input_tokens)
    tok_prompts.append({"prompt_token_ids": input_tokens})
t0 = time.time()
outputs = llm.generate(tok_prompts,
                       sampling_params,
                       # need to increment global ID below if recomputing lora values without restarting kernel
                       lora_request = LoRARequest("UQ_adapter", 1, alora_path),
                       #tokenizer="no_chat_template_tokenizer",
                       k_offsets = [3]*len(prompts_alora),#[3,3],
                      )
t = time.time() -t0
print(f"Time: {t}")
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")




