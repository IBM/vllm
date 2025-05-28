# After starting server using "vllm serve <model> --enable_lora --lora_modules..."

from openai import OpenAI
import time

model_id = "ibm-granite/granite-3.2-8b-instruct"

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

BASE_NAME = "ibm-granite/granite-3.2-8b-instruct"
ALORA_NAME = "ibm-granite/granite-3.2-8b-alora-uncertainty"
invocation_string = "<|start_of_role|>certainty<|end_of_role|>"

###################################################################

prompts = [
    "What is MIT?",
    "<|start_of_role|>user<|end_of_role|>What is MIT?<|end_of_text|>\n",
]

# Base model call
outputs_base = client.completions.create(model=BASE_NAME,
                                         prompt=prompts,  
                                         temperature=0, 
                                         max_tokens=600)
generated_text = []
for output in outputs_base:
    prompt = output.prompt
    generated_text += [output.choices[0].text]
    print(f"Prompt: {prompt!r}, Generated text: {generated_text[-1]!r}")

prompts_alora = [x + y + "<|end_of_text|>\n"+ invocation_string for x,y in zip(prompts, generated_text)] 

# Base model with aLoRA call
t0 = time.time()
alora_outputs = client.completions.create(model=ALORA_NAME,
                                          prompt=prompts_alora, 
                                          temperature=0, 
                                          max_tokens=10)
t = time.time() -t0
print(f"Time: {t}")
for output in alora_outputs:
    prompt = output.prompt
    generated_text = output.choices[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


