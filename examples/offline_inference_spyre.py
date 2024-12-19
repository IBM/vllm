import os
import time

from vllm import LLM, SamplingParams

max_tokens = 3

os.environ["VLLM_SPYRE_WARMUP_PROMPT_LENS"] = '64'
os.environ["VLLM_SPYRE_WARMUP_NEW_TOKENS"] = str(max_tokens)
os.environ['VLLM_SPYRE_WARMUP_BATCH_SIZES'] = '1'

# Sample prompts.
template = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request. Be polite in your response to the "
    "user.\n\n### Instruction:\n{}\n\n### Response:")
prompt1 = template.format(
    "Provide a list of instructions for preparing chicken soup for a family "
    "of four.")
prompts = [
    prompt1,
]

# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=max_tokens,
                                 temperature=0.0,
                                 ignore_eos=True)
# Create an LLM.
llm = LLM(model="/models/llama-7b-chat",
          tokenizer="/models/llama-7b-chat",
          max_model_len=2048,
          block_size=2048,
          device="spyre")

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
print("=============== GENERATE")
t0 = time.time()
outputs = llm.generate(prompts, sampling_params)
print("Time elaspsed for %d tokens is %.2f sec" %
      (len(outputs[0].outputs[0].token_ids), time.time() - t0))
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
print(output.outputs[0])
