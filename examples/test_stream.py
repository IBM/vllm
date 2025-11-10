import asyncio
import time
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

engine_args = AsyncEngineArgs(model="facebook/opt-125m", enforce_eager=True)
model = AsyncLLMEngine.from_engine_args(engine_args)

def generate_streaming(prompt):
    results_generator = model.generate(
        prompt,
        SamplingParams(temperature=0.0, logprobs=1),
        request_id=time.monotonic()
    )
    for request_output in results_generator:
        text = request_output.outputs[0].text
        tokens = request_output.outputs[0].token_ids
        logprobs = request_output.outputs[0].logprobs
        print(text, tokens, logprobs)

generate_streaming("hello")
