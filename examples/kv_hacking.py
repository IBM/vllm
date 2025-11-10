import os

from vllm import LLM, SamplingParams

os.environ["VLLM_USE_V1"] = "1"

llm = LLM(
    model="facebook/opt-125m",
    gpu_memory_utilization=0.4,
    enforce_eager=True,
    block_size=16,
)

doc1 = "The Arsenal Football Club, commonly known as simply Arsenal, is a professional football club based in Islington, North London, England. They compete in the Premier League, the top tier of English football. In domestic football, Arsenal have won 13 league titles (including one unbeaten title), a record 14 FA Cups, two League Cups, 17 FA Community Shields, and a Football League Centenary Trophy. In European football, they have one European Cup Winners' Cup and one Inter-Cities Fairs Cup. In terms of trophies won, it is the third-most successful club in English football.[2]"
doc2 = "Switzerland,[d] officially the Swiss Confederation,[e] is a landlocked country located in west-central Europe.[f][13] It is bordered by Italy to the south, France to the west, Germany to the north, and Austria and Liechtenstein to the east. Switzerland is geographically divided among the Swiss Plateau, the Alps and the Jura; the Alps occupy the greater part of the territory, whereas most of the country's nearly 9 million people are concentrated on the plateau, which hosts its largest cities and economic centres, including Zurich, Geneva, and Lausanne.[14]"

tokenizer = llm.get_tokenizer()

N_BLOCKS = 2

tok1 = tokenizer(doc1)["input_ids"][: N_BLOCKS * 16]
tok2 = tokenizer(doc2)["input_ids"][: (N_BLOCKS * 16)]

assert len(tok1) == N_BLOCKS * 16
assert len(tok2) == N_BLOCKS * 16

prompt1 = {"prompt_token_ids": tok1}

prompt2 = {"prompt_token_ids": tok2}

prompt12 = {"prompt_token_ids": tok1 + tok2}

# only do prefill
prefill_params = SamplingParams(temperature=0.0, max_tokens=1)

print("----------- PREFILL PROMPT1 --------------:")
output = llm.generate(prompt1, prefill_params)
print(output)

print("----------- PREFILL PROMPT2 --------------:")
output = llm.generate(prompt2, prefill_params)
print(output)

print("----------- PREFILL PROMPT12 --------------:")
output = llm.generate(prompt12, prefill_params)
print(output)
