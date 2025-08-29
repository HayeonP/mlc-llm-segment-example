import time
from mlc_llm import MLCEngine
from tqdm import tqdm
# MODEL_DIR = "/home/rubis/workspace/mlc-llm/dist/Llama-3.2-1B-Instruct-q4f16_1-MLC"
# MODEL_SO = MODEL_DIR + "/Llama-3.2-1B-Instruct-q4f16_1-MLC.so"

MODEL_DIR = "/home/rubis/workspace/llama/mlc-llm-models/llama-3.2-1b/workspace"
MODEL_SO = MODEL_DIR + "/llama-3.2-1b-cuda.so"

prompt = "Answer the following question in one sentence. What is the capital of South Korea?"


# Load from compiled MLC-LLM workspace
print("# Init MLCEngine")
engine = MLCEngine(
    MODEL_DIR,
    model_lib=MODEL_SO,
    device="cuda",
    mode="local"
)

# Warm-up
print("# Inference")
for _ in range(1):
    resp = engine.chat.completions.create(
        model=MODEL_DIR,
        messages=[{"role":"user", "content":prompt}],
        max_tokens=10,
        stream=False
    )

print("# Termination")
engine.terminate()

print("# Get output")
mlc_out = "".join([choice.message.content for choice in resp.choices])
print("========== MLC-LLM LLaMA 1B ==========")

print("\nMLC-LLM Output:")
print(mlc_out.strip())
