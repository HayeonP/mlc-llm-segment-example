import time
from mlc_llm import MLCEngine
from tqdm import tqdm
MODEL_DIR = "/home/rubis/workspace/mlc-llm/dist/Llama-3.2-1B-Instruct-q4f16_1-MLC"
MODEL_SO = MODEL_DIR + "/Llama-3.2-1B-Instruct-q4f16_1-MLC.so"

prompt = "Answer the following question in one sentence. What is the capital of South Korea?"


# Load from compiled MLC-LLM workspace
engine = MLCEngine(
    MODEL_DIR,
    model_lib=MODEL_SO,
    device="cuda",
    mode="local"
)

# Warm-up
for _ in range(5):
    _ = engine.chat.completions.create(
        model=MODEL_DIR,
        messages=[{"role":"user", "content":prompt}],
        max_tokens=10,
        stream=False
    )

# Timed runs
mlc_times = []
for _ in tqdm(range(100)):
    start = time.time()
    _ = engine.chat.completions.create(
        model=MODEL_DIR,
        messages=[{"role":"user", "content":prompt}],
        max_tokens=16,
        stream=False
    )
    end = time.time()
    mlc_times.append(end - start)

mlc_worst = max(mlc_times)

# Final output print
resp = engine.chat.completions.create(
    model=MODEL_DIR,
    messages=[{"role":"user", "content":prompt}],
    max_tokens=16,
    stream=False
)
engine.terminate()

mlc_out = "".join([choice.message.content for choice in resp.choices])
print("========== MLC-LLM LLaMA 1B ==========")

print("\nMLC-LLM Output:")
print(mlc_out.strip())
print(f"MLC-LLM Inference Time: {end - start:.3f} seconds")
