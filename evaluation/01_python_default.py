import time
from mlc_llm import MLCEngine
from tqdm import tqdm
            
MODEL_DIR = "/home/rubis/workspace/llama/mlc-llm-models/llama-3.2-1b/workspace"
MODEL_SO = MODEL_DIR + "/llama-3.2-1b-cuda.so"

prompt = "Why USA is the one of the strongest country?"


# Load from compiled MLC-LLM workspace
engine = MLCEngine(
    MODEL_DIR,
    model_lib=MODEL_SO,
    device="cuda",
    mode="local"
)

n = 1
warmup = 2
inference_time_list = []
max_tokens = 256
for _ in tqdm(range(n+warmup)):
    start = time.time()
    resp = engine.chat.completions.create(
        model=MODEL_DIR,
        messages=[{"role":"user", "content":prompt}],
        max_tokens=max_tokens,
        stream=False,
        seed=4542 # For same experiment
    )
    
    mlc_out = "".join([choice.message.content for choice in resp.choices])
    print(mlc_out.strip())
    
    end = time.time()
    inference_time_list.append(end-start)
    print("=========================")


print("# Termination")
engine.terminate()

inference_time_list = inference_time_list[warmup:]
inference_time_avg = sum(inference_time_list)/len(inference_time_list) * 1000
inference_time_max = max(inference_time_list) * 1000

print("========== Inference time ==========")
print(f"Average response time: {inference_time_avg}ms")
print(f"Worst response time: {inference_time_max}ms")

