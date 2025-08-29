import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, modeling_utils

# ---------------------------------------------
# Hugging Face Inference
# ---------------------------------------------
print("========== Hugging Face LLaMA 1B ==========")
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = "/home/rubis/workspace/llm_models/Llama-3.2-1B-Instruct" # "From meta_llama"
prompt = "Answer the following question in one sentence. What is the capital of South Korea?"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16
)

# Warm-up
for _ in range(5):
    _ = hf_pipeline(prompt, max_new_tokens=10)

# Timed runs
hf_times = []
for _ in range(100):
    start = time.time()
    _ = hf_pipeline(prompt, max_new_tokens=16, do_sample=True)
    end = time.time()
    hf_times.append(end - start)

hf_worst = max(hf_times)
hf_output = hf_pipeline(prompt, max_new_tokens=16)[0]["generated_text"]

print("Hugging Face Output:")
print(hf_output.strip())
print(f"Hugging Face Worst Inference Time (100 runs): {hf_worst:.3f} seconds\n")