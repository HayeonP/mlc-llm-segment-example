import re
import matplotlib.pyplot as plt


chunk_list = [4, 8, 16, 32, 64, 128, 256]
# chunk_list = [256]

plt.figure(figsize=(8, 4))

for chunk in chunk_list:

    # ----- ① txt 파일 읽기 -----
    # prefill_log.txt 파일이 같은 폴더에 있다고 가정
    with open(f"output_chunk_{chunk}.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    # ----- ② 데이터 파싱 -----
    # "prefill: 18ms" 형태에서 숫자만 추출
    x = []
    y = []
    
    for i, line in enumerate(lines):
        if (i+1) * chunk > 2048: break
        line = line.strip()
        if not line:
            continue
        if "prefill" in line:
            try:
                value = int(line.split(":")[1].replace("ms", "").strip())
                y.append(value)
                x.append((i+1) * chunk)
            except ValueError: 
                pass  # 혹시 형식이 다른 줄은 무시
        

    # ----- ④ 그래프 그리기 -----
    plt.plot(x, y, marker='o', markersize=5, linewidth=1.5, label=f"chunk={chunk}")
    print(f"Chunk size: {chunk} / total prefill time: {sum(y)} ms")

plt.title("Prefill Time per Step")
plt.xlabel("Step Index")
plt.ylabel("Prefill Time (ms)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig('plot.png')
plt.close()

for chunk in chunk_list:

    # ----- ① txt 파일 읽기 -----
    # prefill_log.txt 파일이 같은 폴더에 있다고 가정
    with open(f"output_chunk_{chunk}.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    # ----- ② 데이터 파싱 -----
    # "prefill: 18ms" 형태에서 숫자만 추출
    x = []
    accumulated_y = []
    
    for i, line in enumerate(lines):
        if (i+1) * chunk > 2048: break
        line = line.strip()
        if not line:
            continue
        if "prefill" in line:
            try:
                value = int(line.split(":")[1].replace("ms", "").strip())
                x.append((i+1) * chunk)
                if i == 0: accumulated_y.append(value)
                else: accumulated_y.append(accumulated_y[-1]+value)
                
                if accumulated_y[-1] > 2000: break
            except ValueError:
                pass  # 혹시 형식이 다른 줄은 무시
        

    # ----- ④ 그래프 그리기 -----
    plt.plot(x, accumulated_y, marker='o', markersize=5, linewidth=1.5, label=f"chunk={chunk}")

plt.title("Prefill Time per Step")
plt.xlabel("Step Index")
plt.ylabel("Accumulated Prefill Time (ms)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig('plot_accumulated.png')