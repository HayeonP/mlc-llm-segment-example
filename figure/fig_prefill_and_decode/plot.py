import re
import matplotlib.pyplot as plt


input_length_list = [44, 399, 774, 1149, 1524, 1899, 1908, 2274, 2649, 3024, 3399, 3783, 4524]

plt.figure(figsize=(8, 4))

for input_length in input_length_list:

    # ----- ① txt 파일 읽기 -----
    # prefill_log.txt 파일이 같은 폴더에 있다고 가정
    with open(f"output_chunk64_input{input_length}.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    # ----- ② 데이터 파싱 -----
    # "prefill: 18ms" 형태에서 숫자만 추출
    x = []
    y = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if "execute" in line:
            try:
                value = float(line.split(":")[1].replace("ms", "").strip())
                y.append(value)
                x.append(input_length + (i+1))
            except ValueError: 
                pass  # 혹시 형식이 다른 줄은 무시
        

    # ----- ④ 그래프 그리기 -----
    # 
    plt.plot(x, y, linewidth=0.5, label=f"input_length={input_length}", color='black')
    print(f"input length: {input_length}, total execute time: {sum(y)} ms")

with open(f"output_chunk_64_prefill.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# ----- ② 데이터 파싱 -----
# "prefill: 18ms" 형태에서 숫자만 추출
x = []
y = []
chunk = 64
i = 1
for line in lines:
    if i * chunk > 4096: break
    line = line.strip()
    if not line:
        continue
    if "debug" in line:
        continue
    if "prefill" in line:
        try:
            value = int(line.split(":")[1].replace("ms", "").strip())
            y.append(value)
            x.append(i * chunk)
            i = i + 1
        except ValueError: 
            pass  # 혹시 형식이 다른 줄은 무시

# ----- ④ 그래프 그리기 -----
if chunk == 16:
    marker = 'o'
    markerfacecolor='none'
    markersize=5
elif chunk == 32:
    marker = 's'
    markersize=8
elif chunk == 64:
    marker = 'x'
    markersize=5
    markerfacecolor='none'
elif chunk == 128:
    marker = 'o'
    markersize=8
    markerfacecolor='black'
elif chunk == 256:
    marker = 's'
    markersize=8
    markerfacecolor='black'
plt.plot(x, y, marker=marker, markerfacecolor=markerfacecolor, markersize=markersize, linewidth=0.5, label=f"chunk={chunk}", color='black')
print(f"Chunk size: {chunk} / total prefill time: {sum(y)} ms")


plt.xlabel("Last token index")
plt.ylabel("Time (ms)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlim(0, 4096)
plt.tight_layout()
plt.savefig('plot.pdf')
plt.close()
