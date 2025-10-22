import re
import matplotlib.pyplot as plt


chunk_list = [16, 32, 64, 128, 256]
# chunk_list = [256]

plt.figure(8,4)
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
        if (i+1) * chunk > 1024: break
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
    if chunk == 16:
        marker = 'o'
        markerfacecolor='none'
        markersize=5
    elif chunk == 32:
        marker = 's'
        markersize=8
    elif chunk == 64:
        marker = 'x'
        markersize=12
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

plt.ylim(0,200)
plt.xlabel("Last token index")
plt.ylabel("Prefill time (ms)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig('plot.pdf')
plt.close()

plt.figure(8,4)
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
        if (i+1) * chunk > 1024: break
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
        
    if chunk == 16:
        marker = 'o'
        markerfacecolor='none'
        markersize=5
    elif chunk == 32:
        marker = 's'
        markersize=8
    elif chunk == 64:
        marker = 'x'
        markersize=12
        markerfacecolor='none'
    elif chunk == 128:
        marker = 'o'
        markersize=8
        markerfacecolor='black'
    elif chunk == 256:
        marker = 's'
        markersize=8
        markerfacecolor='black'
    # ----- ④ 그래프 그리기 -----
    plt.plot(x, accumulated_y, marker=marker, markersize=markersize, markerfacecolor=markerfacecolor, linewidth=1.5, label=f"chunk={chunk}", color='black')

plt.xlabel("Last token index")
plt.ylabel("Accumulated prefill time (ms)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig('plot_accumulated.pdf')