# -*- coding: utf-8 -*-

import subprocess
import sys
import time
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统常用中文字体
rcParams['axes.unicode_minus'] = False    # 正常显示负号

python_file = "myapp.py"
range_limit = 10
test_sizes = [100, 500, 1000, 2000]
prof_dir = "prof_files"
os.makedirs(prof_dir, exist_ok=True)

total_times = []

for n in test_sizes:
    prof_file = os.path.join(prof_dir, f"result_{n}_questions.prof")
    print(f"\n分析 {n} 道题，生成 {prof_file} ...")

    start_time = time.perf_counter()

    subprocess.run(
        [sys.executable, "-m", "cProfile", "-o", prof_file, python_file, "-n", str(n), "-r", str(range_limit)])

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    total_times.append(elapsed)
    print(f"总耗时: {elapsed:.3f} 秒")

    # 打开 SnakeViz 网页，并在浏览器标签显示题量
    subprocess.Popen(["snakeviz", prof_file])

# 绘制题量 vs 总耗时折线图
plt.figure(figsize=(8, 5))
plt.plot(test_sizes, total_times, marker='o', color='blue', label='总运行时间 (秒)')
plt.title("四则运算题目生成性能趋势")
plt.xlabel("题目数量")
plt.ylabel("总运行时间 (秒)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("performance_trend.png", dpi=300)
plt.show()
print("\n折线图已保存为 performance_trend.png")
