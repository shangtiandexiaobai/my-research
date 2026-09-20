"""PSO 最小可运行版本 —— 核心逻辑只有 6 行。

运行：
    python tutorials/pso_step1_minimal.py

学习目标
--------
把 PSO 从"论文里的公式"变成"你脑子里的一幅画"：
一群鸟在找食物，每只鸟的下一步怎么走，只由三样东西决定。

这个文件**故意不用仓库里的任何框架代码**，全部逻辑摊平在一个文件里。
先把这 100 行彻底看懂，再去看 algorithms/pso.py 就毫无难度了。
"""

import os
from pathlib import Path

import numpy as np

# matplotlib 缓存目录（避免在受限环境里因权限问题崩溃）
_CACHE = Path(__file__).resolve().parents[1] / ".mplcache"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE))


# =====================================================================
# 1. 目标函数：Sphere（碗形，最低点在原点，最小值 0）
# =====================================================================
def sphere(x):
    """x 是长度 dim 的一维向量，返回 f(x) = sum(x_i^2)。"""
    return float(np.sum(x ** 2))


# =====================================================================
# 2. 问题设置
# =====================================================================
dim = 30                    # 变量个数（30 维）
lb, ub = -100.0, 100.0      # 每个变量的取值范围
n_particles = 50            # 粒子个数（"鸟"的数量）
max_iter = 500              # 迭代多少代

# =====================================================================
# 3. 初始化
# =====================================================================
rng = np.random.default_rng(0)                        # 固定种子 => 结果可复现

x = rng.uniform(lb, ub, size=(n_particles, dim))      # 每只鸟的当前位置
v = np.zeros((n_particles, dim))                      # 每只鸟的当前速度（从静止开始）

pbest = x.copy()                                      # 每只鸟"自己去过的最好位置"
pbest_f = np.array([sphere(p) for p in x])            # 那个位置的函数值

gi = int(np.argmin(pbest_f))                          # 目前全场最好的是第几只鸟
gbest = pbest[gi].copy()                              # 全场最好位置
gbest_f = float(pbest_f[gi])                          # 全场最好值

# 三个关键参数
w = 0.7        # 惯性权重：多大程度上保持原来的飞行方向
c1 = 1.5       # 认知系数：多相信"自己的经验"
c2 = 1.5       # 社会系数：多相信"群体的经验"

history = [gbest_f]

# =====================================================================
# 4. 主循环 —— ★★★ PSO 的全部精华就在中间这三行 ★★★
# =====================================================================
for t in range(max_iter):
    # 每一维、每一只鸟都用独立的随机数（这是 PSO 随机性的来源）
    r1 = rng.random((n_particles, dim))
    r2 = rng.random((n_particles, dim))

    # ★ 三股力合成速度：惯性 + 奔向自己的历史最优 + 奔向群体的历史最优
    v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)

    # ★ 用速度更新位置，并保证不越界
    x = np.clip(x + v, lb, ub)

    # ★ 评价新位置
    f = np.array([sphere(p) for p in x])

    # ---------------------------------------------------------------
    # 下面两段是"记账"，不是 PSO 本身，但少了它算法就没有意义
    # ---------------------------------------------------------------
    # 更新每只鸟自己的历史最优（只有更好才更新）
    better = f < pbest_f
    pbest[better] = x[better]
    pbest_f[better] = f[better]

    # 更新全场最优
    gi = int(np.argmin(pbest_f))
    if pbest_f[gi] < gbest_f:
        gbest = pbest[gi].copy()
        gbest_f = float(pbest_f[gi])

    history.append(gbest_f)

# =====================================================================
# 5. 输出结果
# =====================================================================
print("=" * 60)
print("PSO 极简版运行结果")
print("=" * 60)
print(f"问题        : Sphere, {dim} 维")
print(f"粒子数      : {n_particles}")
print(f"迭代代数    : {max_iter}")
print(f"最终最优值  : {gbest_f:.6e}")
print(f"理论最优值  : 0")
print(f"最优解前 5 维: {np.round(gbest[:5], 4)}")
print()
print("收敛过程（每 100 代）:")
for i in range(0, max_iter + 1, 100):
    print(f"  第 {i:>4} 代: {history[i]:.6e}")

# =====================================================================
# 6. 画收敛曲线
# =====================================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(__file__).resolve().parents[1] / "results" / "figures" / "tutorial"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.plot(range(len(history)), history, linewidth=1.6, color="#d35400")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness (log scale)")
    ax.set_title(f"Minimal PSO on Sphere (D={dim}, pop={n_particles})")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    save_path = out_dir / "step1_minimal_pso.png"
    fig.savefig(save_path, bbox_inches="tight")
    print(f"\n收敛曲线已保存: {save_path}")
except Exception as e:
    print(f"\n[跳过绘图] {e}")

# =====================================================================
# 7. 你的第一个练习（不要跳过）
# =====================================================================
print(
    """
======================================================================
动手练习（做完再进 step2）
======================================================================
1. 把上面标 ★ 的三行注释掉，凭记忆自己重写一遍，看结果是否一致。
2. 把 w 改成 0.0，会发生什么？为什么？
3. 把 c2 改成 0.0（去掉"社会经验"），5 只鸟各自为战，结果会怎样？
4. 把 n_particles 改成 1，还能算"群"吗？
5. 把 sphere 换成 rastrigin（benchmarks/classic.py 里有），还收敛吗？
6. 想一想：如果 max_iter 无限大，PSO 一定会找到全局最优吗？

想清楚这 6 个问题，你就已经比 80% 只会抄公式的人更懂 PSO 了。
======================================================================
"""
)
