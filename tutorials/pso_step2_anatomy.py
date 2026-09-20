"""PSO 解剖实验 —— 用受控实验搞懂每个部件到底在干什么。

运行：
    python tutorials/pso_step2_anatomy.py

四个实验
--------
实验一  惯性权重 w 控制什么？
实验二  c1 / c2 怎么配？
实验三  种群规模越大越好吗？
实验四  速度限幅 + 它与参数的交互作用  ← 会复现出仓库旧版 PSO 的缺陷

**重要提醒**：下面每张表后的【观察要点】都是根据**实际跑出来的数据**写的，
不是抄课本。如果你自己跑出来的结论和它不一样，**以你的数据为准**，
并把差异记进 docs/experiment_log.md —— 这正是科研该有的态度。
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

_CACHE = Path(__file__).resolve().parents[1] / ".mplcache"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.classic import ackley, rastrigin, shifted_rotated, sphere  # noqa: E402

FIG_DIR = ROOT / "results" / "figures" / "tutorial"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 一个参数全开的 PSO（比仓库版多暴露几个开关，方便做对照实验）
# =====================================================================
def pso(
    func, dim, lb, ub,
    pop_size=50, max_fes=15000, seed=0,
    w=0.9, w_min=0.4, c1=2.0, c2=2.0,
    vmax_ratio=None,      # None = 不限幅（仓库旧版的做法）
    v_init_ratio=0.1,
):
    rng = np.random.default_rng(seed)
    n_iter = max(1, max_fes // pop_size)
    span = ub - lb

    x = rng.uniform(lb, ub, size=(pop_size, dim))
    v = rng.normal(0.0, v_init_ratio * span, size=(pop_size, dim))
    vmax = vmax_ratio * span if vmax_ratio is not None else None

    f = np.array([func(p) for p in x])
    pbest, pbest_f = x.copy(), f.copy()
    gi = int(np.argmin(pbest_f))
    gbest, gbest_f = pbest[gi].copy(), float(pbest_f[gi])

    curve = [(0, gbest_f)]
    nfes = pop_size

    for t in range(n_iter):
        wt = w - (w - w_min) * t / n_iter
        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))

        v = wt * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        if vmax is not None:
            v = np.clip(v, -vmax, vmax)          # ← 速度限幅
        x = np.clip(x + v, lb, ub)

        f = np.array([func(p) for p in x])
        nfes += pop_size

        better = f < pbest_f
        pbest[better] = x[better]
        pbest_f[better] = f[better]

        gi = int(np.argmin(pbest_f))
        if pbest_f[gi] < gbest_f:
            gbest, gbest_f = pbest[gi].copy(), float(pbest_f[gi])

        curve.append((nfes, gbest_f))

    return np.asarray(curve, dtype=float), gbest_f


# =====================================================================
# 实验工具
# =====================================================================
def run_config(configs, func, dim, lb, ub, runs=5, max_fes=15000):
    results = {}
    for name, params in configs.items():
        curves, finals = [], []
        for s in range(runs):
            curve, best = pso(func, dim, lb, ub, seed=s, max_fes=max_fes, **params)
            curves.append(curve)
            finals.append(best)
        min_len = min(len(c) for c in curves)
        stacked = np.vstack([c[:min_len, 1] for c in curves])
        med = np.median(stacked, axis=0)
        x_axis = curves[0][:min_len, 0]
        results[name] = (np.column_stack([x_axis, med]), float(np.mean(finals)), float(np.std(finals)))
    return results


def report(results, title, setting, note=""):
    print("\n" + "=" * 82)
    print(title)
    print(f"设置：{setting}")
    print("=" * 82)
    print(f"{'配置':<36}{'均值':>16}{'标准差':>16}")
    print("-" * 82)
    for name, (_curve, mean, std) in results.items():
        print(f"{name:<36}{mean:>16.4e}{std:>16.4e}")
    best = min(results.items(), key=lambda kv: kv[1][1])[0]
    print(f"\n本次最优：{best}")
    if note:
        print(note)


def plot(results, title, filename):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=150)
        for name, (curve, _m, _s) in results.items():
            ax.plot(curve[:, 0], curve[:, 1], label=name, linewidth=1.4)
        ax.set_yscale("log")
        ax.set_xlabel("FES (function evaluations)")
        ax.set_ylabel("Best fitness (log scale)")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.legend(frameon=False, fontsize=7.5)
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        plt.close(fig)
        print(f"图已保存: results/figures/tutorial/{filename}")
    except Exception as e:
        print(f"[跳过绘图] {e}")


# =====================================================================
DIM = 30
RAS_LB, RAS_UB = -5.12, 5.12

print(__doc__)
print(f"统一维度 = {DIM}，50 个粒子\n")


# =====================================================================
# 实验一：惯性权重
# =====================================================================
r1 = run_config(
    {
        "w=0.2 固定":        dict(w=0.2, w_min=0.2, c1=2.0, c2=2.0, vmax_ratio=0.2),
        "w=0.4 固定":        dict(w=0.4, w_min=0.4, c1=2.0, c2=2.0, vmax_ratio=0.2),
        "w=0.6 固定":        dict(w=0.6, w_min=0.6, c1=2.0, c2=2.0, vmax_ratio=0.2),
        "w=0.9 固定":        dict(w=0.9, w_min=0.9, c1=2.0, c2=2.0, vmax_ratio=0.2),
        "线性递减 0.9->0.4":  dict(w=0.9, w_min=0.4, c1=2.0, c2=2.0, vmax_ratio=0.2),
    },
    rastrigin, DIM, RAS_LB, RAS_UB, runs=5, max_fes=15000,
)
report(
    r1, "实验一 · 惯性权重 w",
    "Rastrigin 30维, 15000 FES, 50 粒子, 5 次独立运行",
    """
【实测结论】
- w=0.9 明显最差（约 2.3e+02），粒子"刹不住车"，在多峰地形上乱跳。
- w=0.2 / 0.4 / 0.6 差距不大，w=0.4 附近最好（约 5.1e+01）。
- ⚠️ 注意：**"线性递减 0.9->0.4" 这次并没有赢过固定 w=0.4**（6.8e+01 vs 5.1e+01）。
  课本常说"前期大 w 探索、后期小 w 开发"，那是一个合理的**启发式**，
  但在具体问题+具体 FES 预算下**不一定成立**。
  → 这就是科研和背书的分界线：**不要假设，去测。**
""",
)
plot(r1, "Effect of inertia weight w on Rastrigin (D=30)", "exp1_inertia_weight.png")


# =====================================================================
# 实验二：c1 / c2
# =====================================================================
r2 = run_config(
    {
        "c1=2.0, c2=2.0":            dict(w=0.9, w_min=0.4, c1=2.0, c2=2.0, vmax_ratio=0.2),
        "c1=2.5, c2=0.5 重个人":      dict(w=0.9, w_min=0.4, c1=2.5, c2=0.5, vmax_ratio=0.2),
        "c1=0.5, c2=2.5 重群体":      dict(w=0.9, w_min=0.4, c1=0.5, c2=2.5, vmax_ratio=0.2),
        "c1=0.0, c2=2.0 无个体经验":   dict(w=0.9, w_min=0.4, c1=0.0, c2=2.0, vmax_ratio=0.2),
        "c1=1.4943, c2=1.4943 Clerc": dict(w=0.729, w_min=0.729, c1=1.4943, c2=1.4943, vmax_ratio=0.2),
    },
    rastrigin, DIM, RAS_LB, RAS_UB, runs=5, max_fes=15000,
)
report(
    r2, "实验二 · 认知系数 c1 与社会系数 c2",
    "Rastrigin 30维, 15000 FES, 50 粒子, 5 次独立运行",
    """
【实测结论】
- c1=0（完全不要个人经验）明显最差（约 1.2e+02），多样性崩塌最快。
- 其余四组在 6.0e+01 ~ 7.1e+01 之间，差距**并不大**，谈不上谁碾压谁。
  这次是"重个人"（c1=2.5, c2=0.5）略优。
- ⚠️ 诚实的判断：5 次运行、单个函数，**这个差异还不足以支撑结论**。
  真要下结论，需要 30 次运行 + 多个函数 + 统计检验（这正是规范实验的意义）。
- 可以确定的是：c1 与 c2 **不能都为 0**，否则退化为随机游走；
  c1=0 会显著变差。至于"最佳配比"，因问题而异，属于可研究的空间（见指南 P2）。
""",
)
plot(r2, "Cognitive vs social coefficients on Rastrigin (D=30)", "exp2_c1_c2.png")


# =====================================================================
# 实验三：种群规模（固定 FES）
# =====================================================================
r3 = run_config(
    {
        "pop=10":  dict(pop_size=10, w=0.9, w_min=0.4, c1=2.0, c2=2.0, vmax_ratio=0.2),
        "pop=30":  dict(pop_size=30, w=0.9, w_min=0.4, c1=2.0, c2=2.0, vmax_ratio=0.2),
        "pop=50":  dict(pop_size=50, w=0.9, w_min=0.4, c1=2.0, c2=2.0, vmax_ratio=0.2),
        "pop=100": dict(pop_size=100, w=0.9, w_min=0.4, c1=2.0, c2=2.0, vmax_ratio=0.2),
        "pop=200": dict(pop_size=200, w=0.9, w_min=0.4, c1=2.0, c2=2.0, vmax_ratio=0.2),
    },
    rastrigin, DIM, RAS_LB, RAS_UB, runs=5, max_fes=15000,
)
report(
    r3, "实验三 · 种群规模（FES 固定为 15000）",
    "Rastrigin 30维, 15000 FES, 5 次独立运行",
    """
【实测结论】
- 最小值出现在 pop=30（约 4.8e+01），之后**单调变差**：pop=200 最差（约 1.4e+02）。
- 原因很直接：FES 固定时 N × T = 15000 是常数。
  pop=200 只能跑 75 代 —— 粒子还没来得及把信息传开，预算就烧完了。
- 结论：**"种群越大越好"是错的。** 而且做论文时必须固定 FES 再比种群规模，
  否则"种群大但代数少"和"种群小但代数多"根本不可比。
""",
)
plot(r3, "Population size on Rastrigin (D=30, fixed FES)", "exp3_population_size.png")


# =====================================================================
# 实验四：速度限幅 × 参数组（本实验最重要）
# =====================================================================
f_shift, _, _ = shifted_rotated(sphere, DIM, shift=60.0, seed=1)
r4 = run_config(
    {
        "c=2.0,线性w | 无限幅":   dict(w=0.9, w_min=0.4, c1=2.0, c2=2.0, vmax_ratio=None),
        "c=2.0,线性w | 限幅0.05": dict(w=0.9, w_min=0.4, c1=2.0, c2=2.0, vmax_ratio=0.05),
        "c=2.0,线性w | 限幅0.2":  dict(w=0.9, w_min=0.4, c1=2.0, c2=2.0, vmax_ratio=0.2),
        "Clerc | 无限幅":         dict(w=0.729, w_min=0.729, c1=1.4943, c2=1.4943, vmax_ratio=None),
        "Clerc | 限幅0.05":       dict(w=0.729, w_min=0.729, c1=1.4943, c2=1.4943, vmax_ratio=0.05),
        "Clerc | 限幅0.2":        dict(w=0.729, w_min=0.729, c1=1.4943, c2=1.4943, vmax_ratio=0.2),
        "Clerc | 限幅0.5":        dict(w=0.729, w_min=0.729, c1=1.4943, c2=1.4943, vmax_ratio=0.5),
    },
    f_shift, DIM, -100.0, 100.0, runs=10, max_fes=30000,
)
report(
    r4, "实验四 · 速度限幅，以及它与参数的交互作用",
    "平移 Sphere 30维, 30000 FES, 50 粒子, 10 次独立运行",
    """
【实测结论】
1. **不限幅一定很差**：两种参数组下都是 1e+03 ~ 1e+04 量级，等于没收敛。
   原因：不限幅时速度可以无界增长，粒子被边界反复"拍"在墙上。
   → 仓库旧版 algorithms/pso.py 就是这种写法（它另外还有一个 gbest 更新条件
     写错、导致全局最优被冻结的 bug，见 docs/PSO学习指南.md 第六节）。
   注意：本文件的 pso() 函数本身没有那个 bug，所以这里测出的差异
   **纯粹来自限幅与参数**。

2. **"限幅 0.2" 这个书本上常见的取值，在这组实验里反而很差**
   （Clerc 参数下 3.1e+02，比限幅 0.05 的 1.5e-10 差了 12 个数量级）。
   → 别迷信"经验值 10%~20%"，**限幅大小必须自己调**。

3. **参数与限幅有交互作用**：同样的限幅 0.05，Clerc 参数组比 c=2.0 组
   好 3 个数量级 —— 说明"把参数放进收敛区间"本身就很关键。

4. 限幅太大（0.5）几乎等于没限幅。
""",
)
plot(r4, "Velocity clamping x parameter set on shifted Sphere (D=30)", "exp4_velocity_clamping.png")

print(
    """
======================================================================
做完这四个实验，你应该能回答：
======================================================================
1. 为什么仓库旧版 PSO 在平移 Sphere 上是 1e+04 量级？
2. 「线性递减 w 一定比固定 w 好吗？」——你的数据怎么说？
3. 为什么 c1 和 c2 不能都为 0？c1=0 会发生什么？
4. 固定 FES 时，种群从 50 改到 200 是变强还是变弱？为什么？
5. 为什么不能迷信"限幅取搜索范围的 10%~20%"这个说法？

把答案写进 docs/experiment_log.md —— 这是你第一次正式的"实验+结论"训练。
然后去做 step3 挑战题。
======================================================================
"""
)
