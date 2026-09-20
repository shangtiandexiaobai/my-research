"""粒子群优化算法（Particle Swarm Optimization, PSO）。

参考文献
--------
1. Kennedy J., Eberhart R. Particle swarm optimization. Proceedings of ICNN'95, 1995.
2. Clerc M., Kennedy J. The particle swarm - explosion, stability, and convergence
   in a multidimensional complex space. IEEE Transactions on Evolutionary
   Computation, 2002, 6(1): 58-73.

实现要点
--------
- **速度限幅（velocity clamping）**：``vmax_ratio`` 默认 **0.05**，即把速度限制在
  搜索范围的 ±5% 以内。**这不是可选项**：没有限幅时速度可以无界增长，粒子被
  边界反复截断，种群根本无法收敛。默认值 0.05 是实测选出来的，不是拍脑袋——
  见下方"限幅取值实测"。
- **参数默认值取 Clerc 收缩系数那一组**：w = 0.729，c1 = c2 = 1.4943。
  这组参数位于收敛区间内，是本算法作为**基线**的稳妥选择。
- 惯性权重支持线性递减（``w_max`` -> ``w_min``）；两者相等即为恒定。

历史记录（别删，这是你理解算法的线索）
--------------------------------------
本文件早期版本在 30 维平移+旋转 Sphere 上，5 次独立运行的平均最优值是 **4.31e+04**
（理论最优值 0），等于完全没有收敛。逐项消融后定位到**三个互相独立的缺陷**：

    配置                                     均值（越小越好）   相对上一行
    ① 冻结 gbest + 无限幅 + c1=c2=2.0         4.31e+04         ——（旧版）
    ② 修正 gbest + 无限幅 + c1=c2=2.0         1.44e+04         约 3 倍
    ③ 修正 gbest + 限幅 0.05 + c1=c2=2.0      4.92e-02         约 5.5 个数量级
    ④ 修正 gbest + 限幅 0.05 + Clerc 参数     3.54e-06         约 4 个数量级
    ⑤ 修正 gbest + 限幅 0.20 + Clerc 参数     3.70e+02         反而变差

    （30 维平移+旋转 Sphere；max_fes = 20000；5 次独立运行；复现见
     tutorials/pso_step2_anatomy.py 与 docs/PSO学习指南.md 第六节）

三个缺陷按影响排序：
    1. **缺速度限幅** —— 影响最大（②→③ 好了 5.5 个数量级）
    2. **gbest 更新条件写错** —— 拿基类的 self._best_f 去比较，条件恒为假，
       全局最优从初始化后再也不更新（详见主循环内的注释）
    3. **参数组落在收敛区间外** —— c1=c2=2.0 配 w=0.9（③→④ 又好 4 个数量级）

从 4.31e+04 到 3.54e-06：**10 个数量级的提升，一个新算子都没有加**，
全部来自"修 bug + 把参数放进合理区间"。
这就是导师们常说的那句话：**先把基线调对，再谈创新。**

限幅取值实测（Clerc 参数 w=0.729, c1=c2=1.4943；30 维；30000 FES；10 次独立运行）：

    vmax_ratio       无限幅      0.02      0.05      0.1       0.2       0.5
    Sphere(平移)     5.73e+03   1.71e-09  1.46e-10  2.78e-10  3.15e+02  1.50e+03
    Rastrigin        1.01e+02   3.03e+01  2.38e+01  3.69e+01  6.33e+01  7.38e+01
    Ackley           1.31e+00   1.31e+00  7.69e-01  7.81e-01  9.90e-01  1.38e+00

    结论：
    1) 不限幅在任何问题上都最差；
    2) 0.05 在三个问题上都是最好 —— 注意这**不等于**"0.05 是普适最优"，
       只能说明"文献里常引用的 10%~20% 未必好，必须自己调"；
    3) 0.2（本书早期采用的默认值）在平移 Sphere 上比 0.05 差 12 个数量级。

复现实验见 ``tutorials/pso_step2_anatomy.py``（实验一 ~ 实验四），
理论讲解见 ``docs/PSO学习指南.md``。

⚠️ 科研提醒：**基线算法必须调到它自己的合理水平**。用一个人为削弱的基线去
衬托自己的改进，是无效对比，属于误导性实验，审稿人一旦发现会直接拒稿。
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .base import OptimizeResult, Optimizer


class PSO(Optimizer):
    """粒子群优化算法。

    Parameters
    ----------
    pop_size : int
        种群规模（粒子个数）。
    max_fes : int
        最大函数评价次数。所有算法统一用它作为停止条件，保证比较公平。
    seed : int, optional
        随机种子。**必须固定**，否则结果不可复现。
    w_max, w_min : float
        惯性权重的上下界。两者相等 => 恒定权重；否则从 w_max 线性递减到 w_min。
    c1, c2 : float
        认知系数（信任自己的经验）与社会系数（信任群体的经验），通常取相近的值。
    vmax_ratio : float or None
        速度限幅比例，相对搜索范围 (ub - lb)。None 表示不限幅（**不推荐**）。
        默认 0.05 来自实测；不同问题的最优取值不同，正式实验里应当把它当作
        一个需要调的超参数，或改用自适应限幅。
    v_init_ratio : float
        初始速度的标准差，相对搜索范围。取 0 表示从静止开始。
    """

    name = "PSO"

    def __init__(
        self,
        pop_size: int = 50,
        max_fes: int = 300_000,
        seed: Optional[int] = None,
        w_max: float = 0.729,
        w_min: Optional[float] = None,
        c1: float = 1.4943,
        c2: float = 1.4943,
        vmax_ratio: Optional[float] = 0.05,
        v_init_ratio: float = 0.1,
    ) -> None:
        super().__init__(pop_size, max_fes, seed)
        self.w_max = w_max
        self.w_min = w_max if w_min is None else w_min
        self.c1 = c1
        self.c2 = c2
        self.vmax_ratio = vmax_ratio
        self.v_init_ratio = v_init_ratio

    def optimize(self, func, dim: int, lb, ub) -> OptimizeResult:
        self._reset()
        t0 = time.perf_counter()

        span = np.asarray(ub, dtype=float) - np.asarray(lb, dtype=float)
        n_iter = self._max_iter()

        # ---------------- 初始化 ----------------
        pop = self._init_population(dim, lb, ub)
        if self.v_init_ratio > 0:
            vel = self.rng.normal(0.0, self.v_init_ratio * span, size=pop.shape)
        else:
            vel = np.zeros_like(pop)

        vmax = None
        if self.vmax_ratio is not None:
            vmax = self.vmax_ratio * span

        fit = np.array([self._evaluate(func, x) for x in pop])
        pbest = pop.copy()
        pbest_f = fit.copy()

        g_idx = int(np.argmin(pbest_f))
        gbest = pbest[g_idx].copy()
        gbest_f = float(pbest_f[g_idx])
        self._record()

        # ---------------- 主循环 ----------------
        for it in range(n_iter):
            if self._nfes >= self.max_fes:
                break

            # 惯性权重：恒定或线性递减
            w = self.w_max - (self.w_max - self.w_min) * it / n_iter

            r1 = self.rng.random(pop.shape)
            r2 = self.rng.random(pop.shape)

            # 三股力合成速度：惯性 + 认知 + 社会
            vel = w * vel + self.c1 * r1 * (pbest - pop) + self.c2 * r2 * (gbest - pop)

            # 速度限幅（关键！）
            if vmax is not None:
                vel = np.clip(vel, -vmax, vmax)

            # 位置更新 + 边界处理
            pop = self._clip(pop + vel, lb, ub)

            fit = np.array([self._evaluate(func, x) for x in pop])

            # 更新个体历史最优（只有更小才更新）
            better = fit < pbest_f
            pbest[better] = pop[better]
            pbest_f[better] = fit[better]

            # 更新全局历史最优
            # ⚠️ 注意这里比较的必须是"gbest 自己记住的值"，不能拿基类的 self._best_f 来比：
            #    self._best_f 是"所有已评价点的最小值"，而 min(pbest_f) 恒等于它，
            #    所以写成 `if pbest_f[g_idx] < self._best_f` 会让条件**永远为假**，
            #    gbest 从初始化后再也不更新，整个种群一直飞向最初那个倒霉位置。
            #    （本文件早期版本就踩了这个坑，见模块文档的消融实验。）
            g_idx = int(np.argmin(pbest_f))
            if pbest_f[g_idx] < gbest_f:
                gbest = pbest[g_idx].copy()
                gbest_f = float(pbest_f[g_idx])

            self._record()

        return self._finalize(pbest, pbest_f, t0)
