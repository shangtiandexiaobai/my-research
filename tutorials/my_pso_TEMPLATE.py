"""从零实现你自己的 PSO —— 骨架文件

使用方式
--------
1. 复制成 my_pso.py：
       copy tutorials\\my_pso_TEMPLATE.py tutorials\\my_pso.py
2. 搜索文件里的 TODO，一处一处补完（**不要偷看 algorithms/pso.py**，写完再看）
3. 验收：
       python tutorials\\pso_step3_challenge.py

验收要求
--------
R1  接口：PSO(pop_size=..., max_fes=..., seed=...).optimize(func, dim, lb, ub)
R2  返回对象含 best_x / best_f / curve / nfes 四个属性
R3  curve 是 (N, 2) 数组，两列为 [FES, 历史最优值]，且第二列单调不增
R4  统计函数评价次数，不得超过 max_fes（允许溢出不超过一轮种群评价）
R5  固定 seed 时两次运行结果**完全一致**
R6  最终解必须落在 [lb, ub] 内
R7  在 Sphere(30维) 上 < 1e-2；平移 Sphere(30维) 上 < 1；Rastrigin(30维) 上 < 100
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Result:
    """一次运行的结果。字段名不要改，验收脚本按名字读取。"""

    best_x: np.ndarray
    best_f: float
    curve: np.ndarray
    nfes: int


class PSO:
    """粒子群优化算法（你自己写的那一版）。"""

    def __init__(
        self,
        pop_size: int = 50,
        max_fes: int = 30_000,
        seed: Optional[int] = 0,
        w: float = 0.729,
        w_min: Optional[float] = None,
        c1: float = 1.4943,
        c2: float = 1.4943,
        vmax_ratio: Optional[float] = 0.2,
    ) -> None:
        # ------------------------------------------------------------------
        # TODO 1
        # 保存所有参数；用 np.random.default_rng(seed) 创建随机数发生器。
        # 注意：w_min 为 None 时表示"惯性权重恒定，不递减"。
        # ------------------------------------------------------------------
        raise NotImplementedError("TODO 1")

    def optimize(self, func, dim: int, lb, ub) -> Result:
        """在 [lb, ub]^dim 上最小化 func。"""
        # ------------------------------------------------------------------
        # TODO 2  初始化
        #   x      : (pop_size, dim) 在 [lb, ub] 上均匀随机    <- 位置
        #   v      : (pop_size, dim) 初始速度（想想：全 0 好还是小随机好？）
        #   pbest  : 每只粒子的历史最优位置（一开始就是 x）
        #   pbest_f: 对应的函数值
        #   gbest  / gbest_f : 全局最优
        #   别忘了给评价次数计数器置 0
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # TODO 3  主循环
        #   循环代数 = max_fes // pop_size
        #   每一代：
        #     a) 计算本代的惯性权重 wt
        #        （如果 w_min 是 None，就恒为 w；否则从 w 线性降到 w_min）
        #     b) 抽两组随机数 r1, r2，形状与 x 相同
        #     c) 速度更新：v = wt*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
        #     d) 速度限幅：如果 vmax_ratio 不是 None，把 v 截断到 ±vmax_ratio*(ub-lb)
        #     e) 位置更新 + 边界处理：x = clip(x+v, lb, ub)
        #     f) 评价新位置的函数值，并累加评价次数
        #     g) 更新 pbest / pbest_f（只有更小才更新）
        #     h) 更新 gbest / gbest_f
        #     i) 记录一条曲线点：(当前累计 FES, 当前 gbest_f)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # TODO 4  收敛曲线
        #   把记录点整理成 (N, 2) 的 numpy 数组，第一列 FES，第二列最优值。
        #   想想：这个数组的第二列为什么一定单调不增？
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # TODO 5  返回 Result(best_x=..., best_f=..., curve=..., nfes=...)
        # ------------------------------------------------------------------
        raise NotImplementedError("TODO 2~5")


# =====================================================================
# 自测小工具：写完 PSO 后可以直接运行本文件做快速检查
# =====================================================================
if __name__ == "__main__":
    def sphere(x):
        return float(np.sum(x ** 2))

    opt = PSO(pop_size=50, max_fes=20000, seed=0)
    res = opt.optimize(sphere, 30, -100.0, 100.0)
    print(f"best_f = {res.best_f:.6e}   (期望 < 1e-2)")
    print(f"nfes   = {res.nfes}")
    print(f"curve shape = {res.curve.shape}")
    print(f"best_x 是否在界内: {bool(np.all(res.best_x >= -100) and np.all(res.best_x <= 100))}")
