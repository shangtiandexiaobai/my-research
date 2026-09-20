"""所有优化算法的统一接口。

约定（务必遵守，否则算法之间无法公平比较）：

1. 目标函数签名固定为 ``func(x: np.ndarray) -> float``，``x`` 是一维向量；
2. **所有**函数评价必须通过 ``self._evaluate()``，以便统一统计 FES
   （Function Evaluations，函数评价次数）——这是元启发式算法比较的基准单位；
3. 评价次数用尽（达到 ``max_fes``）立即停止；
4. 结果统一封装为 :class:`OptimizeResult`。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

Objective = Callable[[np.ndarray], float]


@dataclass
class OptimizeResult:
    """一次独立运行的结果。"""

    best_x: np.ndarray
    best_f: float
    #: 收敛曲线，shape = (n_records, 2)，两列依次为 [FES, 当前最优值]
    curve: np.ndarray
    nfes: int
    runtime: float = 0.0
    extra: Dict = field(default_factory=dict)


class Optimizer:
    """优化算法基类。

    Parameters
    ----------
    pop_size : int
        种群规模。
    max_fes : int
        最大函数评价次数（所有算法统一用这个作为停止条件）。
    seed : int, optional
        随机种子。固定它才能复现结果。
    """

    name: str = "Optimizer"

    def __init__(
        self,
        pop_size: int = 50,
        max_fes: int = 300_000,
        seed: Optional[int] = None,
    ) -> None:
        self.pop_size = int(pop_size)
        self.max_fes = int(max_fes)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._nfes = 0
        self._best_f = np.inf
        self._curve: List[tuple] = []

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _reset(self) -> None:
        self._nfes = 0
        self._best_f = np.inf
        self._curve = []

    def _evaluate(self, func: Objective, x: np.ndarray) -> float:
        """评价一次并记账。**所有算法都必须走这里。**"""
        self._nfes += 1
        f = float(func(x))
        if f < self._best_f:
            self._best_f = f
        return f

    def _record(self) -> None:
        """记录当前 FES 下的最优值（用于画收敛曲线）。"""
        self._curve.append((self._nfes, self._best_f))

    def _init_population(self, dim: int, lb, ub) -> np.ndarray:
        """在 [lb, ub] 上均匀随机初始化种群。"""
        return self.rng.uniform(lb, ub, size=(self.pop_size, dim))

    @staticmethod
    def _clip(x: np.ndarray, lb, ub) -> np.ndarray:
        """边界处理：越界截断。"""
        return np.clip(x, lb, ub)

    def _max_iter(self) -> int:
        """按 FES 预算换算的迭代代数。"""
        return max(1, self.max_fes // self.pop_size)

    # ------------------------------------------------------------------
    # 子类必须实现
    # ------------------------------------------------------------------
    def optimize(self, func: Objective, dim: int, lb, ub) -> OptimizeResult:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 收尾
    # ------------------------------------------------------------------
    def _finalize(self, pop: np.ndarray, fitness: np.ndarray, t0: float) -> OptimizeResult:
        idx = int(np.argmin(fitness))
        curve = np.asarray(self._curve, dtype=float)
        if curve.size == 0:
            curve = np.zeros((0, 2))
        else:
            curve = curve.reshape(-1, 2)
        return OptimizeResult(
            best_x=pop[idx].copy(),
            best_f=float(fitness[idx]),
            curve=curve,
            nfes=self._nfes,
            runtime=time.perf_counter() - t0,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.name}(pop_size={self.pop_size}, max_fes={self.max_fes}, seed={self.seed})"
