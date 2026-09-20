"""差分进化算法（Differential Evolution, DE）。

参考文献
--------
Storn R., Price K. Differential evolution - a simple and efficient
heuristic for global optimization over continuous spaces.
Journal of Global Optimization, 1997.

实现要点
--------
- 变异策略：``DE/rand/1``，即 v = x_r1 + F * (x_r2 - x_r3)；
- 交叉策略：二项式交叉（binomial crossover），交叉概率 CR；
- 选择策略：贪婪选择，子代不优则不替换（保证种群不退化）。
"""

from __future__ import annotations

import time

import numpy as np

from .base import OptimizeResult, Optimizer


class DE(Optimizer):
    name = "DE"

    def __init__(
        self,
        pop_size: int = 50,
        max_fes: int = 300_000,
        seed=None,
        F: float = 0.5,
        CR: float = 0.9,
    ) -> None:
        super().__init__(pop_size, max_fes, seed)
        self.F = F
        self.CR = CR

    def optimize(self, func, dim, lb, ub) -> OptimizeResult:
        self._reset()
        t0 = time.perf_counter()

        pop = self._init_population(dim, lb, ub)
        fit = np.array([self._evaluate(func, x) for x in pop])
        self._record()

        for _ in range(self._max_iter()):
            if self._nfes >= self.max_fes:
                break
            for i in range(self.pop_size):
                # 随机选 3 个不同于 i 的个体
                candidates = self.rng.choice(
                    np.delete(np.arange(self.pop_size), i), size=3, replace=False
                )
                r1, r2, r3 = candidates
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])
                mutant = self._clip(mutant, lb, ub)

                cross = self.rng.random(dim) < self.CR
                if not cross.any():
                    cross[self.rng.integers(dim)] = True
                trial = np.where(cross, mutant, pop[i])

                f_trial = self._evaluate(func, trial)
                if f_trial < fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial

            self._record()

        return self._finalize(pop, fit, t0)
