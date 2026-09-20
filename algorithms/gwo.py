"""灰狼优化算法（Grey Wolf Optimizer, GWO）。

参考文献
--------
Mirjalili S., Mirjalili S.M., Lewis A. Grey wolf optimizer.
Advances in Engineering Software, 2014.

实现要点
--------
- 社会等级：alpha（最优）、beta（次优）、delta（第三优）引导整个种群；
- 系数向量 A 随 a 从 2 线性递减到 0，实现探索到开发的过渡；
- |A| > 1 时偏探索，|A| < 1 时偏开发。
"""

from __future__ import annotations

import time

import numpy as np

from .base import OptimizeResult, Optimizer


class GWO(Optimizer):
    name = "GWO"

    def __init__(self, pop_size: int = 50, max_fes: int = 300_000, seed=None) -> None:
        super().__init__(pop_size, max_fes, seed)

    def optimize(self, func, dim, lb, ub) -> OptimizeResult:
        self._reset()
        t0 = time.perf_counter()

        pop = self._init_population(dim, lb, ub)
        fit = np.array([self._evaluate(func, x) for x in pop])

        order = np.argsort(fit)
        alpha, beta, delta = pop[order[0]].copy(), pop[order[1]].copy(), pop[order[2]].copy()
        f_alpha, f_beta, f_delta = fit[order[0]], fit[order[1]], fit[order[2]]
        self._record()

        max_iter = self._max_iter()
        for it in range(max_iter):
            if self._nfes >= self.max_fes:
                break

            a = 2.0 - 2.0 * it / max_iter

            def step(leader):
                r1 = self.rng.random((self.pop_size, dim))
                r2 = self.rng.random((self.pop_size, dim))
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                return leader - A * np.abs(C * leader - pop)

            x1 = step(alpha)
            x2 = step(beta)
            x3 = step(delta)
            pop = self._clip((x1 + x2 + x3) / 3.0, lb, ub)

            fit = np.array([self._evaluate(func, x) for x in pop])

            order = np.argsort(fit)
            for rank, target in enumerate(("alpha", "beta", "delta")):
                idx = order[rank]
                if rank == 0 and fit[idx] < f_alpha:
                    alpha, f_alpha = pop[idx].copy(), fit[idx]
                elif rank == 1 and fit[idx] < f_beta:
                    beta, f_beta = pop[idx].copy(), fit[idx]
                elif rank == 2 and fit[idx] < f_delta:
                    delta, f_delta = pop[idx].copy(), fit[idx]

            self._record()

        # 用三匹头狼中最好的作为最终解
        best = min([(f_alpha, alpha), (f_beta, beta), (f_delta, delta)], key=lambda t: t[0])
        final_pop = pop.copy()
        final_fit = fit.copy()
        final_pop[0], final_fit[0] = best[1], best[0]

        return self._finalize(final_pop, final_fit, t0)
