"""粒子群优化算法（Particle Swarm Optimization, PSO）。

参考文献
--------
Kennedy J., Eberhart R. Particle swarm optimization.
Proceedings of ICNN'95, 1995.

实现要点
--------
- 惯性权重 w 线性递减（0.9 → 0.4），前期偏探索、后期偏开发；
- c1（个体认知）与 c2（社会认知）取经典值 2.0；
- 速度未做显式限幅，靠位置截断保证可行性。
"""

from __future__ import annotations

import numpy as np

from .base import OptimizeResult, Optimizer


class PSO(Optimizer):
    name = "PSO"

    def __init__(
        self,
        pop_size: int = 50,
        max_fes: int = 300_000,
        seed=None,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
    ) -> None:
        super().__init__(pop_size, max_fes, seed)
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2

    def optimize(self, func, dim, lb, ub) -> OptimizeResult:
        self._reset()
        import time

        t0 = time.perf_counter()

        pop = self._init_population(dim, lb, ub)
        vel = self.rng.normal(0.0, 0.1 * (np.asarray(ub) - np.asarray(lb)), size=pop.shape)

        fit = np.array([self._evaluate(func, x) for x in pop])
        pbest = pop.copy()
        pbest_f = fit.copy()

        g_idx = int(np.argmin(pbest_f))
        gbest = pbest[g_idx].copy()
        self._record()

        for it in range(self._max_iter()):
            if self._nfes >= self.max_fes:
                break
            w = self.w_max - (self.w_max - self.w_min) * it / self._max_iter()
            r1 = self.rng.random(pop.shape)
            r2 = self.rng.random(pop.shape)

            vel = w * vel + self.c1 * r1 * (pbest - pop) + self.c2 * r2 * (gbest - pop)
            pop = self._clip(pop + vel, lb, ub)

            fit = np.array([self._evaluate(func, x) for x in pop])

            better = fit < pbest_f
            pbest[better] = pop[better]
            pbest_f[better] = fit[better]

            g_idx = int(np.argmin(pbest_f))
            if pbest_f[g_idx] < self._best_f:
                gbest = pbest[g_idx].copy()

            self._record()

        return self._finalize(pbest, pbest_f, t0)
