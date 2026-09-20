"""经典基准测试函数（Classic Benchmark Functions）。

论文里通常按"单峰 / 多峰"分类：

- **单峰（Unimodal）**：只有一个全局最优，用来检验**收敛精度/开发能力**
- **多峰（Multimodal）**：大量局部最优，用来检验**全局搜索能力/跳出局部最优**

⚠️ 正式投稿实验请使用 **CEC 系列基准集**（CEC2017 / CEC2020 / CEC2022），
它们包含平移、旋转、混合、复合函数，能有效防止"算法在简单函数上刷分"。
本模块的经典函数仅用于**开发阶段快速调试**。
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

#: 经典函数的默认搜索范围
DEFAULT_BOUNDS = (-100.0, 100.0)


def sphere(x: np.ndarray) -> float:
    """Sphere（单峰），全局最优 f(0,...,0) = 0。"""
    return float(np.sum(x**2))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock 香蕉函数（单峰但等高线弯曲，山谷狭窄难收敛），最优值 0。"""
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2))


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin（高度多峰，大量规则分布的局部最优），最优值 0。"""
    n = x.size
    return float(10.0 * n + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    """Ackley（多峰，最优值 0）。"""
    n = x.size
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
    term2 = -np.exp(np.sum(np.cos(2.0 * np.pi * x)) / n)
    return float(term1 + term2 + 20.0 + np.e)


def griewank(x: np.ndarray) -> float:
    """Griewank（多峰，最优值 0）。"""
    n = x.size
    i = np.arange(1, n + 1)
    return float(np.sum(x**2) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))) + 1.0)


def schwefel(x: np.ndarray) -> float:
    """Schwefel（多峰，全局最优点远离原点，容易把人引到错误方向），最优值 0。"""
    n = x.size
    return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def shifted_rotated(
    func: Callable[[np.ndarray], float],
    dim: int,
    shift: float = 0.0,
    seed: int = 0,
) -> Tuple[Callable[[np.ndarray], float], np.ndarray, np.ndarray]:
    """构造平移 + 旋转版本，避免算法"恰好"利用原点附近的对称性。

    Returns
    -------
    (wrapped_func, shift_vector, rotation_matrix)
    """
    rng = np.random.default_rng(seed)
    o = rng.uniform(-shift, shift, size=dim) if shift else np.zeros(dim)
    m = np.linalg.qr(rng.normal(size=(dim, dim)))[0]

    def wrapped(x: np.ndarray) -> float:
        return func(m @ (x - o))

    return wrapped, o, m


#: 函数注册表：名称 -> (函数, 搜索下界, 搜索上界, 类型)
CLASSIC_FUNCTIONS: Dict[str, Tuple[Callable, float, float, str]] = {
    "Sphere": (sphere, -100.0, 100.0, "unimodal"),
    "Rosenbrock": (rosenbrock, -30.0, 30.0, "unimodal"),
    "Rastrigin": (rastrigin, -5.12, 5.12, "multimodal"),
    "Ackley": (ackley, -32.0, 32.0, "multimodal"),
    "Griewank": (griewank, -600.0, 600.0, "multimodal"),
    "Schwefel": (schwefel, -500.0, 500.0, "multimodal"),
}


def get_function(name: str, dim: int, shift: bool = False, seed: int = 0):
    """按名称取测试函数。

    Parameters
    ----------
    name : str
        函数名，见 :data:`CLASSIC_FUNCTIONS`。
    dim : int
        维度。
    shift : bool
        是否使用平移+旋转版本（更接近 CEC 基准的形式）。
    seed : int
        平移向量与旋转矩阵的随机种子，**同一个种子的不同算法必须一致**。

    Returns
    -------
    (func, lb, ub)
    """
    if name not in CLASSIC_FUNCTIONS:
        raise KeyError(f"未知函数 {name!r}，可选：{list(CLASSIC_FUNCTIONS)}")
    base, lb, ub, _kind = CLASSIC_FUNCTIONS[name]
    if shift:
        func, _, _ = shifted_rotated(base, dim, shift=0.3 * (ub - lb), seed=seed)
        return func, lb, ub
    return base, lb, ub


def describe() -> str:
    """打印函数清单，便于快速查看。"""
    lines = [f"{'名称':<12}{'类型':<12}{'范围'}"]
    for name, (_f, lb, ub, kind) in CLASSIC_FUNCTIONS.items():
        lines.append(f"{name:<12}{kind:<12}[{lb}, {ub}]")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(describe())
