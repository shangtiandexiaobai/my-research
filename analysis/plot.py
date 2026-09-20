"""绘图工具（论文级配图）。

说明：图内文字统一使用英文，避免中文字体缺失导致显示为方块。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

# matplotlib 需要一个可写的缓存目录。某些环境下用户目录不可写，
# 这里统一把缓存指到仓库内的 .mplcache/，避免因权限问题直接崩溃。
_MPL_CACHE = Path(__file__).resolve().parents[1] / ".mplcache"
try:
    _MPL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
except Exception:  # pragma: no cover
    pass

try:
    import matplotlib

    matplotlib.use("Agg")  # 无界面环境下也能出图
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _check() -> None:
    if plt is None:
        raise ImportError("绘图需要 matplotlib，请先安装：pip install matplotlib")


def plot_convergence(
    curves: Dict[str, np.ndarray],
    title: str = "Convergence Curve",
    xlabel: str = "FES",
    ylabel: str = "Best fitness (log scale)",
    save_path: Optional[str] = None,
    log_scale: bool = True,
):
    """画收敛曲线。

    Parameters
    ----------
    curves : dict
        ``{算法名: curve}``，curve 的 shape 为 (n, 2)，列为 [FES, 最优值]。
    """
    _check()
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    for name, curve in curves.items():
        curve = np.asarray(curve, dtype=float).reshape(-1, 2)
        if curve.size == 0:
            continue
        ax.plot(curve[:, 0], curve[:, 1], label=name, linewidth=1.4)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_boxplot(
    data: Dict[str, Sequence[float]],
    title: str = "Distribution of Best Fitness",
    ylabel: str = "Best fitness",
    save_path: Optional[str] = None,
    log_scale: bool = True,
):
    """画箱线图，展示 30 次独立运行的稳定性。"""
    _check()
    names = list(data.keys())
    values = [np.asarray(data[n], dtype=float) for n in names]

    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(names)), 4.4), dpi=150)
    try:
        # matplotlib >= 3.9 用 tick_labels
        bp = ax.boxplot(values, tick_labels=names, patch_artist=True, widths=0.55)
    except TypeError:
        # 兼容旧版本
        bp = ax.boxplot(values, labels=names, patch_artist=True, widths=0.55)
    for patch in bp["boxes"]:
        patch.set_facecolor("#d6e4f0")
        patch.set_edgecolor("#2b4c7e")
    for median in bp["medians"]:
        median.set_color("#c0392b")
        median.set_linewidth(1.6)
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_rank_bar(mean_ranks: Dict[str, float], title: str = "Average Rank", save_path: Optional[str] = None):
    """画平均排名柱状图（越低越好）。"""
    _check()
    names = list(mean_ranks.keys())
    ranks = [mean_ranks[n] for n in names]
    order = np.argsort(ranks)
    names = [names[i] for i in order]
    ranks = [ranks[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(names)), 4.0), dpi=150)
    ax.bar(names, ranks, color="#4a7ebb")
    for i, r in enumerate(ranks):
        ax.text(i, r + 0.05, f"{r:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("Mean rank (lower is better)")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
