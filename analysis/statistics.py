"""统计检验工具。

论文里做算法对比，**光比均值是不够的**，必须给出统计显著性证据：

- 两两比较：**Wilcoxon 秩和检验**（Wilcoxon rank-sum test），p < 0.05 视为显著；
- 多算法整体比较：**Friedman 检验**（计算平均排名），再用 Nemenyi 后续检验画 CD 图。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

try:  # scipy 是可选依赖，缺失时仍可计算排名
    from scipy import stats as _stats
except Exception:  # pragma: no cover
    _stats = None


def _check_scipy() -> None:
    if _stats is None:
        raise ImportError(
            "统计检验需要 scipy，请先安装：pip install scipy"
        )


def wilcoxon_rank_sum(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> Dict:
    """Wilcoxon 秩和检验（独立样本，非参数）。

    Parameters
    ----------
    a, b : 两组独立运行的结果（例如某算法在某个函数上 30 次独立运行的最优值）
    alpha : 显著性水平

    Returns
    -------
    dict，含 statistic / p_value / significant / better
    """
    _check_scipy()
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    stat, p = _stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant": bool(p < alpha),
        # 谁更好：均值更小者（最小化问题）
        "better": "a" if a.mean() < b.mean() else ("b" if b.mean() > a.mean() else "tie"),
    }


def friedman_test(score_matrix: np.ndarray) -> Dict:
    """Friedman 检验。

    Parameters
    ----------
    score_matrix : shape = (n_problems, n_algorithms)
        每行是一个测试函数（或一个测试场景），每列是一个算法在该函数上的
        **多次独立运行的平均值**。

    Returns
    -------
    dict，含 mean_ranks / statistic / p_value
        ``mean_ranks`` 越小越好，是论文表格最后一列的"平均排名"。
    """
    m = np.asarray(score_matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("score_matrix 必须是二维数组 (n_problems, n_algorithms)")

    n_problems, n_algs = m.shape
    # 手工计算排名（不依赖 scipy），列方向逐行排名，秩从 1 开始
    ranks = np.vstack([_rank_row(row) for row in m])
    mean_ranks = ranks.mean(axis=0)

    out = {
        "mean_ranks": mean_ranks,
        "n_problems": n_problems,
        "n_algorithms": n_algs,
        "statistic": None,
        "p_value": None,
    }

    if _stats is not None and n_problems > 1 and n_algs > 2:
        stat, p = _stats.friedmanchisquare(*[m[:, j] for j in range(n_algs)])
        out["statistic"] = float(stat)
        out["p_value"] = float(p)

    return out


def _rank_row(row: np.ndarray) -> np.ndarray:
    """对一行数值升序排名，数值相同取平均秩（处理并列）。"""
    order = np.argsort(row, kind="mergesort")
    ranks = np.empty(len(row), dtype=float)
    ranks[order] = np.arange(1, len(row) + 1, dtype=float)
    # 处理并列：相同值取平均秩
    values = row[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def summarize(runs: np.ndarray) -> Dict[str, float]:
    """把一组独立运行结果汇总成论文表格需要的统计量。"""
    runs = np.asarray(runs, dtype=float)
    return {
        "mean": float(np.mean(runs)),
        "std": float(np.std(runs, ddof=1)) if runs.size > 1 else 0.0,
        "best": float(np.min(runs)),
        "worst": float(np.max(runs)),
        "median": float(np.median(runs)),
    }


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """论文表格标准写法：``1.2345E+00 (4.5678E-01)``。"""
    return f"{mean:.{decimals}E} ({std:.{decimals}E})"


def paired_wilcoxon_vs_best(
    results_by_algo: Dict[str, Sequence[float]],
    proposed: str,
    alpha: float = 0.05,
) -> List[Dict]:
    """把"本文算法 vs 其他每个算法"的检验一次性算完，输出可直接进表格的行。"""
    rows = []
    for name, values in results_by_algo.items():
        if name == proposed:
            continue
        r = wilcoxon_rank_sum(results_by_algo[proposed], values, alpha=alpha)
        rows.append(
            {
                "compare": f"{proposed} vs {name}",
                "p_value": r["p_value"],
                "significant": r["significant"],
                "winner": proposed if r["better"] == "a" else name,
            }
        )
    return rows
