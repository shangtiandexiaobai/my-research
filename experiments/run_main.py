"""主实验入口：多算法 × 多测试函数 × 多次独立运行。

用法
----
    python experiments/run_main.py

输出
----
    results/raw/runs.csv          逐次运行的原始结果（每行 = 一次独立运行）
    results/raw/curves/*.npz      收敛曲线原始数据
    results/tables/summary.csv    均值/标准差汇总表
    results/tables/wilcoxon.csv   两两显著性检验（需 scipy）
    results/figures/*.png         收敛曲线、箱线图、平均排名

约定
----
- **同一个函数、同一次运行编号 i**，所有算法使用相同的随机种子（seed=i），
  这是公平比较的标准做法（配对比较）。
- 结果表格一律由脚本生成，禁止手工修改。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# 允许从仓库根目录直接运行本脚本
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.de import DE  # noqa: E402
from algorithms.gwo import GWO  # noqa: E402
from algorithms.pso import PSO  # noqa: E402
from analysis.statistics import friedman_test, summarize, wilcoxon_rank_sum  # noqa: E402
from benchmarks.classic import get_function  # noqa: E402

ALGORITHMS = {"PSO": PSO, "DE": DE, "GWO": GWO}

DEFAULT_CONFIG = {
    "dim": 30,
    "max_fes": 20000,
    "runs": 5,
    "pop_size": 50,
    "functions": ["Sphere", "Rosenbrock", "Rastrigin", "Ackley", "Griewank", "Schwefel"],
    "algorithms": {name: {"pop_size": 50} for name in ALGORITHMS},
    "output_dir": "results",
    "save_curves": True,
    "make_plots": True,
}


def _coerce(value: str):
    """把字符串转成 int / float / bool / None，失败则原样返回。"""
    low = value.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def _simple_yaml_load(text: str) -> dict:
    """极简 YAML 子集解析器（只支持本项目的 config.yaml 语法）。

    支持：注释、``key: value``、一层嵌套映射、``- item`` 列表。
    有了它，**即使没安装 PyYAML 也能读取 config.yaml**，避免环境问题阻塞实验。
    """
    root: dict = {}
    stack = [(-1, root)]  # (缩进, 容器)

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        i += 1
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        # 回退到正确的父容器
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]

        if stripped.startswith("- "):
            if isinstance(container, list):
                container.append(_coerce(stripped[2:].strip()))
            continue

        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key, value = key.strip(), value.strip()

        if value == "":
            # 空值：看下一非空行决定它是列表还是嵌套映射
            nxt = ""
            for cand in lines[i:]:
                c = cand.split("#", 1)[0].rstrip()
                if c.strip():
                    nxt = c
                    break
            child = [] if nxt.strip().startswith("- ") else {}
            container[key] = child
            stack.append((indent, child))
        else:
            container[key] = _coerce(value)

    return root


def load_config(path: Path) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # 深拷贝
    if not path.exists():
        print(f"[警告] 未找到 {path}，使用内置默认配置")
        return cfg

    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        user_cfg = yaml.safe_load(text) or {}
    except ImportError:
        # 没装 PyYAML 也能正常工作：改用内置极简解析器
        print("[提示] 未安装 PyYAML，已启用内置极简 YAML 解析器读取 config.yaml")
        user_cfg = _simple_yaml_load(text)

    cfg.update({k: v for k, v in user_cfg.items() if v is not None})
    return cfg


def run_one(alg_name: str, func_name: str, cfg: dict):
    """跑一个算法在一个函数上的多次独立运行。"""
    dim = int(cfg["dim"])
    runs = int(cfg["runs"])

    best_values = []
    curves = {}

    for i in range(runs):
        # 同一函数同一 run 编号下，所有算法共享同一个平移/旋转与随机种子
        func, lb, ub = get_function(func_name, dim, shift=True, seed=1000 + i)

        cls = ALGORITHMS[alg_name]
        params = dict(cfg.get("algorithms", {}).get(alg_name, {}))
        pop_size = int(params.pop("pop_size", cfg["pop_size"]))
        opt = cls(pop_size=pop_size, max_fes=int(cfg["max_fes"]), seed=i, **params)

        result = opt.optimize(func, dim, lb, ub)
        if not np.isfinite(result.best_f):
            # 数值溢出等异常情况，记为一个极大值而不是崩溃
            result.best_f = 1e30
        best_values.append(result.best_f)
        if cfg.get("save_curves"):
            curves[f"run{i}"] = result.curve

    return np.asarray(best_values, dtype=float), curves


def main() -> int:
    parser = argparse.ArgumentParser(description="群智能算法对比实验")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--runs", type=int, default=None, help="覆盖独立运行次数")
    parser.add_argument("--max-fes", type=int, default=None, help="覆盖最大评价次数")
    parser.add_argument("--dim", type=int, default=None, help="覆盖维度")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    for key, val in (("runs", args.runs), ("max_fes", args.max_fes), ("dim", args.dim)):
        if val is not None:
            cfg[key] = val

    out_dir = ROOT / cfg["output_dir"]
    (out_dir / "raw" / "curves").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    alg_names = [a for a in cfg.get("algorithms", {}) if a in ALGORITHMS] or list(ALGORITHMS)
    func_names = list(cfg["functions"])

    print("=" * 72)
    print(f"实验设置：dim={cfg['dim']}  max_fes={cfg['max_fes']}  runs={cfg['runs']}  "
          f"pop_size={cfg['pop_size']}")
    print(f"算法：{alg_names}")
    print(f"函数：{func_names}")
    print("=" * 72)

    raw_rows = []          # 逐次运行结果
    summary = {}           # (func, alg) -> 统计量
    curves_store = {}      # (func, alg) -> {run_i: curve}

    for func_name in func_names:
        for alg_name in alg_names:
            t0 = time.perf_counter()
            values, curves = run_one(alg_name, func_name, cfg)
            elapsed = time.perf_counter() - t0

            stats = summarize(values)
            summary[(func_name, alg_name)] = stats
            curves_store[(func_name, alg_name)] = curves

            for i, v in enumerate(values):
                raw_rows.append(
                    {
                        "function": func_name,
                        "algorithm": alg_name,
                        "dim": cfg["dim"],
                        "run": i,
                        "best_f": v,
                    }
                )

            print(
                f"{func_name:<12} {alg_name:<6} "
                f"mean={stats['mean']:.4e}  std={stats['std']:.4e}  "
                f"best={stats['best']:.4e}  ({elapsed:.1f}s)"
            )

    # ---------------- 保存原始结果 ----------------
    raw_path = out_dir / "raw" / "runs.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as f:
        f.write("function,algorithm,dim,run,best_f\n")
        for r in raw_rows:
            f.write(f"{r['function']},{r['algorithm']},{r['dim']},{r['run']},{r['best_f']:.10e}\n")
    print(f"\n[OK] 原始结果 -> {raw_path}")

    with (out_dir / "raw" / "config_used.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # ---------------- 汇总表 ----------------
    summary_path = out_dir / "tables" / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        header = ["function"] + [f"{a}_{k}" for a in alg_names for k in ("mean", "std", "best", "worst")]
        f.write(",".join(header) + "\n")
        for func_name in func_names:
            row = [func_name]
            for alg_name in alg_names:
                s = summary[(func_name, alg_name)]
                row += [f"{s['mean']:.6e}", f"{s['std']:.6e}", f"{s['best']:.6e}", f"{s['worst']:.6e}"]
            f.write(",".join(row) + "\n")
    print(f"[OK] 汇总表   -> {summary_path}")

    # ---------------- 打印论文式表格 ----------------
    print("\n" + "=" * 72)
    print("论文式结果表（均值(标准差)，越小越好）")
    print("=" * 72)
    header = f"{'Function':<12}" + "".join(f"{a:>22}" for a in alg_names)
    print(header)
    print("-" * len(header))
    for func_name in func_names:
        line = f"{func_name:<12}"
        for alg_name in alg_names:
            s = summary[(func_name, alg_name)]
            line += f"{s['mean']:>12.4e}({s['std']:.2e})"
        print(line)

    # ---------------- 平均排名 ----------------
    matrix = np.array(
        [[summary[(f_, a)]["mean"] for a in alg_names] for f_ in func_names], dtype=float
    )
    fr = friedman_test(matrix)
    print("\n平均排名（Friedman，越小越好）")
    for a, r in sorted(zip(alg_names, fr["mean_ranks"]), key=lambda t: t[1]):
        print(f"  {a:<8}{r:.3f}")
    if fr["p_value"] is not None:
        print(f"  Friedman p-value = {fr['p_value']:.4e}")

    # ---------------- 两两显著性检验 ----------------
    if len(alg_names) >= 2:
        try:
            wilcoxon_rows = []
            for func_name in func_names:
                for i in range(len(alg_names)):
                    for j in range(i + 1, len(alg_names)):
                        a, b = alg_names[i], alg_names[j]
                        va = [r["best_f"] for r in raw_rows if r["function"] == func_name and r["algorithm"] == a]
                        vb = [r["best_f"] for r in raw_rows if r["function"] == func_name and r["algorithm"] == b]
                        res = wilcoxon_rank_sum(va, vb)
                        wilcoxon_rows.append((func_name, a, b, res["p_value"], res["significant"], res["better"]))

            wp = out_dir / "tables" / "wilcoxon.csv"
            with wp.open("w", encoding="utf-8", newline="") as f:
                f.write("function,algo_a,algo_b,p_value,significant,better\n")
                for row in wilcoxon_rows:
                    f.write(f"{row[0]},{row[1]},{row[2]},{row[3]:.6e},{row[4]},{row[5]}\n")
            print(f"\n[OK] Wilcoxon 检验 -> {wp}")
        except ImportError as e:
            print(f"\n[跳过] {e}")

    # ---------------- 出图 ----------------
    if cfg.get("make_plots"):
        try:
            from analysis.plot import plot_boxplot, plot_convergence, plot_rank_bar

            for func_name in func_names:
                curves = {}
                for alg_name in alg_names:
                    cs = curves_store[(func_name, alg_name)]
                    if cs:
                        # 取所有 run 的中位轨迹作为代表
                        arrs = [c for c in cs.values() if len(c)]
                        if arrs:
                            min_len = min(len(a) for a in arrs)
                            stacked = np.vstack([a[:min_len, 1] for a in arrs])
                            med = np.median(stacked, axis=0)
                            x = arrs[0][:min_len, 0]
                            curves[alg_name] = np.column_stack([x, med])
                if curves:
                    plot_convergence(
                        curves,
                        title=f"Convergence on {func_name} (D={cfg['dim']})",
                        save_path=str(out_dir / "figures" / f"convergence_{func_name}.png"),
                    )

                box_data = {
                    a: [r["best_f"] for r in raw_rows if r["function"] == func_name and r["algorithm"] == a]
                    for a in alg_names
                }
                plot_boxplot(
                    box_data,
                    title=f"Distribution on {func_name} (D={cfg['dim']}, {cfg['runs']} runs)",
                    save_path=str(out_dir / "figures" / f"boxplot_{func_name}.png"),
                )

            plot_rank_bar(
                dict(zip(alg_names, fr["mean_ranks"])),
                title="Average Rank over all functions",
                save_path=str(out_dir / "figures" / "mean_rank.png"),
            )
            print(f"[OK] 图片     -> {out_dir / 'figures'}")
        except ImportError as e:
            print(f"[跳过] 绘图：{e}")

    print("\n完成。请记得在 docs/experiment_log.md 记录本次实验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
