"""PSO 挑战题验收测试 —— 检查你自己写的 tutorials/my_pso.py 是否合格。

先做这两步：
    copy tutorials\\my_pso_TEMPLATE.py tutorials\\my_pso.py
    （然后补完 my_pso.py 里的 TODO 1~5）

再运行：
    python tutorials\\pso_step3_challenge.py

本文件只做检查，不教你写代码。全部 9 项 PASS 才算过关。
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import numpy as np

_CACHE = Path(__file__).resolve().parents[1] / ".mplcache"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tutorials"))

from benchmarks.classic import rastrigin, shifted_rotated, sphere  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    mark = "[PASS]" if ok else "[FAIL]"
    print(f"  {mark} {name}" + (f"   {detail}" if detail else ""))


print("=" * 78)
print("PSO 挑战题验收")
print("=" * 78)

# ---------------------------------------------------------------------
# 检查 0：文件是否存在、能否导入
# ---------------------------------------------------------------------
try:
    from my_pso import PSO  # type: ignore
except ImportError as e:
    print(f"\n[无法导入 my_pso] {e}")
    print("\n请先执行：")
    print("    copy tutorials\\my_pso_TEMPLATE.py tutorials\\my_pso.py")
    print("然后补完里面的 TODO 1~5，再运行本脚本。")
    raise SystemExit(1)
except NotImplementedError as e:
    print(f"\n[还没写完] {e}")
    print("\n请打开 tutorials\\my_pso.py 把 TODO 补完，再运行本脚本。")
    raise SystemExit(1)

DIM = 30
MAXFES = 30_000

# ---------------------------------------------------------------------
# R1 / R2：接口与返回字段
# ---------------------------------------------------------------------
print("\n[R1/R2] 接口与返回结构")
try:
    opt = PSO(pop_size=50, max_fes=MAXFES, seed=0)
    res = opt.optimize(sphere, DIM, -100.0, 100.0)
    has_fields = all(hasattr(res, f) for f in ("best_x", "best_f", "curve", "nfes"))
    check("R1 接口可调用", True)
    check("R2 返回含 best_x/best_f/curve/nfes", has_fields)
except NotImplementedError as e:
    print(f"\n[还没写完] {e}\n请补完 tutorials\\my_pso.py 的 TODO。")
    raise SystemExit(1)
except Exception as e:
    print(f"\n[运行出错] {type(e).__name__}: {e}")
    raise SystemExit(1)

# ---------------------------------------------------------------------
# R3：收敛曲线形状与单调性
# ---------------------------------------------------------------------
print("\n[R3] 收敛曲线")
curve = np.asarray(res.curve, dtype=float)
check("curve 是二维数组", curve.ndim == 2, f"shape={curve.shape}")
check("curve 有 2 列 (FES, 最优值)", curve.shape[1] == 2 if curve.ndim == 2 else False)
mono = bool(np.all(np.diff(curve[:, 1]) <= 1e-12)) if len(curve) > 1 else False
check("第二列单调不增", mono, "记录的是历史最优，所以只能变好不能变差")

# ---------------------------------------------------------------------
# R4：FES 记账
# ---------------------------------------------------------------------
print("\n[R4] 函数评价次数记账")
pop = 50
ok_fes = res.nfes <= MAXFES + pop
check("nfes 不超过 max_fes + 一轮", ok_fes, f"nfes={res.nfes}, max_fes={MAXFES}")

# ---------------------------------------------------------------------
# R5：可复现性
# ---------------------------------------------------------------------
print("\n[R5] 可复现性")
a = PSO(pop_size=50, max_fes=10_000, seed=42).optimize(sphere, DIM, -100.0, 100.0)
b = PSO(pop_size=50, max_fes=10_000, seed=42).optimize(sphere, DIM, -100.0, 100.0)
same = abs(a.best_f - b.best_f) < 1e-12 and np.allclose(a.best_x, b.best_x)
check("同 seed 结果完全一致", same, f"{a.best_f:.6e} vs {b.best_f:.6e}")

# ---------------------------------------------------------------------
# R6：边界合法性
# ---------------------------------------------------------------------
print("\n[R6] 解在边界内")
in_bounds = bool(np.all(res.best_x >= -100.0) and np.all(res.best_x <= 100.0))
check("best_x 在 [lb, ub] 内", in_bounds)

# ---------------------------------------------------------------------
# R7：三个测试函数上的精度
# ---------------------------------------------------------------------
print(f"\n[R7] 性能测试（维度={DIM}，最大评价次数={MAXFES}，3 个随机种子取均值）")
f_shift, _, _ = shifted_rotated(sphere, DIM, shift=60.0, seed=1)

for label, (func, lb, ub, thr) in {
    "Sphere (原始)":   (sphere, -100.0, 100.0, 1e-2),
    "Sphere (平移)":   (f_shift, -100.0, 100.0, 1.0),
    "Rastrigin":       (rastrigin, -5.12, 5.12, 100.0),
}.items():
    t0 = time.perf_counter()
    vals = [PSO(pop_size=50, max_fes=MAXFES, seed=s).optimize(func, DIM, lb, ub).best_f
            for s in range(3)]
    mean = float(np.mean(vals))
    check(f"{label} 均值 < {thr:g}", mean < thr,
          f"实测 {mean:.4e}  (耗时 {time.perf_counter()-t0:.1f}s)")

# ---------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------
n_pass = sum(1 for _, ok, _ in results if ok)
n_total = len(results)
print("\n" + "=" * 78)
print(f"结果：{n_pass}/{n_total} 项通过")
print("=" * 78)
if n_pass == n_total:
    print("""
全部通过。接下来你可以：
  1. 打开 algorithms/pso.py，对比你的实现和仓库版本的差异
  2. 用我的算法文件替换仓库算法：把 my_pso.py 的逻辑移植进 algorithms/pso.py
  3. 进 step2 的实验里，把你的 PSO 加进去和另外几个算法比一比
""")
else:
    print("\n未通过的项目请看上面 [FAIL] 行，回到 my_pso.py 对应位置修改。")
    print("提示：最常见的原因是忘了速度限幅、忘了边界处理、或 FES 记账方式不对。")
