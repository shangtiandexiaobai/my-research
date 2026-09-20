# PSO 学习阶梯（tutorials/）

配套理论文档：`../docs/PSO学习指南.md`

**先读指南，再动代码。** 不要一上来就抄算法。

---

## 阶梯总览（建议 2 周）

| 阶段 | 时间 | 做什么 | 文件 | 过关标准 |
|---|---|---|---|---|
| **Step 1** 看懂 | 第 1–2 天 | 跑通极简版，手改三行核心 | `pso_step1_minimal.py` | 能凭记忆重写核心 3 行 |
| **Step 2** 解剖 | 第 3–6 天 | 四个受控实验，理解每个部件 | `pso_step2_anatomy.py` | 能解释每个参数调大调小的后果 |
| **Step 3** 从零写 | 第 7–12 天 | 自己实现并通过验收 | `my_pso_TEMPLATE.py` → `my_pso.py` + `pso_step3_challenge.py` | 9 项验收全 PASS |
| **Step 4** 对着改 | 第 13–14 天 | 对比 `algorithms/pso.py`，写实验报告 | `../docs/experiment_log.md` | 写下 4 个结论 |

---

## Step 1 · 看懂（1–2 天）

```bash
python tutorials/pso_step1_minimal.py
```

产出：一张收敛曲线图 `results/figures/tutorial/step1_minimal_pso.png`

**关键**：这个文件故意不用仓库框架，100 行摊平。核心只有三行：

```python
v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)   # 三股力合成速度
x = np.clip(x + v, lb, ub)                                   # 用速度更新位置
f = np.array([sphere(p) for p in x])                         # 评价
```

做完文件末尾列出的 6 个动手练习再往下走。

---

## Step 2 · 解剖（3–6 天）

```bash
python tutorials/pso_step2_anatomy.py
```

产出：4 张对比图 + 4 张数据表
- `exp1_inertia_weight.png` —— 惯性权重 w
- `exp2_c1_c2.png` —— 认知/社会系数
- `exp3_population_size.png` —— 种群规模（固定 FES）
- `exp4_velocity_clamping.png` —— 速度限幅（**会复现仓库原版 PSO 的缺陷**）

**这个脚本最重要的不是代码，是每张图下面的【观察要点】。**

写完实验报告：把四个实验的结论用你自己的话写进 `../docs/experiment_log.md`。

---

## Step 3 · 从零写（7–12 天）

```powershell
copy tutorials\my_pso_TEMPLATE.py tutorials\my_pso.py
# 打开 my_pso.py，补完 TODO 1~5
python tutorials\pso_step3_challenge.py
```

验收 9 项（接口 / 返回字段 / 曲线形状 / 曲线单调性 / FES 记账 / 可复现性 / 边界合法 / 三个函数的精度）。

**纪律：写的时候不要打开 `algorithms/pso.py`。** 写完再对比，差异才是你的收获。

---

## Step 4 · 对着改（13–14 天）

1. `diff` 一下你的 `my_pso.py` 和仓库的 `algorithms/pso.py`，列出所有差异
2. 逐条问自己：谁对？为什么？
3. 在 `../docs/experiment_log.md` 写下四条结论：
   - 惯性权重 w 的作用与取值建议
   - c1 / c2 的作用与取值建议
   - 种群规模在固定 FES 下的影响
   - 速度限幅为什么不可省
4. 把结论发给导师，作为第一次正式汇报的内容

---

## 常见卡点

| 症状 | 原因 |
|---|---|
| `ModuleNotFoundError: my_pso` | 没执行 copy，`tutorials/my_pso.py` 不存在 |
| 结果每次都不一样 | 没固定 seed，或没用 `default_rng(seed)` |
| Sphere 上卡在 1e1 量级 | 缺速度限幅，或 w 太大配 c1=c2=2 |
| 曲线上下波动 | 记录的是**当前位置**而不是**历史最优** |
| `SyntaxError: 'return' outside function` | 把 `raise NotImplementedError` 删了但没补实现 |
