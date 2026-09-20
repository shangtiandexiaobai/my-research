# my-research

群智能 / 元启发式优化算法的科研代码仓库（广西民族大学 人工智能学院 · 周永权教授课题组方向）。

本仓库的目的不是"存代码"，而是保证**每一篇论文的实验都能被一键复现**。

---

## 目录结构

```
my-research/
├── algorithms/            # 优化算法（统一接口）
│   ├── base.py            #   Optimizer 基类 + OptimizeResult
│   ├── pso.py             #   粒子群
│   ├── de.py              #   差分进化
│   ├── gwo.py             #   灰狼优化
│   └── proposed.py        #   ← 你自己的算法写在这里（待创建）
├── benchmarks/            # 测试函数集
│   └── classic.py         #   Sphere/Rosenbrock/Rastrigin/Ackley/Griewank/Schwefel
├── applications/          # 真实应用（特征选择、图像分割、路径规划…）
├── experiments/           # 实验入口（唯一允许改参数的地方）
│   ├── config.yaml        #   所有实验参数
│   └── run_main.py        #   主实验：多算法 × 多函数 × 多次独立运行
├── analysis/              # 结果分析
│   ├── statistics.py      #   Wilcoxon 秩和检验、Friedman 检验
│   └── plot.py            #   收敛曲线、箱线图
├── results/               # 实验输出（可提交，保证可追溯）
│   ├── raw/               #   逐次运行的原始数据
│   ├── tables/            #   汇总表格（可直接粘进论文）
│   └── figures/           #   论文级配图
├── docs/                  # 过程记录
│   ├── reading_notes.md   #   文献四栏表
│   └── experiment_log.md  #   实验日志
├── papers/                # 手稿
└── requirements.txt
```

## 快速开始

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 跑一遍主实验（默认规模很小，用于验证流程，约 1~2 分钟）
python experiments/run_main.py

# 3. 看结果
#    results/tables/summary.csv   均值/标准差汇总
#    results/figures/*.png        收敛曲线、箱线图
```

## 实验纪律（重要）

1. **所有参数只写在 `experiments/config.yaml`**，不要散落在代码里。
2. **每次运行前设置随机种子**，结果必须可复现。
3. **独立运行次数 ≥ 30**（本仓库默认值调小了，只为快速验证流程，正式实验务必改回 30）。
4. **每个策略单独做消融实验**，证明它自己的贡献。
5. **跑完实验立刻写 `docs/experiment_log.md`**，否则一个月后你必然忘记当时的配置。
6. **不要手工改结果数据**，所有表格必须由脚本从 `results/raw/` 生成。

## 正式实验的推荐设置

| 项目 | 快速验证（默认） | 正式实验（投稿用） |
|---|---|---|
| 维度 | 30 | 10 / 30 / 50 / 100 |
| 最大评价次数 MaxFES | 20 000 | 300 000（或 10 000 × dim） |
| 独立运行次数 | 5 | **30** |
| 测试集 | 6 个经典函数 | CEC2017 / CEC2020 / CEC2022 |
| 对比算法 | 3 个 | 8~12 个（含原版基线 + 近 3 年 SOTA） |
| 统计检验 | 无 | Wilcoxon + Friedman/Nemenyi |

## 提交规范

```
feat: 新增 XXX 算法实现
fix:  修正 DE 边界处理越界问题
exp:  跑完 CEC2017 30维 30次实验，结果存入 results/raw
docs: 更新文献四栏表
refactor: 重构 Optimizer 基类接口
```

## 关联远程仓库（等你准备好账号后再做）

先在 Gitee（国内推荐）或 GitHub 上**新建一个空仓库**（不要勾选自动生成 README），然后：

```bash
# 关联远程
git remote add origin https://gitee.com/<你的用户名>/my-research.git

# 首次推送
git branch -M main
git push -u origin main
```

查看当前远程：

```bash
git remote -v
```

修改远程地址：

```bash
git remote set-url origin <新的仓库地址>
```

> 推送时若要求输入密码，Gitee/GitHub 都已不支持账号密码，需要使用 **个人访问令牌（Personal Access Token）** 代替密码。

## 提交身份（当前是占位值，请尽快改成你自己）

```bash
git config user.name  "你的姓名拼音"
git config user.email "你的学号@stu.gxmzu.edu.cn"
```

如果想对所有仓库生效，加 `--global`。
