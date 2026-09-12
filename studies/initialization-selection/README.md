# RFM 初始化选择：计算预算对照试验

作者：Xiao Tian；实验设计、实现和材料整理使用了 Codex 辅助。

本研究接续原来的固定幅度符号干预工作，比较完整响应预测库、随机短探测、逐位贪心短探测和随机完整训练。新试验单独存放，不修改原稿。

完成后优先阅读 `研究报告.md`、`facts.json` 和 `verification/audit.json`。

| 内容 | 路径 |
|---|---|
| 结果前固定的规则、预算和统计方法 | `protocol.md` |
| 全部方向、验证/测试划分、执行顺序 | `plan.json` |
| 结果前代码及原论文文件散列 | `freeze.json` |
| GPU 和软件版本 | `run-environment.json` |
| 预测库全部模型、验证轨迹和二次响应张量 | `banks/p17/`、`banks/p23/` |
| 每次选优、候选验证分数、模型和断点 | `runs/` |
| 所有选优完成后的整体封存 | `selection-seal.json` |
| 逐方向最终数据，含未修改参考 | `evaluation/directions.csv` |
| 每个测试点的原始预测和标签 | `evaluation/*.npz` |
| 候选级数据汇总 | `analysis/candidates.csv` |
| 均值、六项配对比较、成本 | `analysis/summary.csv`、`paired_comparisons.csv`、`costs.csv` |
| 科学图，可用于文稿 | `analysis/*.png`、`*.svg`、`*.pdf` |
| 原实现比对和精确断点续算 | `preflight/checks.json` |
| 八次完整选优重跑及全量核查 | `verification/audit.json` |

独立统计单位是 40 个方向，每个模数 20 个。候选训练、库探测和重跑均不能算成独立方向。40 行未修改参考已包含在随机完整训练的计算中。

这是本地试验材料。文件准备完成不等于已经公开、投稿或通过外部评审。

## 复现

当前 Windows 工作区含原研究 `outputs/rfm-mechanism` 和原稿 `outputs/arxiv-submission`。在此工作区运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\outputs\rfm-init-selection\reproduce.ps1
```

该命令在并列的新目录 `outputs/rfm-init-selection-reproduction` 中重做完整试验；已有目标目录时拒绝覆盖。需要原研究作为实现一致性对照、Ubuntu-24.04 WSL、现有 CUDA Python 环境及 RTX 4070。GPU/库版本变化可能影响近乎并列的数值结果；文中精确一致仅指同机同环境的核查。

仅重新计算统计图表，可在一份完整结果的副本中给旧 `analysis` 目录改名后执行 `src/analyze.py`。分析脚本不会改选优结果。不要删除或改写原始封存结果以制造一致性。

源码衍生部分遵循 GPL-3.0，见 `LICENSE` 与 `NOTICE.md`。
