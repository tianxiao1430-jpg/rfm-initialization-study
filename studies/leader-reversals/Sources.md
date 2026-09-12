# 来源与分析口径

本次仅分析本地原研究档案，无外部数据。

- 原清单：../rfm-mechanism/results/all_runs.csv，844条运行；主要分组来自frozen/confirmation_plan.json，80条。
- 原始输出：每条运行的final.npz/history及data.npz；标签、每步指标与配置均核对。
- 输出快照：source_manifest.json记录4224项文件的原始散列；coverage.json记录覆盖和复算误差。
- 主要口径：原确认方向；逐测试点与平均剖面分开；并列集合被严格超过才算失去领先。
- 本次属于事后探索；无新训练、无中途干预、无外部复现、无新颖性检索。
