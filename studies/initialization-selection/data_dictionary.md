# 数据字典

输入为两个长度 p 的 one-hot 向量拼接，标签为 (a+b) mod p。训练包含所有 a≠b 的有序对（p17 为272个，p23为506个）；对角线输入的固定验证/测试划分见 plan.json。数据完全可由这些整数重建，无外部数据集。

## evaluation/directions.csv

- `p`：模数；`seed`：生成初始交叉块方向的种子，是配对统计单位。
- `method`：bank / random_probe / greedy_probe / random_full；unmodified 是 random_full 已训练的 code0 参考。
- `chosen_code`：符号翻转二进制编码，0为原方向。第k位对应傅里叶坐标k+1；坐标0固定。
- `test_acc`：最终留出点正确比例，范围0–1；报告中乘100显示百分比。
- `test_correct`、`test_n`：正确点数、测试点数（11或15）。
- `test_mean_margin`：正确类分数减最大错误类分数的点均值。
- `test_normalized_margin`：每个点的上述差除以该点各类别中心化分数的RMS，再对点求平均。不同于 bank 张量的平均对齐 profile margin。
- `validation_acc`、`validation_normalized_margin`：最终第59轮验证指标。短探测的实际选优依据是 candidates.csv 中第10轮指标，而非这里的最终验证值。
- `validation_test_gap`：最终验证准确率减测试准确率。
- `online_fits`、`online_updates`：在线核拟合次数、AGOP更新次数；每条完整轨迹60次拟合59次更新。
- `charged_fits`：bank加上完整构建库成本的1/20，其余等于在线次数。unmodified的60只是其已包含的描述性工作量，不能重复求总和。
- `online_seconds`：同步CUDA的在线墙钟时间，含计算和保存模型的I/O；不含进程启动、数据/张量加载。
- `charged_seconds`：bank加上构建库墙钟时间的1/20；其余等于在线秒数。unmodified留空，避免重复计时。
- `unique_candidates`：实际训练的候选轨迹数量，bank在线为1。bank枚举的预测分数数目另见candidates.csv，不能当成已训练的候选。
- `peak_cuda_allocated_bytes`：PyTorch峰值分配量，不含显示桌面或其他进程，不是整卡总显存。
- `final_kernel_condition`：最终训练核矩阵的2范数条件数。
- `relative_solve_residual`：‖Kα−Y‖F/‖Y‖F；小残差本身不证明病态系统有小前向误差。
- `near_top_ties_relative_1e8`：最大与第二大类别分数差 ≤ 1e−8×类别中心化RMS的测试点数，是近并列诊断而非剔除规则。
- `source`：实际模型的相对路径。

## 数组

`selected.npz` 包含 m0、最终m、alpha、训练核k、index=59 和60轮validation_history；`selected_probe.npz` 保存选定短探测的第10轮状态，续算恢复其m、alpha、k。

`evaluation/p*-s*-*.npz` 包含 predictions（测试点×p）、targets（同形状one-hot标签）、diagonal_indices、最终validation_predictions。所有 test_acc 均可从 predictions.argmax(1)==targets.argmax(1) 复算。

`banks/p*/tensor.npz` 中 q 的维度为p×d×d，d=(p−1)/2；baseline 是平均对齐后中心化的基线输出。候选只使用 aᵀQ_c a 的归一化正确偏移间隔排序，未使用旧研究的拟合校准。

## 分析

`paired_comparisons.csv` 的差为 bank 减竞争方法，单位百分点。置信区间是方向层配对bootstrap，p值是双侧配对符号翻转，六项比较使用Holm校正。单项区间没有多重比较校正。未显著不能解释为等效。

`costs.csv` 的 break_even_queries 是用本次实测延迟/预算外推的盈亏平衡查询数，假设每查询搜索投入保持此次设定；不是其他预算下重新验证过的精度结果。
