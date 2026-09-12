# 数据字典

## CSV

- `p`：模数；`seed`：方向随机种子。确定性operator阶段的seed用于频率索引，不能算随机样本。
- `phase`：discovery（探索）、probe（6次评估的早期探测）、operator（确定性基底）、confirmation（预留方向）、intervention（干预）、boundary（边界）。
- `kind`：cross、diag、circ、residual、none、mode、mode_pair、mode_difference或custom。custom的`amount`指向initializations中的冻结矩阵文件。
- `acc` / `actual_acc`：实测测试准确率，0–1。未乘100；1/p表示只有一个测试点正确，并不表示有意义的泛化。
- `predicted_acc`：冻结模型预测的最终准确率，0–1。
- `error` / `absolute_error`：预测准确率与实测准确率差的绝对值，0–1。论文乘100后称“百分点”。
- `constant_acc`：40个探索方向平均准确率，作为固定常数预测。
- `mse`：对所有测试点与全部one-hot类别分数计算的均方误差。
- `mean_margin` / `min_margin`：正确类别分数减去最高错误类别分数，跨测试点取均值/最小值。单位是原始输出分数，未标准化。
- `relative_residual`：核拟合的||Kα−Y||F/||Y||F；有岭项时使用实际拟合矩阵。
- `symmetry_defect`：||M−PMP||F/||M||F，P交换两个输入块。
- `epsilon`：初始化扰动相对I的Frobenius范数。M₀=I+epsilon√(2p)E，||E||F=1。
- `bandwidth`、`formula`：code时核分母2×bandwidth²；paper时核分母bandwidth。
- `centering`：是否从每个样本的输入梯度减去样本平均梯度。
- `circulant_fraction`：cross块中循环投影的Frobenius能量占比；其他初始化标量定义见src/components.py。
- `signal`：二次预测剖面中，偏移0与最大其他偏移之间的分数差，除以剖面RMS；正负不变，不是概率。
- `profile_cosine`：预测与观测的平均类别偏移剖面的余弦；平均剖面相似不保证每个测试点分类相同。
- `elapsed_s`：每次运行内的墙钟计时，包含运行内I/O，不等于纯GPU时间或整个研究工时。

`intervention.csv`为宽表：每行一个原始方向，base_acc为未干预结果；phase_best/worst是有利/不利符号；remove为移除循环成分并重新归一化。control0/1/2是预先固定的随机对照；control_mean先在方向内平均，effect为干预减该平均。

## 每次运行的NPZ/JSON

- `config.json`：完整实测配置。
- `environment.json`：软件、GPU、精度设置、执行源码与自定义初始化散列。
- `data.npz`：`pairs`形状[p²,2]，`train_mask`形状[p²]，以及训练样本散列。没有从外部下载的数据。
- `final.npz`：`m0,m`形状[2p,2p]；`pred,y`形状[p,p]；`history`形状[T,p,p]。测试点顺序(a,a)，a=0…p−1；类别顺序0…p−1。T通常60，probe阶段6。
- `states.npz`：`steps`与`matrices`对应t=0,1,2,5,10,59中实际存在的时刻；`agop0`为第一次矩阵更新开方前的AGOP。不是每一步都保存了矩阵，但每一步都保存了输出。
- `metrics.jsonl`：一行一轮指标；`summary.json`给出状态、记录数、末轮与异常字段。
- `frozen/quadratic_p*.npz`：响应张量q形状[p,d,d]，正弦基basis形状[d,2p,2p]，d=(p−1)/2；epsilon为建立张量的有限探测幅度。不是精确解析Hessian。
- `initializations/*.npz`：冻结干预方向E，形状[2p,2p]，全矩阵Frobenius范数1。
- `results/readout_*.npz`：固定最终M的更高精度读出；stored、refined及y。不能据此宣称高精度训练。

使用`numpy.load(path, allow_pickle=False)`读取NPZ。没有需要执行的序列化pickle对象。高精度readout文件含Linux long double类型，应在所附WSL环境中读取；普通final、states和模型文件均使用标准float64。
