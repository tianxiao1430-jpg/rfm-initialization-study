"""Generate the manuscript and checkable claims directly from audited results."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/'results'/name).read_text())
def pct(x):return f'{100*x:.1f}'
def ci(d):return f"{pct(d['mean'])} [{pct(d['ci95'][0])}, {pct(d['ci95'][1])}]"
def main():
    c=read('confirmation_analysis.json');i=read('intervention_analysis.json');b=read('boundary_analysis.json');n=read('numerical_check.json');a=read('audit.json');s=read('summary.json')
    assert a['passed'];im=i['metrics']['all'];freeze=json.loads((ROOT/'frozen/execution_seal.json').read_text())
    facts=dict(title='固定频率幅度下的符号相互作用：对称固定点分割中RFM初始化的预测与干预',
               status='本地研究论文初稿；未公开、未投稿、未获外部评审',audit=a,study=s,confirmation=c['metrics'],intervention=i['metrics'],boundary=b['metrics'],
               provenance=dict(frozen_at=freeze['timestamp_utc'],forecast_file='frozen/forecasts.json',per_direction_predictions='results/confirmation.csv',
                               paired_interventions='results/intervention.csv',parameter_cases='results/boundary.csv',all_runs='results/all_runs.csv',all_steps='results/all_steps.csv'),
               distinctions=['局部对称性选择规则是条件命题','最终二次响应是有限幅度数值近似','新模数需要重新建立响应张量','符号选择使用已知模加任务结构','未作整条轨迹任意精度训练'])
    (ROOT/'facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
    predrows=[]
    for p in [17,23]:
        m=c['metrics'][str(p)];rr=[r for r in c['rows'] if r['p']==p];always1=float(np.mean([1-r['actual_acc'] for r in rr]))
        predrows.append(f"| {p} | {m['n']} | {ci(m['mae'])} | {pct(m['constant_mae']['mean'])} | {pct(always1)} | {pct(m['success_agreement'])}% |")
    introws=[]
    for label,key,control in [('有利符号','phase_best','phase_best_control_mean'),('不利符号','phase_worst','phase_worst_control_mean'),('移除循环分量','remove','remove_control_mean')]:
        stat=im[key+'_effect'];lo,hi=stat['seed_cluster_ci95']
        introws.append(f"| {label} | {pct(im['mean_acc'][key])}% | {pct(im['mean_acc'][control])}% | {pct(stat['mean'])} [{pct(lo)}, {pct(hi)}] |")
    boundaryrows=[]
    names={'default':'默认','paper':'论文核分母2.5','uncentered':'关闭中心化','paper_uncentered':'二者同时改变','bw15':'σ=1.5','bw20':'σ=2.0','bw30':'σ=3.0','eps001':'ε=.001','eps03':'ε=.03'}
    for setting,label in names.items():
        mm=[next(r for r in b['metrics'] if r['p']==p and r['setting']==setting) for p in [17,23]]
        boundaryrows.append(f"| {label} | {pct(mm[0]['mean_acc'])}% | {pct(mm[1]['mean_acc'])}% | {pct(mm[0]['mae'])} / {pct(mm[1]['mae'])} |")
    lo,hi=im['best_minus_worst']['seed_cluster_ci95'];nummax=max(r['max_abs_prediction_difference'] for r in n['results'])
    text=f'''# {facts['title']}

2026-09-06 · 本地研究初稿 · 作者及机构待研究者确认 · 尚未公开或同行评审

## 摘要

在模加任务的交换固定点分割中，同样打破交换对称性的RFM初始化可以产生完全不同的泛化结果。我们进一步分解初始化的cross块，区分循环分量、非循环残差及循环频率之间的符号关系。局部等变性推导表明，在导数存在时，diag奇扰动与非循环cross不能在线性阶注入循环cross通道；解析微分数值计算显示第一步对该通道具有较高增益。随后构造有限幅度二次响应张量，在未参与校准的p17、p23随机方向上，最终准确率的预测平均误差分别为4.2和6.6个百分点。对预先固定的30个方向，保持循环部分逐频幅度和非循环残差不变，仅选择不同频率符号，平均准确率由不利符号的4.0%变为有利符号的97.6%。等改动距离随机对照支持这一差异具有结构选择性。结果同时受核带宽约定和梯度中心化影响。本工作提供特定RFM设置下可复核的预测与干预证据，不主张完整泛化理论或通用初始化方法。

## 1. 问题与贡献范围

研究问题是：**当训练数据与扰动总范数相同，为什么一些交换奇初始化有效，另一些失败？能否在运行新方向前预测，并通过控制初始化内部结构改变结果？**

循环特征、循环投影和RFM泛化已有[Mallinar等（2025）](https://proceedings.mlr.press/v267/mallinar25a.html)研究；改变结构初始化及其群轨道泛化已有[Tomàs等（2026，§4、附录F）](https://arxiv.org/html/2604.00316v2)讨论。两层网络中的相位对齐和频率竞争也已有[He等（2026）](https://arxiv.org/html/2602.16849v1)分析，循环embedding初始化见[Gu等（2025，附录E）](https://arxiv.org/html/2504.03162v2)。因此，“循环结构有用”“相位会影响学习”均不作为本轮新贡献。

候选增量限定为三点：固定点分割中初始化块的局部选择规则；可在新随机方向上检验的循环二次响应模型；保持逐频幅度的符号干预及匹配对照。详细覆盖比较见[related_work.md](related_work.md)，并不构成全球新颖性证明。

## 2. 方法与数据

输入x=[e_a,e_b]，标签(a+b) mod p；训练所有a≠b，测试所有a=b。默认Gaussian核为exp(-d²_M/12.5)，无岭插值，中心化AGOP平方根更新；FP64、ε=.01、共60次评估（t=0…59）。初值M₀=I+ε√(2p)E，||E||F=1。实验在RTX 4070与WSL中实际运行。

把交换奇扰动写作E=[[A,B],[-B,-A]]，A对称、B反对称。循环平均C=𝒞B与残差R=B-C正交。循环cross有d=(p-1)/2个实正弦基坐标a_k。最终输出先按类别偏移c-2a对齐，得到平均分数剖面。

| 阶段 | 运行数 | 作用 |
|---|---:|---|
| 初始探索与旧结果复现 | 82 | 40个新方向、循环分解及重复核查 |
| 五步循环探测 | 50 | 保留早期弱预测的否定证据 |
| 确定性响应基底及检查 | 112 | p17/p23张量、差频与半幅度检查 |
| 冻结预测确认 | 80 | p17新50方向、p23新30方向 |
| 干预及匹配对照 | 360 | 固定30方向，每个12种干预 |
| 参数边界 | 160 | 两个模数各10方向、8种额外设置 |
| 合计 | {s['runs']} | {s['records']:,}条逐轮记录 |

这些运行不是844个独立样本。预测的统计单位是每个模数内的随机方向；同一方向上的测试点、重复运行和三个随机对照不能当成额外独立种子。跨模数复用相同整数seed可能共享随机数，因此合并干预结果另外按seed聚类核查。

## 3. 机制：允许增长的通道与最终分类分开

交换对称性推出固定点预测F_t(ε)=F_t(-ε)。在局部导数存在的假设下，共同平移和取负的等变性进一步禁止diag或非循环cross在线性阶产生循环cross分量。计算第一步导数，在p17/p23/p29的循环方向上增益约1.66/1.69/1.71，而所查残差方向约.89–.91、diag约.24–.36。后续轨迹实际放大了初始循环分量。

![局部选择与轨迹](figures/01_channel_selection.png)

但**放大并不保证正确分类**。旧p17的纯循环方向仍有2/5完全失败。初次核拟合的diag二阶响应也不为零：因此“第一拟合二阶项只由cross主导”的说法被否定。证明条件、解析微分公式、PSD平方根的限制及反例见[机制推导.md](机制推导.md)。

## 4. 二次响应与冻结预测

以ε₀=.002探测单基方向和两基相加方向，建立数值张量Q_c，预测平均输出信号为ε² aᵀQ_c a。Q的交叉项保留不同频率之间的符号关系。再把正确偏移0相对其他偏移的标准化分数差，用40个p17探索方向作单调校准，预测最终准确率。

张量不是解析闭式解：p17需要37次完整RFM基底/基线运行，p23需要67次；另外8次只检验近似。新模数重新建立了张量，但未用p23随机方向的结果重新校准。因此不是零计算跨模数迁移。构造张量不使用测试标签；分数解释及符号选择使用已知模加任务规则，须与未知任务的盲预测区分。

冻结时间：{freeze['timestamp_utc']}。逐方向预测、矩阵、计划及代码散列均在确认运行前写入[frozen](frozen/forecasts.csv)。这是可核查的本地冻结记录，不是独立平台的预注册或优先权证明。

| 模数 | 新方向数 | 预测MAE及95%区间（百分点） | 探索均值常数MAE | 总预测100%的MAE | ≥90%成功分类一致率 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(predrows)}

![未参与拟合的方向](figures/02_frozen_prediction.png)

预测平均剖面的余弦范围为p17的0.99857–0.999996、p23的0.99765–0.99996；相对L2误差中位数约1.45%和4.16%。这支持剖面形状近似，但不意味着每个测试点的分类都正确。p23成功分类一致率83.3%，仅略高于“全部会成功”的80%，此处不能夸大。

一个强替代方案是直接训练到第10步：它预测最终准确率的MAE只有0.12/0.87个百分点，优于本模型。二次张量的优势在于建立后无需为每个方向训练RFM、并能解释符号干预；小批量使用时不应宣称它更省计算。

## 5. 同幅度、不同符号的干预

事先固定p17 seed100–119、p23 seed100–109。保持非循环残差、扰动总范数和所有循环|a_k|不变，枚举相对符号，分别选择模型预测最有利与最不利的方向。另作循环分量移除实验。每个干预有3个匹配总范数和改动距离的随机对照；符号对照还匹配循环能量与原残差，但不保证逐频幅度相同。

| 干预 | 实测平均准确率 | 对应3个随机对照的平均准确率 | 与对照差及seed聚类95%区间（百分点） |
|---|---:|---:|---:|
{chr(10).join(introws)}

有利与不利符号的直接配对差为{pct(im['best_minus_worst']['mean'])}个百分点，seed聚类95%区间[{pct(lo)}, {pct(hi)}]。这不是通过增加循环能量得到的差异。三组匹配对照比较的预设Monte Carlo符号置换加Holm校正p值约1.5e-5；按共享seed聚类的敏感性检查仍支持差异，完整数值在results/intervention_analysis.json。

![配对干预和对照](figures/03_sign_interventions.png)

![事先固定的seed100示例](figures/05_fixed_energy_example.png)

同时保留失败：原准确率≤10%的合格方向只有4个，有利干预使其中3个达到≥90%；不能宣称普遍救回。原准确率≥90%的20个方向中，不利符号使17个降至≤10%，移除循环分量使20个均降至≤10%。全体方向而非仅成功案例均在[intervention.csv](results/intervention.csv)。

## 6. 成立边界

下表固定每个模数seed100–109，模型保持默认校准，不因参数改变而重拟合。该子集与整套50/30方向的平均准确率不同，不能混用分母。

| 设置 | p17平均准确率 | p23平均准确率 | 冻结预测MAE：p17 / p23（百分点） |
|---|---:|---:|---:|
{chr(10).join(boundaryrows)}

![参数网格](figures/04_parameter_boundaries.png)

论文核约定与关闭中心化同时改变时，默认预测误差增至约27.5/30.5个百分点。这明确限制了结果的参数适用范围。ε=.001/.01/.03和其他带宽点只支持所测网格，不能推出连续参数区域的充分条件。

## 7. 数值与复现核查

20条旧轨迹在初始化、最终矩阵和全60步预测上逐位相同。全部{a['records']:,}条保存分数已由原始预测重新计算；代码、冻结预测、计划、数据分割及矩阵几何全部核查。16个固定最终矩阵以64位尾数long double进行迭代改进，其中2个追加50位十进制LU，正确数量均不变；最大读出差约{nummax:.2e}。

这仅核查保存矩阵上的最终读出，不是全程任意精度训练。PSD平方根附近的数值问题和更大模数不能据此一概排除。完整核查见[audit.json](results/audit.json)与[numerical_check.json](results/numerical_check.json)。

## 8. 论文价值与剩余问题

相较原探索笔记，这轮增加了可审查的局部命题、冻结预测、独立方向验证和保持频率幅度的双向干预。最值得主张的是**特定RFM设置中的符号相互作用能预测并控制成败**。

仍未完成的是：推导最终响应张量的闭式表达或统一误差界；解释跨参数迁移失败；扩大未参与探索的任务/模数；由独立研究者核查新颖性及证明。初稿尚不能保证发表，更不能以大量运行数代替概念贡献。下一项最小外部反馈应是让原作者或相关研究者复现预测与一对同幅度干预；本轮没有向任何人发送材料。

## 数据与执行披露

所有数据入口及字段在[README.md](README.md)和[data_dictionary.md](data_dictionary.md)。[facts.json](facts.json)将论文数字映射到机器可读结果。实验和初稿由Codex在用户授权下完成；基础代码来源、许可和AI辅助披露见[NOTICE.md](NOTICE.md)。
'''
    (ROOT/'研究论文初稿.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(manuscript='研究论文初稿.md',facts='facts.json',characters=len(text)),ensure_ascii=False))
if __name__=='__main__':main()
