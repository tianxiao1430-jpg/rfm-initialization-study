"""Build a readable research note and figures from verified original score histories."""
from pathlib import Path
import collections
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from analyze import OUT, ROOT, tolerances, sequences, write_csv, dump


def read(name):
    return list(csv.DictReader((OUT/name).open(encoding='utf-8')))


def load(phase,name):
    with np.load(ROOT/'results'/phase/name/'final.npz',allow_pickle=False) as z:
        return z['history']


def main():
    runrows=read('run_summary.csv')
    events=read('events.csv')
    primary=[r for r in runrows if r['phase']=='confirmation']
    timeline=[]
    # Check every possible anchor, so "stable after step12" is not interpolation.
    for run in primary:
        p=int(run['p']);h=load('confirmation',run['name'])
        for view in ['point','profile']:
            scores=[s for v,a,c,s in sequences(h,p) if v==view]
            eligible=np.zeros(60,int);displaced=np.zeros(60,int)
            unique=np.zeros(60,int)
            for s in scores:
                rms,tol=tolerances(s)
                for k in range(60):
                    w=s[k].max()-s[k]<=tol[k]
                    eligible[k]+=w.sum()<p;unique[k]+=w.sum()==1
                    if k<59 and w.sum()<p:
                        d=s[k+1:,~w].max(1)-s[k+1:,w].max(1)
                        displaced[k]+=bool(np.any(d>tol[k+1:]))
            for k in range(60):
                timeline.append(dict(name=run['name'],p=p,view=view,step=k,sequences=len(scores),
                    eligible=int(eligible[k]),unique=int(unique[k]),displaced=int(displaced[k])))
    write_csv('primary_all_steps_by_run.csv',timeline)
    step_summary=[]
    for p in [17,23]:
        for view in ['point','profile']:
            for t in range(60):
                rr=[r for r in timeline if r['p']==p and r['view']==view and r['step']==t]
                step_summary.append(dict(p=p,view=view,step=t,runs=len(rr),
                    runs_with_displacement=sum(r['displaced']>0 for r in rr),
                    sequences=sum(r['sequences'] for r in rr),
                    displaced_sequences=sum(r['displaced'] for r in rr),
                    eligible_sequences=sum(r['eligible'] for r in rr)))
    write_csv('primary_all_steps_summary.csv',step_summary)
    assert all(r['runs_with_displacement']==0 for r in step_summary if r['step']>=12)
    assert all(r['runs_with_displacement']==0 for r in step_summary if r['step']>=10 and r['view']=='profile')
    # Exact event timing, including non-confirmation groups and partial horizons.
    timing=collections.Counter((e['phase'],int(e['p']),e['view'],e['event'],int(e['step'])) for e in events)
    write_csv('event_timing.csv',[dict(phase=ph,p=p,view=v,event=ev,step=t,count=n)
        for (ph,p,v,ev,t),n in sorted(timing.items())])
    # Matched amplitude comparison: all 20 directions, including the 17 without reversals.
    matched=[]
    for p in [17,23]:
        for seed in range(100,110):
            normal=next(r for r in runrows if r['phase']=='confirmation' and r['name']==f'p{p}_s{seed}_cross')
            larger=next(r for r in runrows if r['phase']=='boundary' and r['name']==f'p{p}_s{seed}_eps03')
            configs=[]
            for phase,r in [('confirmation',normal),('boundary',larger)]:
                configs.append(json.loads((ROOT/'results'/phase/r['name']/'config.json').read_text()))
            assert configs[0]['epsilon']==.01 and configs[1]['epsilon']==.03
            assert {k:v for k,v in configs[0].items() if k!='epsilon'}=={k:v for k,v in configs[1].items() if k!='epsilon'}
            matched.append(dict(p=p,seed=seed,default_name=normal['name'],larger_name=larger['name'],
                default_demotions=int(normal['point_correct_to_wrong']),
                larger_demotions=int(larger['point_correct_to_wrong']),
                default_final_accuracy=float(normal['final_acc']),larger_final_accuracy=float(larger['final_acc'])))
    write_csv('matched_amplitude_comparison.csv',matched)
    # Example rules: latest primary correctness flip; first sorted boundary demotion;
    # first sorted latest boundary correctness flip. All witnesses remain in events.csv.
    pc=[e for e in events if e['phase']=='confirmation' and e['view']=='point' and e['event']=='wrong_to_correct']
    bd=[e for e in events if e['phase']=='boundary' and e['view']=='point' and e['event']=='correct_to_wrong']
    bc=[e for e in events if e['phase']=='boundary' and e['view']=='point' and e['event']=='wrong_to_correct']
    order=lambda e:(e['name'],int(e['test_a']))
    examples=[sorted([e for e in pc if int(e['step'])==max(int(x['step']) for x in pc)],key=order)[0],
              sorted(bd,key=order)[0],
              sorted([e for e in bc if int(e['step'])==max(int(x['step']) for x in bc)],key=order)[0]]
    traces=[];example_checks=[]
    for index,e in enumerate(examples):
        p=int(e['p']);a=int(e['test_a']);label=2*a%p;s=load(e['phase'],e['name'])[:,a,:]
        rms,tol=tolerances(s)
        other=s.copy();other[:,label]=-np.inf
        m=s[:,label]-other.max(1)
        for t in range(60):
            traces.append(dict(example=index+1,phase=e['phase'],name=e['name'],p=p,test_a=a,truth=label,step=t,
                raw_argmax=int(s[t].argmax()),correct_score=float(s[t,label]),best_wrong_score=float(other[t].max()),
                margin=float(m[t]),rms=float(rms[t]),tolerance=float(tol[t]),normalized_margin=float(m[t]/rms[t])))
        b,t=int(e['previous_step']),int(e['step'])
        for relative in [1e-10,1e-8,1e-6,1e-4]:
            _,tt=tolerances(s,relative)
            assert abs(m[b])>tt[b] and abs(m[t])>tt[t] and m[b]*m[t]<0
        example_checks.append(dict(e,min_endpoint_margin_over_tolerance=min(abs(m[b])/tol[b],abs(m[t])/tol[t]),
                                   all_four_tolerances_confirm=True))
    write_csv('example_traces.csv',traces)
    dump('examples.json',example_checks)
    # Verify all 27 boundary demotions under the same more-conservative tolerances.
    for e in bd:
        p=int(e['p']);a=int(e['test_a']);label=2*a%p;s=load(e['phase'],e['name'])[:,a,:]
        m=s[:,label]-np.delete(s,label,axis=1).max(1)
        b,t=int(e['previous_step']),int(e['step'])
        _,tt=tolerances(s,1e-4)
        assert m[b]>tt[b] and m[t]<-tt[t]
    fonts=[Path('C:/Windows/Fonts/msyh.ttc'),Path('C:/Windows/Fonts/simhei.ttf')]
    font=next((p for p in fonts if p.exists()),None)
    if font:
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'#fbfaf6','axes.facecolor':'#fbfaf6','savefig.facecolor':'#fbfaf6'})
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(12.6,5.6),sharey=True)
    for ax,p,n in zip(axs,[17,23],[50,30]):
        for view,color,label,style in [('point','#176b75','至少一道题的领先答案被反超','-'),
                                        ('profile','#c05a32','平均得分剖面的领先答案被反超','--')]:
            rr=[r for r in step_summary if r['p']==p and r['view']==view and r['step']<=18]
            ax.plot([r['step'] for r in rr],[100*r['runs_with_displacement']/r['runs'] for r in rr],
                    color=color,linestyle=style,marker='o',markersize=3,label=label)
        ax.set(title=f'p = {p} · {n} 个初始化方向',xlabel='观察第几步的领先答案',xlim=(0,18),ylim=(-3,105))
        ax.set_xticks([0,1,3,5,8,10,12,15,18]);ax.grid(axis='y',alpha=.2)
        k10=next(r for r in step_summary if r['p']==p and r['view']=='point' and r['step']==10)
        ax.annotate(f"第10步后：{k10['runs_with_displacement']}/{n} 个方向仍有反超",xy=(10,100*k10['runs_with_displacement']/n),
                    xytext=(8,35),fontsize=10,arrowprops={'arrowstyle':'->','color':'#333333'})
    axs[0].set_ylabel('之后至第59步仍发生反超的方向比例（%）')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=1,frameon=False,fontsize=10)
    fig.suptitle('前几步的领先答案经常改变；第10–12步后才趋于稳定',fontsize=16,y=.98)
    fig.text(.5,.02,'已保存的80个原确认方向 · 每个方向等权 · 事后描述性结果 · 并列阈值 rtol=1e-8 · 完整观察至59步',ha='center',fontsize=9,color='#555555')
    fig.subplots_adjust(bottom=.28,top=.86,wspace=.12)
    for ext in ['png','svg']:fig.savefig(figdir/f'01_leader_stability.{ext}',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(11.5,9),sharex=True)
    titles=['默认参数：本组最晚的错转对出现在第12步',
            '增大扰动：第7步已经答对，第8步又被错误答案反超',
            '关闭梯度中心化：直到第18步才从错变对']
    for index,(ax,e,title) in enumerate(zip(axs,examples,titles)):
        rr=[r for r in traces if r['example']==index+1]
        ts=np.array([r['step'] for r in rr]);mm=np.array([r['normalized_margin'] for r in rr])
        ax.plot(ts,mm,color='#176b75' if index!=1 else '#c05a32',lw=2)
        ax.axhline(0,color='#555555',lw=.9);ax.axvline(int(e['step']),color='#888888',ls=':',lw=1)
        ax.set_title(title,loc='left',fontsize=12)
        ax.text(.99,.06,f"{e['name']}  ·  测试点({e['test_a']},{e['test_a']})",transform=ax.transAxes,ha='right',fontsize=9)
        ax.set_ylabel('正确答案领先幅度\n（除以中心化分数RMS）',fontsize=10)
        ax.grid(axis='y',alpha=.17)
        # Full-range plots conceal small but decisive zero crossings; show their scale.
        t=int(e['step']);window=(ts>=t-1)&(ts<=t+2)
        zoom=ax.inset_axes([.57,.27,.39,.43])
        zoom.plot(ts[window],mm[window],marker='o',markersize=4,
                  color='#176b75' if index!=1 else '#c05a32')
        zoom.axhline(0,color='#555555',lw=.8)
        zoom.axvline(t,color='#888888',ls=':',lw=.8)
        zoom.set_xticks(ts[window]);zoom.tick_params(labelsize=8)
        zoom.set_title('反转附近放大（纵轴同一定义）',fontsize=9)
        zoom.grid(axis='y',alpha=.15)
    axs[-1].set_xlabel('已完成的特征更新次数（第0步为初始拟合）')
    axs[-1].set_xlim(0,59);axs[-1].set_xticks([0,5,8,10,12,18,30,40,50,59])
    fig.suptitle('正值：正确答案领先；负值：错误答案领先',fontsize=16,y=.98)
    fig.text(.5,.012,'插图按“最晚反转 / 首个参数反例”选取；全部事件另存CSV。边界案例不计入主要80方向。',ha='center',fontsize=10,color='#555555')
    fig.subplots_adjust(top=.91,bottom=.105,hspace=.5)
    for ext in ['png','svg']:fig.savefig(figdir/f'02_reversal_examples.{ext}',dpi=180)
    plt.close(fig)
    ce=[e for e in pc]
    concentrated=sum(int(e['step']) in [8,9] for e in ce)
    table=[]
    for k in [1,3,5,8,9,10,11,12]:
        cells=[]
        for view in ['point','profile']:
            for p in [17,23]:
                r=next(r for r in step_summary if r['p']==p and r['view']==view and r['step']==k)
                cells.append(f"{r['runs_with_displacement']}/{r['runs']}")
        table.append('| '+str(k)+' | '+' | '.join(cells)+' |')
    late=[r for r in primary if int(r['point_last_leader_replacement'] or 0)>10]
    late_text='、'.join(r['name'] for r in late)
    # Phase summary preserves the partial probe horizon.
    phase_table=[]
    for phase in ['confirmation','discovery','operator','intervention','boundary','probe']:
        rr=[r for r in runrows if r['phase']==phase]
        maxstep=max(int(r['point_last_leader_replacement'] or 0) for r in rr)
        demotions=sum(int(r['point_correct_to_wrong']) for r in rr)
        affected=sum(int(r['point_correct_to_wrong'])>0 for r in rr)
        phase_table.append(f"| {phase} | {len(rr)} | {rr[0]['horizon']} | {maxstep} | {affected} / {demotions} |")
    primary_point_seq=[r for r in read('sequence_summary.csv') if r['phase']=='confirmation' and r['view']=='point']
    profile_seq=[r for r in read('sequence_summary.csv') if r['phase']=='confirmation' and r['view']=='profile']
    finite_endpoint_wrong=sum(r['final_correct_state']=='-1' for r in primary_point_seq)
    result=f'''# 训练早期领先的答案，会不会后来被反超？

Xiao Tian · 原始RFM轨迹的事后分析 · {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}

**会。主要80个初始化方向中，每个方向在第1步和第5步之后都至少有一道题换过明确领先的答案。第10步之后仍有5个方向、7道题发生反超；最后一次在第12步。第12步及之后的每个时刻都检查过，至第59步没有再发现领先集合被整体反超。**

这里的“反超”可以是错误答案换成另一个错误答案，也可以是正确答案后来超过错误答案。默认参数的80个方向里，没有发现明确的“先对后错”。但参数边界中有这种反例，见下文。

## 1. 做了什么

直接读取原研究全部844份final.npz中的history，不重训。主要80个confirmation方向按p17的50个方向、p23的30个方向分别统计。其余发现、基底、干预、参数变化分开报告。

这里的confirmation仅是原研究的目录/阶段名称。本次分析使用的是已经看过的旧数据，属于生成假设的事后分析，不构成新的独立确认。

完整轨迹观察0–59步；0表示第一次核拟合、尚未进行AGOP特征更新，1表示已进行一次更新。50条早期probe只观察到5步，它们没有被混入59步的稳定性结论。

每个得分向量把距最高分不超过tol的类别视为并列领先。默认tol=max(1e-8×类别中心化RMS, 64×float64机器精度×最大绝对分数)。早期领先集合中所有成员都被集合外答案严格超过，才算该集合失去领先。没有可区分领先者时单独记录。这避免把近乎并列的错误类别下标跳动当成确定的反超，但阈值不是全训练误差的数学上界。

同时看两层数据：逐测试点的答案；以及把每道题正确答案对齐到偏移0后的平均得分剖面。平均剖面可能比最后一道具体题更早稳定，不能替代逐点分析。

## 2. 早期答案后来是否改变：全部主要方向的实测计数

表中分子是“观察第k步后，至59步还发生过反超的初始化方向数”，分母是该模数全部方向。逐点列只要求该方向至少一道题发生反超；所有这些锚点都具有可区分的领先者。

| 观察时刻k | 逐点 p17 | 逐点 p23 | 平均剖面 p17 | 平均剖面 p23 |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

![主要方向的领先稳定性](figures/01_leader_stability.png)

平均剖面最晚在第10步完成最后一次领先替换。逐点最晚为p17第12步、p23第11步。第10步后仍变化的5个方向：{late_text}。

主要80个方向共有1540条逐点序列，其中74个方向、1254个测试点出现明确的错转对；没有明确的对转错。1254次错转对中，{concentrated}次发生在第8或9步（按记录计算为{100*concentrated/1254:.1f}%）。这些是相关测试点的描述性计数，不是1540个独立样本。

稳定也不等于学会：终点仍有{finite_endpoint_wrong}个测试点明确由错误答案领先。有些轨迹稳定在错误答案上。平均剖面有68/80条出现错转对，但不能据此推算逐点正确率。

## 3. 一个具体例子

p17、seed111、题目(13,13)，正确类别为9：第11步错误类别3仍领先；第12步正确类别9反超。正确分数减最高错误分数由-8.29115e-8变成+7.62873e-7，对应默认阈值约4.15e-12，两个时刻的差距均远大于阈值。这是可在原始分数中直接核查的反转。

该方向的平均剖面已在第10步转为正确答案领先，因此“平均已经稳定”不能推出“每道题都稳定”。

## 4. 参数边界：不能写成普遍的锁定定律

在原先已经做过的20组配对幅度实验中，仅把epsilon从0.01增至0.03，其余配置逐项核对一致。默认幅度下这20条轨迹没有明确的先对后错；较大幅度下，3条轨迹共27个测试点出现先对后错：

- p17_s107_eps03：3个点；
- p23_s102_eps03：14个点；
- p23_s105_eps03：10个点。

这些对转错发生在第8–10步，并且在更保守的1e-4相对并列口径下，27个反例仍都成立。以p17_s107_eps03的(4,4)为例，第7步正确类别8领先，第8步被错误类别15超过；正确间隔由+0.000165724变成-0.002193143。不能把默认设置的单向错转对现象外推到更大扰动。

关闭梯度中心化的已有实验中，反转还能推迟到第18步，例如p17_s100_uncentered的(9,9)。这不是额外新运行的结果。

![反转轨迹与参数反例](figures/02_reversal_examples.png)

| 原实验阶段 | 轨迹数 | 观察终点 | 逐点最晚领先替换步 | 先对后错：轨迹数 / 点事件数 |
|---|---:|---:|---:|---:|
{chr(10).join(phase_table)}

上述阶段不能合并当作844个独立方向。干预是配对资料，operator是确定性基底，discovery还包含早期重复；probe的观察窗口更短。

## 5. 并列敏感性与独立核查

主要80方向的正确性事件在rtol=1e-10、1e-8、1e-6、1e-4四种口径下均保持：p17为666次错转对，p23为588次错转对，对转错均为0。默认口径的逐点领先集合替换共5030次，原始argmax跳动5031次；更保守口径会减少可确认的早期错误类别之间的换位，因此不把换位总数当成无阈值依赖的物理常数。完整敏感性表另附。

全部844条history都存在，47940行每步准确率、MSE、平均/最小间隔已从原始分数复算，最大绝对差约1.11e-16；所有末步history与pred完全一致。独立脚本以另一种对齐与循环写法复算全部80条主要轨迹的锚点结果和正确性反转，并逐条核验全部56452个输出事件的前后分数证据。还测试了并列桥接、错误类别近并列、反超后恢复、初始不确定和全程并列五类构造情形。

原输入4224项文件散列核对无变化（包含原数据和本次分析方案/脚本记录）。这验证的是对已保存输出的分析，没有把复算一致性冒充独立训练、因果机制或严格高精度证明。

另以独立写法核对全部80条主要轨迹、两种视图、0–59步的每一个观察锚点，共38400项整数比较；20组幅度配对和插图全部180行原始分数也已核对。

## 6. 审阅后补充：终点分层与重复翻转

从原始history独立重算每个点的明确正确/错误状态，并与先前逐序列表核对。所有1540个点在第0步都明确错误。按第59步状态分层：

| 终点状态 | 测试点数 | 曾明确正确 | 正确性从未翻转 | 恰好翻转一次 | 翻转两次及以上 |
|---|---:|---:|---:|---:|---:|
| 正确 | 1254 | 1254 | 0 | 1254 | 0 |
| 错误 | 286 | 0 | 286 | 0 | 0 |
| 不确定 | 0 | 0 | 0 | 0 | 0 |

因此，1254次错转对确实对应1254个不同测试点，每点恰好一次；286个终点错点从未明确由正确答案领先。“对转错为0”在终点错点组里不能被解释为“答对之后很稳定”，因为这些点从未答对过；在终点正确组里，则观察到了首次明确答对后没有再明确变错。这是按已知终点作出的描述性分层，不是方向性机制的独立证据。错误类别之间仍可能多次换位，不能把“正确性只翻一次”说成“领先答案只换一次”。

第8–9步聚集的机制还不能从当前档案直接解释。全部80条主要轨迹的度量矩阵仅保存第0、1、2、5、10、59步，没有第8或9步矩阵。逐步输出分数和对称性缺陷标量仍在，但标量不能替代循环扇区能量或完整AGOP状态。第5至10步的稀疏快照也不足以定位第8–9步的能量阈值跨越；本次不据此宣称整体性相变。

若继续追踪这个假设，需要在明确标为探索性回放的实验中补存该窗口的每步矩阵，并用后续新方向验证所提出的预测量。当前结果与短训练基线较强相一致，但尚未证明后续约50步只是分数细化，也没有证明稳定性就是选择器表现的因果解释。

## 7. 对论文意味着什么

- **“第一步领先答案已经决定终点”不符合这些轨迹。** 第1步的领先者在全部80个方向中都至少有一道题被后续答案取代。
- **“默认配置很早就出现预测标签稳定”得到有限样本支持。** 平均剖面在10步后、逐点在12步后稳定，观察仅到59步。这可以解释为什么较短训练值得作为强基线，但本次没有重跑选择器，也没有证明其预算最优。
- **“早期预测标签稳定”仍不等于因果锁定。** 后续得分大小和模型度量仍可变化；没有进行中途状态干预。
- **“二次模型为什么准确”仍未解释。** 最终响应张量本来就由训练至终点的基底轨迹构建，不能把终点拟合准确解释为第一步机制已经得到证明。
- **参数边界是真实反例。** 较大扰动能先对后错，关闭中心化能延迟反转。进一步研究应解释这种差别，并在新方向确认，不把这份事后检查包装成独立发现验证。

## 数据与复跑

- [逐轨迹汇总](run_summary.csv)、[主要方向逐步结果](primary_all_steps_by_run.csv)、[逐序列结果](sequence_summary.csv)。
- [全部反转事件](events.csv)、[反转发生时刻](event_timing.csv)、[各锚点逐测试点结果](anchor_sequences.csv)。
- [幅度配对对照](matched_amplitude_comparison.csv)、[插图原始分数](example_traces.csv)、[插图事件及阈值检查](examples.json)。
- [并列敏感性](sensitivity_anchors.csv)、[正确性敏感性](sensitivity_events.csv)、[原始来源与散列](source_manifest.json)。
- [方案](protocol.md)、[分析脚本](analyze.py)、[独立核查](verify_analysis.py)、[图表与本文生成](build_report.py)。
- [逐步独立核查](verify_timeline.py)、[原分析核查结果](verification.json)、[逐步与报告核查结果](timeline-verification.json)。
- [终点分层](endpoint_stratification.csv)、[逐点正确性历史汇总](endpoint_point_history.csv)、[度量快照覆盖](saved_metric_coverage.csv)、[审阅补充复算脚本](review_addendum.py)、[补充来源散列](review_input_manifest.json)。

在本目录运行 `python -X utf8 analyze.py`、`python -X utf8 verify_analysis.py`、`python -X utf8 review_addendum.py`、`python -X utf8 build_report.py`、`python -X utf8 verify_timeline.py`。依赖numpy和matplotlib；原始数据位于相邻的rfm-mechanism目录。原论文正文未纳入本次结果；本补充分析的公开状态以项目首页为准。

本次分析过程中使用了OpenAI Codex。
'''
    (OUT/'研究报告.md').write_text(result,encoding='utf-8')
    dump('report_facts.json',{'primary_runs':80,'primary_points':1540,'primary_correctness_flips':1254,
        'primary_correct_to_wrong':0,'flips_at_8_or_9':concentrated,
        'latest_primary_point_reversal':12,'latest_primary_profile_reversal':10,
        'after10_runs':len(late),'after10_points':sum(r['displaced_sequences'] for r in step_summary if r['view']=='point' and r['step']==10),
        'matched_larger_amplitude_runs':20,'matched_larger_amplitude_demoted_runs':sum(r['larger_demotions']>0 for r in matched),
        'matched_larger_amplitude_demotions':sum(r['larger_demotions'] for r in matched),
        'boundary_latest_step':18,'all_steps_checked':60,'figures':2})
    (OUT/'Sources.md').write_text('# 来源与分析口径\n\n本次仅分析本地原研究档案，无外部数据。\n\n'
        '- 原清单：../rfm-mechanism/results/all_runs.csv，844条运行；主要分组来自frozen/confirmation_plan.json，80条。\n'
        '- 原始输出：每条运行的final.npz/history及data.npz；标签、每步指标与配置均核对。\n'
        '- 输出快照：source_manifest.json记录4224项文件的原始散列；coverage.json记录覆盖和复算误差。\n'
        '- 主要口径：原确认方向；逐测试点与平均剖面分开；并列集合被严格超过才算失去领先。\n'
        '- 本次属于事后探索；无新训练、无中途干预、无外部复现、无新颖性检索。\n',encoding='utf-8')
    print(json.dumps(json.loads((OUT/'report_facts.json').read_text()),ensure_ascii=False))


if __name__=='__main__':
    main()
