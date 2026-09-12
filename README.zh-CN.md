# RFM 初始化研究

作者：**Xiao Tian**。

这是论文、代码及实验数据的公开仓库。论文研究固定频率幅度时，初始化内部
符号关系怎样影响RFM在模加法上的泛化。后续补充实验发现：当前40个新方向上，
随机短训练筛选已达到100%，完整预测库未显示准确率优势。

- [正式整理的论文 PDF](paper/main.pdf)，沿用2026-09-07已确认版本。
- [原论文配套代码与数据](studies/rfm-study/PUBLICATION_README.md)。
- [后续对照实验报告](studies/initialization-selection/研究报告.md)。
- [逐方向结果](studies/initialization-selection/evaluation/directions.csv)。
- [完整数据下载](https://github.com/tianxiao1430-jpg/rfm-initialization-study/releases/tag/v1.0.0)。

后续实验作为独立补充呈现，不能把它与旧论文的实验混成一次预先设计的研究。
844次原研究运行和160次选优都不等于同等数量的独立样本。小样本100%也不证明
新设置下一直正确。当前没有可核实的公开arXiv编号或同行评审录用声明。

源码保留GPL-3.0及上游署名；论文保留原作者贡献与AI辅助说明。

## Hugging Face

[研究介绍与下载入口](https://huggingface.co/datasets/tianxiao1430-jpg/rfm-initialization-study)已在 Hugging Face 公开。完整数据包目前通过 GitHub Release 下载；Hugging Face 文件镜像尚未上传完成。
