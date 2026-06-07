# Notes: 工件分类建模论文（材料提取）

## 题目（工件分类.docx，权威）
- 工厂对三类工件 A/B/C 做 X 光扫描，数据见 SpecialData.xlsx（实际文件名 SpecData.xlsx）。
- 4 个工作表：A类工件 / B类工件 / C类工件 / 未知类型工件。
- 第一列为工件编号，其余 155 列为 155 个检测位置测量值。
- 任务：
  1. 建模分析三类工件检测数据特征。**禁止建立神经网络模型。**
  2. 给出 10 个未知工件类型。
  3. 撰写论文，详述模型分析与构建并评价模型。

## 数据规模（final_results.json，权威）
- A: 3572 样本 / 155 特征 / 缺失 0
- B: 2643 / 155 / 0
- C: 1763 / 155 / 0
- 合计 N = 7978；未知 10（D1–D10）。

## 权威结果来源优先级
- 权威：final_results.json、final_model_metrics.csv、final_unknown_classification.csv、final_feature_importance.csv、final_modeling_answer.md（四者一致）。
- **过时/冲突**：modeling_scheme.md 的 PCA-QDA(12维) 指标、混淆矩阵、PCA维数敏感性表、分类别 precision 表是旧版本，与权威文件不一致。论文一律采用权威文件数字。

### PCA-QDA(12维) 冲突明细（写作时只用右列权威值）
| 指标 | modeling_scheme.md(旧) | 权威(JSON/CSV/answer) |
|---|---|---|
| 总准确率 | 0.954625 | 0.958010 |
| 平衡准确率 | 0.962086 | 0.963949 |
| A 召回率 | 0.933931 | 0.947088 |
| B 召回率 | 0.952327 | 0.944760 |
| A 精确率 | 0.963605 | 0.958629 |
| B 精确率 | 0.914275 | 0.929635 |
| 混淆(A行) | [3336,236,0] | [3383,189,0] |
| 混淆(B行) | [126,2517,0] | [146,2497,0] |
| PCA r=50 | 0.979067 | 0.979945 |

## 五模型 5 折交叉验证性能（final_model_metrics.csv，权威）
| 模型 | 总准确率 | 宏精确率 | 平衡准确率 | 宏F1 | A召回 | B召回 | C召回 |
|---|---|---|---|---|---|---|---|
| 最近质心 | 0.953497 | 0.962575 | 0.956359 | 0.958897 | 0.973124 | 0.895952 | 1.0 |
| 高斯朴素贝叶斯 | 0.991727 | 0.993953 | 0.991676 | 0.992734 | 1.0 | 0.975028 | 1.0 |
| 正则化LDA | 0.950614 | 0.960884 | 0.953261 | 0.956257 | 0.974804 | 0.884979 | 1.0 |
| 正则化QDA | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| PCA-QDA(12维) | 0.958010 | 0.962755 | 0.963949 | 0.963320 | 0.947088 | 0.944760 | 1.0 |

## 混淆矩阵（JSON，权威；行=真实，列=预测 A/B/C）
- 最近质心: [[3476,96,0],[275,2368,0],[0,0,1763]]
- 朴素贝叶斯: [[3572,0,0],[66,2577,0],[0,0,1763]]
- LDA: [[3482,90,0],[304,2339,0],[0,0,1763]]
- QDA: [[3572,0,0],[0,2643,0],[0,0,1763]]
- PCA-QDA(12): [[3383,189,0],[146,2497,0],[0,0,1763]]

## 特征分析核心数字
- 三类全体均值 A=7.0291 > B=6.2109 > C=3.8578；A 样本内标准差均值 1.4831 远大于 B 0.7847、C 0.7868。
- ANOVA top 位置：134(F=4731.81, η²=0.5427)、138、88、108、75…；类均值顺序均 A>B>C。
- PCA：PC1 解释 0.5417，前10累计 0.5848；类中心 PC1：A=-7.11, B=-1.13, C=16.10（C 远离 A/B）。
- 相邻位置相关：总体 0.6196；分类别 A=0.1189, B=0.6679, C=0.0884（B 类内部相关高 → NB 在 B 类召回最低 0.975）。

## QDA 100% 机制（关键，配图05）
- A/B 在单个位置/均值/PC1 上重叠大（单阈值仅约 94.26%）。
- A/B 在「样本内标准差」上几乎完全可分：最优切割点 s* = 1.105855，准确率 1.0。
- A 类样本内标准差最小值 1.139050 > s* > B 类最大值 1.072661 → 完全分离。
- 结论：QDA 满分来自方差/协方差结构，而非单点线性可分；生产中需监控设备方差标定漂移。

## 未知样本分类（final_unknown_classification.csv，权威；5/5 一致，QDA后验=1.0）
- A: D1, D2, D3, D10
- B: D4, D5, D6
- C: D7, D8, D9

## 模型超参数（final_stepwise_code.py，权威）
- 标准化：训练折内 Z-score（防泄漏）。分层 5 折，seed=20260605。
- 经验先验 π=n_g/N（A 0.4477, B 0.3313, C 0.2210）。
- LDA 收缩 λ=0.60；QDA 收缩 α=0.12；PCA-QDA 用 r=12、收缩 0.10。
- 注：代码读 per-class CSV（kongfei 机器路径），提供数据为单一 SpecData.xlsx 四表；管线等价，论文按 xlsx 四表描述。

## 图表（charts/，chart_index.md）
1. 01_mean_curves — 三类均值曲线±1σ阴影【正文：特征分析】
2. 02_pca_scatter — PCA 散点+类中心【正文：可分性】
3. 03_pca_scree — 碎石图+累计解释率【附录或正文小图】
4. 04_feature_importance — 关键位置 η² 排名【正文：特征分析】
5. 05_ab_std_threshold — A/B 样本内标准差分布+阈值【正文：机制，核心】
6. 06_model_performance — 模型性能对比【正文：模型比较】
7. 07_confusion_matrices — NB/LDA/QDA 混淆矩阵热力图【正文：分类效果】
- 待写作时核对：图06 的 PCA-QDA 柱是否用权威值（应为 0.958010 系列）。

## 格式要求（论文提交和论文格式.docx）
- PDF；第1页封面（标题/作者/院系）；第2页中文摘要+关键词，不超过一页；正文从第3页起，页脚居中阿拉伯页码从1连续。
- 标题三号黑体；一级标题四号黑体居中；二三级小四黑体左对齐；正文小四宋体；单倍行距。
- 引用：正文方括号编号 [1][3]；按引用次序列参考文献。
  - 书籍：[编号] 作者，书名，出版地：出版社，出版年。
  - 期刊：[编号] 作者，论文名，杂志名，卷期号：起止页码，出版年。
  - 网络：[编号] 作者，资源标题，网址，访问时间。
- 提交：2026-06-07 20:00 前；文件名 队长_队员2_队员3。
- 现有 final.tex 模板已按此格式设置（封面/摘要/章节/页码/字体）。

## 候选参考文献（需写作前用 WebSearch 核验真实性后再用）
- Fisher R.A. 1936, The use of multiple measurements in taxonomic problems, Annals of Eugenics 7(2):179-188 —— 判别分析起源。
- Friedman J.H. 1989, Regularized Discriminant Analysis, JASA 84(405):165-175 —— 正则化 LDA/QDA 收缩，直接支撑主模型。
- Hastie, Tibshirani, Friedman 2009, The Elements of Statistical Learning(2nd), Springer —— LDA/QDA/PCA/CV 教材依据。
- Jolliffe & Cadima 2016, PCA: a review and recent developments, Phil. Trans. R. Soc. A 374:20150202 —— PCA 方法依据。
- Hand & Yu 2001, Idiot's Bayes—Not So Stupid After All?, Int. Statistical Review 69(3):385-398 —— 朴素贝叶斯依据。
- Kohavi 1995, A study of cross-validation and bootstrap…, IJCAI —— 交叉验证。
- Cawley & Talbot 2010, On Over-fitting in Model Selection…, JMLR 11:2079-2107 —— 嵌套交叉验证防泄漏。
- Brodersen et al. 2010, The Balanced Accuracy and Its Posterior Distribution, ICPR —— 平衡准确率。
- Montgomery 2009, Introduction to Statistical Quality Control, Wiley —— Hotelling T² 批次监控。
- Mery D. 2015, Computer Vision for X-Ray Testing, Springer —— X 光工业检测背景。
(以上均为常见经典文献，写作前逐条核验卷期页码。)
