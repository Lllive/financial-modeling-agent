# 简历项目条目 — 中文版

---

## 项目条目（条目式，适用于简历项目栏）

**Agent 驱动的金融量化建模系统** | Python · LangChain · PyTorch · XGBoost  
*个人项目 · 2026 年 2 月*

- 基于 **LangChain ReAct（推理-行动）模式**构建端到端自动化机器学习 Agent，将数据探索、预处理、特征工程、模型训练、评估等 6 个关键步骤封装为可组合的 **LangChain Tool**，Agent 自主编排完整流水线，无需外部 LLM API
- 在真实金融时序数据集（81,046 × 321）上实现**按交易日期严格时序切分**（前80%训练/后20%测试），配合自动化数据泄漏审计工具，确保 Scaler/缺失值统计量均仅在训练集上 fit，零未来信息泄漏
- 通过**方差过滤 + 互信息排序**将 300 个原始特征压缩至 Top-100，降低噪声维度，提升模型训练效率与泛化能力
- 在类别不平衡（正负比 ≈ 1:6）场景下训练并横向对比 4 类模型：**逻辑回归**、**XGBoost**、**LightGBM**、**3 层 PyTorch MLP**（BatchNorm + Dropout + Adam），均使用 `scale_pos_weight` 修正类别偏置
- 在测试集上取得最优 AUC = **0.5586**（符合金融时序信噪比极低的客观规律），并自动生成 **ROC 曲线、混淆矩阵、特征重要性、模型对比柱状图**等完整可视化报告

---

## 项目描述（段落式，适用于项目介绍/作品集）

**Agent 驱动的金融量化建模系统** | Python · LangChain · PyTorch · XGBoost · LightGBM  
*2026 年 2 月*

基于 LangChain ReAct 模式构建了一套自主金融量化建模 Agent。Agent 将整条机器学习流水线——EDA 分析、时序预处理、数据泄漏检测、特征筛选、多模型训练与评估——逐步拆解为独立的 LangChain Tool，通过全局状态字典（类 Agent 工作记忆）跨工具共享数据，以"思考→行动→观察"的 ReAct 循环完成端到端自动决策，无需调用 OpenAI/Claude 等外部接口即可运行。

数据集包含 81,046 条金融信号样本（300 个数值特征、12 个三值标签），以 `trade_date` 为量界进行时序 80/20 切分；使用互信息从 300 个特征中筛选 Top-100 有效特征；针对 1:6 的正负类不平衡问题，在逻辑回归、XGBoost、LightGBM 和 PyTorch MLP 四种模型中均引入显式权重补偿。最终 Agent 自动输出 AUC/Precision/Recall/F1 对比表并推荐最优模型，完整的可视化报告（ROC、混淆矩阵、性能对比图、特征重要性图）自动保存为 PNG。

---

## 技术栈关键词（用于 ATS 匹配）

`Python 3.12` · `LangChain` · `ReAct Agent` · `PyTorch` · `XGBoost` · `LightGBM` · `scikit-learn` · `pandas` · `Jupyter Notebook` · `金融时序数据` · `不平衡分类` · `特征工程` · `互信息` · `AUC/ROC` · `金融信号预测`

---

## 亮点提炼（面试口头介绍用）

> "我用 LangChain 的 ReAct 模式做了一个金融量化建模 Agent，把数据清洗、特征筛选、多模型训练、评估这整条流水线拆成 6 个 Tool，Agent 自己编排调用，不依赖任何外部 LLM API，整个过程完全可复现。重点是在时序切分和数据泄漏防护上下了功夫，确保在真实金融场景下的评估可信性。"
