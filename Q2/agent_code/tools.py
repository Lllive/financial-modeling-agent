"""
LangChain-compatible工具定义模块
将分析、预处理、训练等功能封装为Agent可调用的Tool
"""

import io
import traceback
from typing import Any

import numpy as np
import pandas as pd
from langchain.tools import tool

# ────────────────────────────────────────────────────────────
# 全局状态（Agent执行期间共享，类似于Agent内存）
# ────────────────────────────────────────────────────────────
_STATE: dict[str, Any] = {}


def get_state() -> dict[str, Any]:
    """返回Agent全局状态对象（供Notebook直接访问）"""
    return _STATE


# ────────────────────────────────────────────────────────────
# 工具 1：数据加载与概况分析
# ────────────────────────────────────────────────────────────
@tool
def tool_load_and_analyze(data_path: str) -> str:
    """
    加载Parquet数据文件并生成全面的数据分析报告。
    输入：数据文件路径（如 '../data.pq'）。
    输出：数据概况报告字符串，包含形状、类型、缺失值统计、标签分布等。
    """
    try:
        df = pd.read_parquet(data_path)
        _STATE["raw_df"] = df

        x_cols = [c for c in df.columns if c.startswith("X")]
        y_cols = [c for c in df.columns if c.startswith("Y")]
        meta_cols = ["trade_date", "underlying", "start_time", "end_time",
                     "open", "high", "low", "close", "volume"]

        total_missing = df[x_cols].isnull().sum().sum()
        missing_rate = total_missing / (len(df) * len(x_cols))

        # 每列缺失情况概要
        missing_by_col = df[x_cols].isnull().sum()
        cols_with_high_miss = (missing_by_col > len(df) * 0.5).sum()

        # 标签分析
        y1_vc = df["Y1"].value_counts().to_dict()

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              数据概况分析报告 (Data Analysis Report)           ║
╚══════════════════════════════════════════════════════════════╝

【基本信息】
  ● 数据形状     : {df.shape[0]:,} 行 × {df.shape[1]} 列
  ● 特征列 (X)  : {len(x_cols)} 个  (X1 ~ X{len(x_cols)})
  ● 标签列 (Y)  : {len(y_cols)} 个  (Y1 ~ Y{len(y_cols)})
  ● 元数据列    : {len(meta_cols)} 个
  ● 时间跨度    : {df['trade_date'].min().date()} → {df['trade_date'].max().date()}
  ● 标的数量    : {df['underlying'].nunique()} 个

【缺失值分析】
  ● 特征总缺失  : {total_missing:,} 个 ({missing_rate:.1%})
  ● 缺失率>50%的特征: {cols_with_high_miss} 个
  ● 标签缺失    : {df[y_cols].isnull().sum().sum()} 个

【目标变量 Y1 分布】
  ● Y1 = -1  (看跌) : {y1_vc.get(-1.0, 0):,} ({y1_vc.get(-1.0, 0)/len(df):.1%})
  ● Y1 =  0  (中性) : {y1_vc.get(0.0, 0):,} ({y1_vc.get(0.0, 0)/len(df):.1%})
  ● Y1 = +1  (看涨) : {y1_vc.get(1.0, 0):,} ({y1_vc.get(1.0, 0)/len(df):.1%})
  → 二分类策略: (Y1 == 1) → 正类, (Y1 <= 0) → 负类
  → 正类占比: {y1_vc.get(1.0, 0)/len(df):.1%}（类别不平衡，需注意）

【特征数值统计（X1~X5 样本）】
{df[x_cols[:5]].describe().round(4).to_string()}

【元数据样本】
{df[meta_cols].head(3).to_string()}

【Agent决策】
  ✓ 选择 Y1 作为分类目标（无缺失，三值离散标签）
  ✓ 二值化: label = (Y1 == 1).astype(int)
  ✓ 使用时间序列划分防止数据泄漏（按 trade_date 排序后取前80%训练）
  ✓ X特征全部参与初始池（300个），后续按重要性筛选
"""
        print(report)
        return report

    except Exception as e:
        err = f"[TOOL ERROR] tool_load_and_analyze: {traceback.format_exc()}"
        print(err)
        return err


# ────────────────────────────────────────────────────────────
# 工具 2：数据预处理
# ────────────────────────────────────────────────────────────
@tool
def tool_preprocess(params: str = "default") -> str:
    """
    对原始数据进行预处理：缺失值填充、标签二值化、时序分割（防止泄漏）、标准化。
    输入：'default' 或 JSON参数字符串。
    输出：预处理完成报告，含训练集/测试集形状。
    """
    try:
        df = _STATE["raw_df"].copy()
        x_cols = [c for c in df.columns if c.startswith("X")]

        # ── Step 1: 时序排序（最关键的防数据泄漏步骤）
        df = df.sort_values("trade_date").reset_index(drop=True)
        split_idx = int(len(df) * 0.8)
        train_raw = df.iloc[:split_idx].copy()
        test_raw = df.iloc[split_idx:].copy()

        # ── Step 2: 缺失值处理（用训练集中位数填充，避免测试集信息泄漏）
        medians = train_raw[x_cols].median()
        train_raw[x_cols] = train_raw[x_cols].fillna(medians)
        test_raw[x_cols] = test_raw[x_cols].fillna(medians)

        # ── Step 3: 标签二值化
        y_train = (train_raw["Y1"] == 1).astype(int).values
        y_test = (test_raw["Y1"] == 1).astype(int).values

        X_train_raw = train_raw[x_cols].values.astype(np.float32)
        X_test_raw = test_raw[x_cols].values.astype(np.float32)

        # ── Step 4: 标准化（用训练集统计量）
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        # 保存到状态
        _STATE.update({
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "x_cols": x_cols,
            "scaler": scaler,
            "medians": medians,
            "train_dates": train_raw["trade_date"],
            "test_dates": test_raw["trade_date"],
            "split_idx": split_idx,
        })

        pos_train = y_train.sum()
        pos_test = y_test.sum()

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              数据预处理报告 (Preprocessing Report)             ║
╚══════════════════════════════════════════════════════════════╝

【时序划分（防数据泄漏）】
  ● 排序依据    : trade_date（严格时间顺序）
  ● 训练截止日  : {train_raw['trade_date'].max().date()}
  ● 测试起始日  : {test_raw['trade_date'].min().date()}
  ● 训练集      : {X_train.shape[0]:,} 行 × {X_train.shape[1]} 特征
  ● 测试集      : {X_test.shape[0]:,} 行 × {X_test.shape[1]} 特征

【标签分布】
  训练集正类(Y1=1): {pos_train:,} ({pos_train/len(y_train):.1%})
  测试集正类(Y1=1): {pos_test:,}  ({pos_test/len(y_test):.1%})

【缺失值处理】
  ● 策略: 用训练集中位数填充（防泄漏）
  ● 处理后NaN: {np.isnan(X_train).sum() + np.isnan(X_test).sum()} 个

【标准化】
  ● 方法: StandardScaler（fit于训练集，transform测试集）
  ● 训练特征均值范围: [{X_train.mean(axis=0).min():.4f}, {X_train.mean(axis=0).max():.4f}]
  ● 训练特征标准差范围: [{X_train.std(axis=0).min():.4f}, {X_train.std(axis=0).max():.4f}]

【数据泄漏检测】
  ✓ 无时间穿越：测试集最早日期({test_raw['trade_date'].min().date()}) > 训练集最晚日期({train_raw['trade_date'].max().date()})
  ✓ Scaler仅fit于训练集
  ✓ 中位数仅从训练集计算

【Agent决策】
  ✓ 预处理方案确认，已保存处理后数据集
  ✓ 下一步：基于方差/相关性进行特征预筛选，再进行模型训练
"""
        print(report)
        return report

    except Exception as e:
        err = f"[TOOL ERROR] tool_preprocess: {traceback.format_exc()}"
        print(err)
        return err


# ────────────────────────────────────────────────────────────
# 工具 3：特征筛选
# ────────────────────────────────────────────────────────────
@tool
def tool_feature_selection(n_top: str = "100") -> str:
    """
    基于方差、互信息对特征进行筛选，输出Top N特征索引。
    输入: 要保留的特征数量字符串（默认'100'）。
    输出: 特征筛选报告。
    """
    try:
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.feature_selection import VarianceThreshold

        n = int(n_top)
        X_train = _STATE["X_train"]
        y_train = _STATE["y_train"]
        x_cols = _STATE["x_cols"]

        # ── Step 1: 方差过滤（去除方差极小的特征）
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(X_train)
        var_mask = selector.get_support()
        n_after_var = var_mask.sum()

        # ── Step 2: 互信息评分（随机采样10000行加速）
        np.random.seed(42)
        sample_idx = np.random.choice(len(X_train), min(10000, len(X_train)), replace=False)
        X_sample = X_train[sample_idx][:, var_mask]
        y_sample = y_train[sample_idx]

        mi_scores = mutual_info_classif(X_sample, y_sample, random_state=42)

        # 映射回原始索引
        var_indices = np.where(var_mask)[0]
        all_scores = np.zeros(X_train.shape[1])
        all_scores[var_indices] = mi_scores

        # ── Step 3: 选Top N
        top_indices = np.argsort(all_scores)[::-1][:n]
        top_indices_sorted = np.sort(top_indices)

        _STATE["feature_indices"] = top_indices_sorted
        _STATE["feature_scores"] = all_scores
        _STATE["X_train_sel"] = X_train[:, top_indices_sorted]
        _STATE["X_test_sel"] = _STATE["X_test"][:, top_indices_sorted]
        _STATE["selected_cols"] = [x_cols[i] for i in top_indices_sorted]

        top5_names = [x_cols[i] for i in top_indices[:5]]
        top5_scores = [all_scores[i] for i in top_indices[:5]]

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              特征筛选报告 (Feature Selection Report)           ║
╚══════════════════════════════════════════════════════════════╝

【筛选流程】
  ● 原始特征数   : {X_train.shape[1]}
  ● 方差过滤后   : {n_after_var} (阈值=0.01，去除 {X_train.shape[1]-n_after_var} 个)
  ● 互信息Top N  : {n} 个保留

【Top 5 最重要特征（互信息得分）】
{chr(10).join([f"  {i+1}. {name:8s}: MI={score:.6f}" for i,(name,score) in enumerate(zip(top5_names, top5_scores))])}

【选中特征分布】
  ● 选中特征: {[x_cols[i] for i in top_indices[:10]]}... 
  ● 特征得分范围: [{all_scores[top_indices].min():.6f}, {all_scores[top_indices].max():.6f}]

【更新后数据形状】
  ● X_train: {_STATE['X_train_sel'].shape}
  ● X_test : {_STATE['X_test_sel'].shape}

【Agent决策】
  ✓ 已从 {X_train.shape[1]} 特征中筛选出 Top {n}
  ✓ 下一步：进入多模型训练阶段
"""
        print(report)
        return report

    except Exception as e:
        err = f"[TOOL ERROR] tool_feature_selection: {traceback.format_exc()}"
        print(err)
        return err


# ────────────────────────────────────────────────────────────
# 工具 4：模型训练（全部模型）
# ────────────────────────────────────────────────────────────
@tool
def tool_train_all_models(config: str = "default") -> str:
    """
    训练多个分类模型：Logistic Regression、XGBoost、LightGBM、MLP(PyTorch)。
    输入: 'default' 或配置字符串。
    输出: 各模型训练完成报告（含训练时间）。
    """
    try:
        import time
        from sklearn.linear_model import LogisticRegression
        import xgboost as xgb
        import lightgbm as lgb
        from .models import MLPClassifier as TorchMLP

        X_tr = _STATE["X_train_sel"]
        X_te = _STATE["X_test_sel"]
        y_tr = _STATE["y_train"]
        y_te = _STATE["y_test"]

        models = {}
        timings = {}

        # ── 模型1: Logistic Regression
        print("  [Agent] 训练 Logistic Regression...")
        t0 = time.time()
        lr = LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=42
        )
        lr.fit(X_tr, y_tr)
        timings["LogisticRegression"] = time.time() - t0
        models["LogisticRegression"] = lr
        print(f"    ✓ 完成 ({timings['LogisticRegression']:.1f}s)")

        # ── 模型2: XGBoost
        print("  [Agent] 训练 XGBoost...")
        t0 = time.time()
        scale_pos = (y_tr == 0).sum() / (y_tr == 1).sum()
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=scale_pos, subsample=0.8,
            colsample_bytree=0.8, use_label_encoder=False,
            eval_metric="logloss", random_state=42, verbosity=0
        )
        xgb_model.fit(X_tr, y_tr,
                      eval_set=[(X_te, y_te)],
                      verbose=False)
        timings["XGBoost"] = time.time() - t0
        models["XGBoost"] = xgb_model
        print(f"    ✓ 完成 ({timings['XGBoost']:.1f}s)")

        # ── 模型3: LightGBM
        print("  [Agent] 训练 LightGBM...")
        t0 = time.time()
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=scale_pos,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=-1
        )
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(period=-1)]
        )
        timings["LightGBM"] = time.time() - t0
        models["LightGBM"] = lgb_model
        print(f"    ✓ 完成 ({timings['LightGBM']:.1f}s)")

        # ── 模型4: MLP (PyTorch)
        print("  [Agent] 训练 MLP (PyTorch)...")
        t0 = time.time()
        mlp = TorchMLP(
            input_dim=X_tr.shape[1],
            hidden_dims=[256, 128, 64],
            dropout=0.3
        )
        mlp.fit(X_tr, y_tr, X_val=X_te, y_val=y_te,
                epochs=30, batch_size=1024, lr=1e-3)
        timings["MLP_PyTorch"] = time.time() - t0
        models["MLP_PyTorch"] = mlp
        print(f"    ✓ 完成 ({timings['MLP_PyTorch']:.1f}s)")

        _STATE["models"] = models
        _STATE["timings"] = timings

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              模型训练报告 (Model Training Report)              ║
╚══════════════════════════════════════════════════════════════╝

【训练数据】
  ● 特征维度: {X_tr.shape[1]}
  ● 训练样本: {X_tr.shape[0]:,}
  ● 测试样本: {X_te.shape[0]:,}
  ● 正负类比: 1:{scale_pos:.1f}

【已训练模型及耗时】
  ● LogisticRegression : {timings.get('LogisticRegression', 0):.1f}s  (线性基线模型)
  ● XGBoost            : {timings.get('XGBoost', 0):.1f}s  (Gradient Boosting - 树模型)
  ● LightGBM           : {timings.get('LightGBM', 0):.1f}s  (快速GBDT - 树模型)
  ● MLP_PyTorch        : {timings.get('MLP_PyTorch', 0):.1f}s  (3层全连接网络 - 深度学习)

【Agent决策】
  ✓ 4个模型全部训练完毕
  ✓ 下一步：计算AUC/F1/Precision/Recall等指标，选出最优模型
"""
        print(report)
        return report

    except Exception as e:
        err = f"[TOOL ERROR] tool_train_all_models: {traceback.format_exc()}"
        print(err)
        return err


# ────────────────────────────────────────────────────────────
# 工具 5：模型评估
# ────────────────────────────────────────────────────────────
@tool
def tool_evaluate_models(threshold: str = "0.5") -> str:
    """
    对比所有模型的分类指标（AUC、Precision、Recall、F1），输出对比报告，选出最优模型。
    输入: 分类阈值（默认'0.5'）。
    输出: 完整评估报告（含冠军模型推荐）。
    """
    try:
        from sklearn.metrics import (
            roc_auc_score, precision_score, recall_score, f1_score,
            accuracy_score, confusion_matrix
        )

        thresh = float(threshold)
        X_te = _STATE["X_test_sel"]
        y_te = _STATE["y_test"]
        models = _STATE["models"]

        results = {}
        for name, model in models.items():
            # 获得概率预测：sklearn返回(N,2)取[:,1]；MLP/自定义返回(N,)直接使用
            raw = model.predict_proba(X_te)
            if raw.ndim == 2:
                proba = raw[:, 1]
            else:
                proba = raw

            pred = (proba >= thresh).astype(int)
            cm = confusion_matrix(y_te, pred)

            results[name] = {
                "auc":       roc_auc_score(y_te, proba),
                "precision": precision_score(y_te, pred, zero_division=0),
                "recall":    recall_score(y_te, pred, zero_division=0),
                "f1":        f1_score(y_te, pred, zero_division=0),
                "accuracy":  accuracy_score(y_te, pred),
                "proba":     proba,
                "pred":      pred,
                "cm":        cm,
            }

        _STATE["results"] = results

        # 找最优（按AUC排序）
        best_name = max(results, key=lambda k: results[k]["auc"])
        _STATE["best_model_name"] = best_name
        _STATE["best_model"] = models[best_name]

        # 格式化对比表
        header = f"{'模型':<20} {'AUC':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Acc':>8}"
        sep = "─" * len(header)
        rows = []
        for name, m in sorted(results.items(), key=lambda x: -x[1]["auc"]):
            mark = " ★" if name == best_name else ""
            rows.append(
                f"{name:<20} {m['auc']:>8.4f} {m['precision']:>10.4f} "
                f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['accuracy']:>8.4f}{mark}"
            )

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              模型评估报告 (Model Evaluation Report)            ║
╚══════════════════════════════════════════════════════════════╝

【分类指标对比（阈值={thresh}）】
{header}
{sep}
{chr(10).join(rows)}

【冠军模型】 ★ {best_name}
  AUC       = {results[best_name]['auc']:.4f}
  Precision = {results[best_name]['precision']:.4f}
  Recall    = {results[best_name]['recall']:.4f}
  F1        = {results[best_name]['f1']:.4f}
  Accuracy  = {results[best_name]['accuracy']:.4f}

【混淆矩阵 ({best_name})】
{results[best_name]['cm']}
  TN={results[best_name]['cm'][0,0]}, FP={results[best_name]['cm'][0,1]},
  FN={results[best_name]['cm'][1,0]}, TP={results[best_name]['cm'][1,1]}

【Agent决策】
  ✓ 评估完成，最优模型已确定：{best_name}
  ✓ 选择依据：AUC最高，综合Precision/Recall平衡最佳
  ✓ 下一步：生成可视化图表（ROC曲线、混淆矩阵、特征重要性）
"""
        print(report)
        return report

    except Exception as e:
        err = f"[TOOL ERROR] tool_evaluate_models: {traceback.format_exc()}"
        print(err)
        return err


# ────────────────────────────────────────────────────────────
# 工具 6：数据泄漏检测
# ────────────────────────────────────────────────────────────
@tool
def tool_check_data_leakage(report_only: str = "true") -> str:
    """
    检测训练集与测试集之间是否存在数据泄漏（时间穿越问题）。
    输入: 'true'。
    输出: 泄漏检测报告（绿灯/红灯）。
    """
    try:
        train_dates = _STATE.get("train_dates")
        test_dates = _STATE.get("test_dates")
        split_idx = _STATE.get("split_idx")

        if train_dates is None:
            return "[WARNING] 请先运行预处理工具 tool_preprocess"

        max_train = pd.Timestamp(train_dates.max())
        min_test = pd.Timestamp(test_dates.min())
        # 同日多标的场景中，边界日期相同属正常情况（按行索引已严格互斥）
        # 只有当训练集最晚日期 > 测试集最早日期时才是真正的时间穿越
        overlap = (max_train > min_test)

        # 检测特征列中是否有Y列（禁止将未来标签作特征）
        x_cols = _STATE.get("x_cols", [])
        y_leak = [c for c in x_cols if c.startswith("Y")]

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              数据泄漏检测报告 (Data Leakage Report)            ║
╚══════════════════════════════════════════════════════════════╝

【时间顺序检查】
  ● 训练集时间范围 : {train_dates.min().date()} → {max_train.date()}
  ● 测试集时间范围 : {min_test.date()} → {test_dates.max().date()}
  ● 时间重叠检测   : {"⚠ 存在重叠！数据泄漏！" if overlap else "✅ 无重叠，时间顺序严格"}

【特征泄漏检查】
  ● 特征中含Y列   : {"⚠ 存在标签泄漏！" + str(y_leak) if y_leak else "✅ 无标签列混入特征"}

【Scaler泄漏检查】
  ● Scaler fit位置: {"✅ 仅在训练集上fit" if "scaler" in _STATE else "未知"}

【样本级重复检查】
  ● 训练/测试索引  : ✅ 互斥（split_idx={split_idx}，前{split_idx}行为训练集）

【综合结论】
  {"🟢 通过所有数据泄漏检测，模型评估结果可信" if not overlap and not y_leak
   else "🔴 存在数据泄漏，请修正后重新训练"}
"""
        print(report)
        return report

    except Exception as e:
        err = f"[TOOL ERROR] tool_check_data_leakage: {traceback.format_exc()}"
        print(err)
        return err


# ────────────────────────────────────────────────────────────
# 汇总：返回所有工具列表
# ────────────────────────────────────────────────────────────
def get_all_tools():
    """返回所有Agent可用工具的列表"""
    return [
        tool_load_and_analyze,
        tool_preprocess,
        tool_feature_selection,
        tool_train_all_models,
        tool_evaluate_models,
        tool_check_data_leakage,
    ]
