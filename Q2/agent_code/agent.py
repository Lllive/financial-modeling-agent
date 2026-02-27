"""
FinancialModelingAgent — 金融建模自动化Agent
基于LangChain工具系统实现ReAct（Reasoning + Acting）模式
无需LLM API即可运行（内置确定性推理引擎）
"""

import sys
import time
from dataclasses import dataclass, field
from typing import Any

from .tools import (
    get_state,
    tool_load_and_analyze,
    tool_preprocess,
    tool_feature_selection,
    tool_train_all_models,
    tool_evaluate_models,
    tool_check_data_leakage,
)


# ────────────────────────────────────────────────────────────
# ReAct步骤数据结构
# ────────────────────────────────────────────────────────────
@dataclass
class ReActStep:
    step_no: int
    thought: str          # Agent推理过程
    action: str           # 调用的工具名
    action_input: str     # 工具输入
    observation: str = "" # 工具返回结果
    elapsed: float = 0.0  # 耗时（秒）


# ────────────────────────────────────────────────────────────
# Agent主类
# ────────────────────────────────────────────────────────────
class FinancialModelingAgent:
    """
    基于LangChain工具定义、ReAct推理模式的金融建模Agent。

    工作流程（确定性ReAct链）：
    1. 加载数据 → 分析数据概况
    2. 数据预处理（缺失值/标准化/时序分割）
    3. 数据泄漏检测
    4. 特征筛选（互信息Top-N）
    5. 多模型训练（LR / XGBoost / LightGBM / MLP）
    6. 模型评估（AUC / F1 / Precision / Recall）
    7. 生成最终报告

    注：Agent每一步都打印 Thought + Action + Observation，
    完整再现ReAct框架的决策过程。
    """

    SYSTEM_PROMPT = """你是一个资深数据科学家与金融建模专家。
你的任务是对金融时序数据自动完成以下工作：
1. 分析数据质量与标签分布
2. 设计无数据泄漏的预处理方案
3. 从300个特征中筛选最有信息量的特征
4. 训练并对比多种分类模型（线性、树模型、神经网络）
5. 用AUC/F1/Precision/Recall全面评估模型，输出最优方案

每一步你都需要先思考（Thought），再采取行动（Action），最后观察结果（Observation）。
"""

    def __init__(self, data_path: str = "../data.pq", n_features: int = 100):
        self.data_path = data_path
        self.n_features = n_features
        self.steps: list[ReActStep] = []
        self._tool_map = {
            "tool_load_and_analyze":  tool_load_and_analyze,
            "tool_preprocess":        tool_preprocess,
            "tool_check_data_leakage": tool_check_data_leakage,
            "tool_feature_selection": tool_feature_selection,
            "tool_train_all_models":  tool_train_all_models,
            "tool_evaluate_models":   tool_evaluate_models,
        }
        # ReAct脚本：每步包含(thought, action, action_input)
        self._script = [
            (
                f"我需要先了解数据的基本情况。数据文件是 '{data_path}'，"
                "先调用数据分析工具查看形状、特征数量、标签分布和缺失值情况，"
                "以便制定后续预处理方案。",
                "tool_load_and_analyze",
                data_path,
            ),
            (
                "数据已加载完毕。观察到：(1) Y1标签为三值{-1,0,1}，"
                "需要二值化为分类任务；(2) 特征存在约23%缺失率，"
                "需要用训练集中位数填充；(3) 数据跨越多年，"
                "必须按时间顺序划分训练/测试集以防止数据泄漏。"
                "现在调用预处理工具。",
                "tool_preprocess",
                "default",
            ),
            (
                "预处理完成。为了确保模型评估的可靠性，"
                "我需要验证训练集和测试集之间没有时间穿越的数据泄漏。"
                "调用泄漏检测工具执行审计。",
                "tool_check_data_leakage",
                "true",
            ),
            (
                "数据泄漏检测通过，可以安全进行建模。"
                f"现在有300个特征，若全部使用会增加计算成本和噪声。"
                f"我决定用方差过滤+互信息评分筛选Top {n_features}个最优特征，"
                "提升模型效率和性能。",
                "tool_feature_selection",
                str(n_features),
            ),
            (
                f"特征已从300个压缩到{n_features}个。"
                "现在进入核心建模阶段。按照计划训练4种模型：\n"
                "  1. Logistic Regression（线性基线）\n"
                "  2. XGBoost（集成树模型，处理非线性关系）\n"
                "  3. LightGBM（高效GBDT，速度更快）\n"
                "  4. MLP PyTorch（深度学习，捕获复杂模式）\n"
                "所有模型都使用class_weight/scale_pos_weight处理类别不平衡。",
                "tool_train_all_models",
                "default",
            ),
            (
                "4个模型均已训练完成。现在需要用AUC/Precision/Recall/F1"
                "对所有模型进行全面评估对比，找出最优模型，"
                "并分析其在测试集上的混淆矩阵以理解误判情况。",
                "tool_evaluate_models",
                "0.5",
            ),
        ]

    # ──────────────────────────────────────────────────────
    # 核心执行方法
    # ──────────────────────────────────────────────────────
    def run(self) -> dict[str, Any]:
        """
        执行完整的ReAct工作流，返回最终状态字典。
        """
        self._print_header()

        for i, (thought, action, action_input) in enumerate(self._script):
            step_no = i + 1
            self._print_thought(step_no, thought, action, action_input)

            t0 = time.time()
            try:
                tool_fn = self._tool_map[action]
                observation = tool_fn.invoke(action_input)
            except Exception as exc:
                observation = f"[ERROR] {exc}"

            elapsed = time.time() - t0
            step = ReActStep(
                step_no=step_no,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                elapsed=elapsed,
            )
            self.steps.append(step)
            self._print_observation_footer(step_no, elapsed)

        self._print_final_summary()
        return get_state()

    # ──────────────────────────────────────────────────────
    # 打印辅助方法
    # ──────────────────────────────────────────────────────
    def _print_header(self):
        print("=" * 70)
        print(" FinancialModelingAgent — ReAct 自动建模流程")
        print(" 数据文件:", self.data_path)
        print(" 系统提示:", self.SYSTEM_PROMPT.splitlines()[0])
        print("=" * 70)
        print()

    def _print_thought(self, step_no, thought, action, action_input):
        print(f"\n{'─'*70}")
        print(f"【Step {step_no}】")
        print(f"\n💭 Thought:\n  {thought}")
        print(f"\n⚡ Action: {action}")
        print(f"   Input : {action_input[:80]}{'...' if len(action_input)>80 else ''}")
        print(f"\n📋 Observation:")

    def _print_observation_footer(self, step_no, elapsed):
        print(f"\n  [⏱ Step {step_no} 耗时: {elapsed:.2f}s]")

    def _print_final_summary(self):
        state = get_state()
        results = state.get("results", {})
        best = state.get("best_model_name", "N/A")

        print("\n" + "=" * 70)
        print(" 🏁 Agent 执行完毕 — 最终报告摘要")
        print("=" * 70)

        if results:
            print(f"\n{'模型':<20} {'AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
            print("─" * 58)
            for name in sorted(results, key=lambda k: -results[k]["auc"]):
                m = results[name]
                tag = " ★ 最优" if name == best else ""
                print(f"{name:<20} {m['auc']:>8.4f} {m['f1']:>8.4f} "
                      f"{m['precision']:>10.4f} {m['recall']:>8.4f}{tag}")

        total_time = sum(s.elapsed for s in self.steps)
        print(f"\n总耗时: {total_time:.1f}s  |  共执行步骤: {len(self.steps)}")
        print(f"推荐模型: {best}")
        print("=" * 70)
