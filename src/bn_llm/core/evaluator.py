"""数据清洗评估模块

评估数据清洗结果，计算精确率、召回率、F1 分数等指标。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from ..utils.logger import get_logger


@dataclass
class EvaluationMetrics:
    """评估指标"""

    precision: float
    recall: float
    f1: float
    detection_rate: float
    repair_accuracy: float


@dataclass
class EvaluationResult:
    """评估结果"""

    metrics: EvaluationMetrics
    statistics: Dict[str, int] = field(default_factory=dict)
    actual_errors: Dict[Tuple[int, str], str] = field(default_factory=dict)
    repair_attempts: Dict[Tuple[int, str], str] = field(default_factory=dict)
    successful_repairs: Dict[Tuple[int, str], Dict] = field(default_factory=dict)
    wrong_repairs: Dict[Tuple[int, str], str] = field(default_factory=dict)
    missed_errors: Dict[Tuple[int, str], str] = field(default_factory=dict)


class DataCleaningEvaluator:
    """数据清洗评估器

    通过比较清洗后的数据与标准答案来评估清洗效果。
    """

    def __init__(self):
        self.logger = get_logger()

    def evaluate(
        self,
        dirty_data: pd.DataFrame,
        repaired_data: pd.DataFrame,
        ground_truth: pd.DataFrame,
    ) -> EvaluationResult:
        """评估数据清洗结果

        Args:
            dirty_data: 原始脏数据
            repaired_data: 清洗后的数据
            ground_truth: 标准答案数据

        Returns:
            评估结果
        """
        # 转换为字符串类型进行比较
        dirty_str = dirty_data.astype(str)
        repaired_str = repaired_data.astype(str)
        truth_str = ground_truth.astype(str)

        # 找出真实错误
        actual_errors = {}
        for row in dirty_data.index:
            for col in dirty_data.columns:
                if dirty_str.loc[row, col] != truth_str.loc[row, col]:
                    actual_errors[(row, col)] = truth_str.loc[row, col]

        # 找出修复尝试
        repair_attempts = {}
        for row in dirty_data.index:
            for col in dirty_data.columns:
                if dirty_str.loc[row, col] != repaired_str.loc[row, col]:
                    repair_attempts[(row, col)] = repaired_str.loc[row, col]

        # 分析修复情况
        successful_repairs = {}
        wrong_repairs = {}
        missed_errors = {}
        correct_repairs = 0
        wrong_repair_count = 0
        missed_error_count = 0

        for cell, correct_value in actual_errors.items():
            if cell in repair_attempts:
                if str(correct_value) == str(repair_attempts[cell]):
                    correct_repairs += 1
                    successful_repairs[cell] = {
                        correct_value: (
                            f"{dirty_str.loc[cell[0], cell[1]]} ===> {repair_attempts[cell]}"
                        )
                    }
            else:
                missed_errors[cell] = (
                    f"{dirty_str.loc[cell[0], cell[1]]} 应该修复成 "
                    f"{truth_str.loc[cell[0], cell[1]]}, 模型未修复"
                )
                missed_error_count += 1

        # 检查错误的修复
        for cell, repaired_value in repair_attempts.items():
            if cell not in actual_errors:
                # 原来值就是正确的
                wrong_repair_count += 1
                wrong_repairs[cell] = (
                    f"{truth_str.loc[cell[0], cell[1]]} 本身值正确, 而不是改成 {repaired_value}"
                )
            elif str(actual_errors[cell]) != str(repaired_value):
                # 应该修改，但修改错误
                wrong_repair_count += 1
                wrong_repairs[cell] = (
                    f"{dirty_str.loc[cell[0], cell[1]]} 应该修改为 "
                    f"{truth_str.loc[cell[0], cell[1]]}, 而不是 {repaired_value}"
                )

        # 计算指标
        total_repairs = len(repair_attempts)
        total_errors = len(actual_errors)

        precision = correct_repairs / total_repairs if total_repairs > 0 else 0.0
        recall = correct_repairs / total_errors if total_errors > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        detection_rate = total_repairs / total_errors if total_errors > 0 else 0.0
        repair_accuracy = correct_repairs / total_repairs if total_repairs > 0 else 0.0

        metrics = EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            detection_rate=detection_rate,
            repair_accuracy=repair_accuracy,
        )

        statistics = {
            "total_actual_errors": total_errors,
            "total_repair_attempts": total_repairs,
            "correct_repairs": correct_repairs,
            "wrong_repairs": wrong_repair_count,
            "missed_errors": missed_error_count,
        }

        return EvaluationResult(
            metrics=metrics,
            statistics=statistics,
            actual_errors=actual_errors,
            repair_attempts=repair_attempts,
            successful_repairs=successful_repairs,
            wrong_repairs=wrong_repairs,
            missed_errors=missed_errors,
        )

    def log_summary(self, result: EvaluationResult) -> None:
        """输出评估摘要日志"""
        lines = [
            "=" * 50,
            "评估摘要",
            "=" * 50,
            "",
            "1. 错误检测概览:",
            f"   实际错误数: {result.statistics['total_actual_errors']}",
            f"   修复尝试数: {result.statistics['total_repair_attempts']}",
            "",
            "2. 正确修复:",
            f"   成功修复数: {result.statistics['correct_repairs']}",
        ]

        # 成功修复示例
        if result.successful_repairs:
            lines.append("   成功修复示例:")
            for i, (cell, info) in enumerate(result.successful_repairs.items()):
                if i >= 5:
                    break
                row, col = cell
                lines.append(f"   - ({row}, {col}): {list(info.values())[0]}")
            if len(result.successful_repairs) > 5:
                lines.append(f"   ... 还有 {len(result.successful_repairs) - 5} 个")

        # 错误修复
        lines.extend(
            [
                "",
                "3. 错误修复:",
                f"   错误修复数: {result.statistics['wrong_repairs']}",
            ]
        )

        if result.wrong_repairs:
            lines.append("   错误修复示例:")
            for i, (cell, info) in enumerate(result.wrong_repairs.items()):
                if i >= 5:
                    break
                row, col = cell
                lines.append(f"   - ({row}, {col}): {info}")
            if len(result.wrong_repairs) > 5:
                lines.append(f"   ... 还有 {len(result.wrong_repairs) - 5} 个")

        # 遗漏错误
        lines.extend(
            [
                "",
                "4. 遗漏错误:",
                f"   遗漏数: {result.statistics['missed_errors']}",
            ]
        )

        if result.missed_errors:
            lines.append("   遗漏示例:")
            for i, (cell, info) in enumerate(result.missed_errors.items()):
                if i >= 5:
                    break
                row, col = cell
                lines.append(f"   - ({row}, {col}): {info}")
            if len(result.missed_errors) > 5:
                lines.append(f"   ... 还有 {len(result.missed_errors) - 5} 个")

        # 性能指标
        lines.extend(
            [
                "",
                "5. 性能指标:",
                f"   Precision: {result.metrics.precision:.4f}",
                f"   Recall: {result.metrics.recall:.4f}",
                f"   F1 Score: {result.metrics.f1:.4f}",
                "",
                "6. 附加指标:",
                f"   Detection Rate: {result.metrics.detection_rate:.4f}",
                f"   Repair Accuracy: {result.metrics.repair_accuracy:.4f}",
                "",
                "=" * 50,
            ]
        )

        self.logger.info("\n" + "\n".join(lines))


def save_evaluation_result(
    result: EvaluationResult,
    output_path: Path,
) -> None:
    """保存评估结果到 JSON 文件

    Args:
        result: 评估结果
        output_path: 输出文件路径
    """

    def convert_tuple_keys(d: dict) -> dict:
        """将元组键转换为字符串"""
        if not d:
            return {}
        return {f"({k[0]}, {k[1]})" if isinstance(k, tuple) else k: v for k, v in d.items()}

    serializable = {
        "metrics": {
            "precision": result.metrics.precision,
            "recall": result.metrics.recall,
            "f1": result.metrics.f1,
            "detection_rate": result.metrics.detection_rate,
            "repair_accuracy": result.metrics.repair_accuracy,
        },
        "statistics": result.statistics,
        "actual_errors": convert_tuple_keys(result.actual_errors),
        "repair_attempts": convert_tuple_keys(result.repair_attempts),
        "successful_repairs": convert_tuple_keys(result.successful_repairs),
        "wrong_repairs": convert_tuple_keys(result.wrong_repairs),
        "missed_errors": convert_tuple_keys(result.missed_errors),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=4, ensure_ascii=False)


def evaluate_cleaning(
    dirty_data: pd.DataFrame,
    repaired_data: pd.DataFrame,
    ground_truth: pd.DataFrame,
    log_summary: bool = True,
) -> EvaluationResult:
    """便捷函数：评估数据清洗结果

    Args:
        dirty_data: 原始脏数据
        repaired_data: 清洗后的数据
        ground_truth: 标准答案数据
        log_summary: 是否输出摘要日志

    Returns:
        评估结果
    """
    evaluator = DataCleaningEvaluator()
    result = evaluator.evaluate(dirty_data, repaired_data, ground_truth)

    if log_summary:
        evaluator.log_summary(result)

    return result


__all__ = [
    "EvaluationMetrics",
    "EvaluationResult",
    "DataCleaningEvaluator",
    "save_evaluation_result",
    "evaluate_cleaning",
]
