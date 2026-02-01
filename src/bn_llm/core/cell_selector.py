"""单元格选择器模块

基于置信度分数筛选需要 LLM 二次确认的单元格，控制 LLM 调用成本。

主要功能：
1. 计算单元格置信度分数（基于共现矩阵和概率映射）
2. 选择低置信度单元格进行 LLM 修复

实现逻辑参考 raw/bn-llm/llm_repair_df.py:
- 共现频率按列独立归一化
- pro_map = max_prob / second_max_prob，取对数后归一化
- 组合方式: (1 + pro_map) * cooccurrence_freq
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CellSelectionResult:
    """单元格选择结果"""

    selected_cells: List[Tuple[int, str]]
    confidence_scores: Dict[Tuple[int, str], float]
    total_candidates: int
    budget: int


def _get_max_and_second_max_probabilities(
    prob_dict: Dict[str, float],
) -> Tuple[float, float, str, str]:
    """获取概率字典中最大和次大的概率值

    Args:
        prob_dict: 值到概率的映射字典 {value: probability}

    Returns:
        (max_prob, second_max_prob, max_key, second_max_key)
    """
    if not prob_dict:
        return 1.0, 1.0, "", ""

    sorted_items = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_items) == 1:
        # 只有一个值，second_max 设为相同值避免除零
        return sorted_items[0][1], sorted_items[0][1], sorted_items[0][0], sorted_items[0][0]

    max_key, max_prob = sorted_items[0]
    second_max_key, second_max_prob = sorted_items[1]

    # 避免除零，如果次大概率为0，设为很小的值
    if second_max_prob == 0:
        second_max_prob = 1e-10

    return max_prob, second_max_prob, max_key, second_max_key


class CellConfidenceCalculator:
    """单元格置信度计算器

    基于共现矩阵和概率映射计算每个单元格的置信度分数。
    置信度越低的单元格越可能是错误的，需要 LLM 二次确认。

    实现逻辑：
    1. 计算共现频率矩阵，按列独立归一化
    2. 计算 pro_map (max_prob / second_max_prob)，取对数后归一化
    3. 组合: result = (1 + pro_map) * cooccurrence_freq
    4. 取各相关列的平均值作为最终置信度分数
    """

    def __init__(
        self,
        data: pd.DataFrame,
        occurrence_matrix: Dict,
        edges: List[List[str]],
        probabilities: Optional[Dict[Tuple[int, str], Dict[str, float]]] = None,
    ):
        """初始化置信度计算器

        Args:
            data: 数据 DataFrame（通常是修复后的数据）
            occurrence_matrix: 值共现矩阵
            edges: BN 网络的边列表，格式为 [[parent, child], ...]
            probabilities: BN 推理的概率映射，格式为 {(row_idx, col): {value: prob}}
        """
        self.data = data
        self.occurrence_matrix = occurrence_matrix
        self.edges = edges
        self.probabilities = probabilities or {}
        self._relationship_cache: Dict[str, List[str]] = {}

    def get_related_columns(self, column: str) -> List[str]:
        """获取与指定列相关的列（基于 BN 边）

        Args:
            column: 目标列名

        Returns:
            相关列名列表
        """
        if column in self._relationship_cache:
            return self._relationship_cache[column]

        related = []
        for edge in self.edges:
            if len(edge) >= 2:
                if edge[0] == column:
                    related.append(edge[1])
                elif edge[1] == column:
                    related.append(edge[0])

        self._relationship_cache[column] = related
        return related

    def _get_non_isolated_columns(self) -> List[str]:
        """获取非孤立节点列（有边连接的列）

        Returns:
            非孤立列名列表
        """
        non_isolated = []
        for column in self.data.columns:
            if len(self.get_related_columns(column)) > 0:
                non_isolated.append(column)
        return non_isolated

    def _compute_column_cooccurrence_matrix(
        self,
        column_name: str,
    ) -> np.ndarray:
        """计算指定列与其相关列的共现频率矩阵

        对整列数据计算共现频率，并按列独立归一化。

        Args:
            column_name: 目标列名

        Returns:
            shape 为 (行数, 相关列数) 的归一化共现频率矩阵
        """
        related_cols = self.get_related_columns(column_name)

        if not related_cols:
            return np.zeros((len(self.data), 0))

        column_series = self.data[column_name]
        num_rows = len(column_series)

        # 预分配结果数组
        result = np.zeros((num_rows, len(related_cols)))

        # 填充共现频率
        for col_idx, related_col in enumerate(related_cols):
            related_values = self.data[related_col].values

            for i, (idx, value) in enumerate(column_series.items()):
                related_value = related_values[i]

                # 获取共现次数
                try:
                    count = (
                        self.occurrence_matrix.get(column_name, {})
                        .get(value, {})
                        .get(related_col, {})
                        .get(related_value, 0)
                    )
                except (KeyError, TypeError):
                    count = 0

                result[i, col_idx] = count

        # 按列独立归一化（关键步骤）
        for col in range(result.shape[1]):
            col_data = result[:, col]
            min_val = np.min(col_data)
            max_val = np.max(col_data)
            if max_val > min_val:
                result[:, col] = (col_data - min_val) / (max_val - min_val)

        return result

    def _compute_column_pro_map(
        self,
        column_name: str,
    ) -> np.ndarray:
        """计算指定列的 pro_map (概率置信度映射)

        pro_map = max_prob / second_max_prob，取对数后归一化到 [0, 1]

        Args:
            column_name: 目标列名

        Returns:
            shape 为 (行数, 1) 的归一化 pro_map 数组
        """
        num_rows = len(self.data)
        pro_map_ndarray = np.zeros((num_rows, 1))

        for (row_idx, col), prob_dict in self.probabilities.items():
            if col == column_name and row_idx < num_rows:
                max_prob, second_max_prob, _, _ = _get_max_and_second_max_probabilities(prob_dict)
                # 计算 pro_map 值并取对数
                pro_map_value = max_prob / second_max_prob
                pro_map_ndarray[row_idx] = np.log(pro_map_value) if pro_map_value > 0 else 0

        # 归一化到 [0, 1]
        min_val = np.min(pro_map_ndarray)
        max_val = np.max(pro_map_ndarray)
        if max_val > min_val:
            pro_map_ndarray = (pro_map_ndarray - min_val) / (max_val - min_val)
        else:
            # 如果所有值相同，设置为 0
            pro_map_ndarray = np.zeros_like(pro_map_ndarray)

        return pro_map_ndarray

    def _compute_column_confidence_features(
        self,
        column_name: str,
    ) -> Dict[Tuple[int, str], np.ndarray]:
        """计算指定列所有单元格的置信度特征向量

        组合方式: result = (1 + pro_map) * cooccurrence_freq

        Args:
            column_name: 目标列名

        Returns:
            单元格到特征向量的映射 {(row_idx, col_name): feature_vector}
        """
        # 计算共现频率矩阵（已按列归一化）
        cooccurrence_matrix = self._compute_column_cooccurrence_matrix(column_name)

        if cooccurrence_matrix.shape[1] == 0:
            # 无相关列，返回空结果
            return {}

        # 计算 pro_map
        pro_map_ndarray = self._compute_column_pro_map(column_name)

        # 组合: (1 + pro_map) * cooccurrence_freq
        result = (1 + pro_map_ndarray) * cooccurrence_matrix

        # 构建结果字典
        cell_to_vector = {}
        for idx in range(result.shape[0]):
            cell_to_vector[(idx, column_name)] = result[idx]

        return cell_to_vector

    def compute_scores(
        self,
        cells: List[Tuple[int, str]],
    ) -> Dict[Tuple[int, str], float]:
        """计算单元格的置信度分数

        对所有非孤立列的全部行计算置信度特征，然后取平均值作为最终分数。
        置信度分数越低表示该单元格越可能是错误的。

        Args:
            cells: 需要计算置信度的单元格列表 [(row_idx, col_name), ...]

        Returns:
            单元格置信度分数字典
        """
        if not cells:
            return {}

        # 获取非孤立列
        non_isolated_cols = self._get_non_isolated_columns()

        # 计算所有非孤立列的置信度特征向量
        confidence_vectors: Dict[Tuple[int, str], np.ndarray] = {}
        for col in non_isolated_cols:
            col_features = self._compute_column_confidence_features(col)
            confidence_vectors.update(col_features)

        # 将向量转换为标量（取平均值）
        confidence_scores: Dict[Tuple[int, str], float] = {}
        for cell in cells:
            if cell in confidence_vectors:
                vector = confidence_vectors[cell]
                confidence_scores[cell] = float(np.mean(vector)) if len(vector) > 0 else 0.5
            else:
                # 孤立节点或不在计算范围内，设置为中等置信度
                confidence_scores[cell] = 0.5

        return confidence_scores


class LowConfidenceCellSelector:
    """低置信度单元格选择器

    从候选单元格中选择置信度最低的 top-k 个单元格进行 LLM 二次确认。
    """

    def __init__(
        self,
        data: pd.DataFrame,
        occurrence_matrix: Dict,
        edges: List[List[str]],
        probabilities: Optional[Dict[Tuple[int, str], Dict[str, float]]] = None,
    ):
        """初始化选择器

        Args:
            data: 数据 DataFrame
            occurrence_matrix: 值共现矩阵
            edges: BN 网络的边列表
            probabilities: BN 推理的概率映射，格式为 {(row_idx, col): {value: prob}}
        """
        self.calculator = CellConfidenceCalculator(data, occurrence_matrix, edges, probabilities)

    def select(
        self,
        cells: List[Tuple[int, str]],
        budget: int,
    ) -> CellSelectionResult:
        """选择低置信度单元格

        Args:
            cells: 候选单元格列表
            budget: 最大选择数量

        Returns:
            选择结果
        """
        if not cells:
            return CellSelectionResult(
                selected_cells=[],
                confidence_scores={},
                total_candidates=0,
                budget=budget,
            )

        # 计算置信度分数
        confidence_scores = self.calculator.compute_scores(cells)

        # 如果预算足够，返回所有单元格
        if budget <= 0 or budget >= len(cells):
            return CellSelectionResult(
                selected_cells=cells,
                confidence_scores=confidence_scores,
                total_candidates=len(cells),
                budget=budget,
            )

        # 按置信度升序排序（选择置信度最低的）
        sorted_cells = sorted(
            cells,
            key=lambda cell: confidence_scores.get(cell, 0.0),
        )

        return CellSelectionResult(
            selected_cells=sorted_cells[:budget],
            confidence_scores=confidence_scores,
            total_candidates=len(cells),
            budget=budget,
        )


def compute_llm_repair_budget(
    total_cells: int,
    budget_ratio: float = 0.01,
    max_budget: int = -1,
    candidate_count: Optional[int] = None,
) -> int:
    """计算 LLM 修复预算

    Args:
        total_cells: 数据总单元格数（行数 * 列数）
        budget_ratio: 预算比例（如 0.01 表示 1%）
        max_budget: 最大预算数量，-1 表示使用比例计算
        candidate_count: 候选单元格数量，用于限制最终预算

    Returns:
        计算后的预算值
    """
    if max_budget > 0:
        budget = max_budget
    else:
        budget = int(total_cells * budget_ratio)

    # 如果提供了候选数量，确保预算不超过候选数
    if candidate_count is not None:
        budget = min(budget, candidate_count)

    return max(0, budget)


__all__ = [
    "CellSelectionResult",
    "CellConfidenceCalculator",
    "LowConfidenceCellSelector",
    "compute_llm_repair_budget",
]
