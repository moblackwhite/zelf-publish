"""数据修复器模块

基于贝叶斯网络推理和共现矩阵进行数据修复。
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from tqdm import tqdm

from ..utils.logger import get_logger


@dataclass
class RepairResult:
    """修复结果数据类"""

    repaired_data: pd.DataFrame
    repairs: Dict[Tuple[int, str], str] = field(default_factory=dict)
    probabilities: Dict[Tuple[int, str], Dict[str, float]] = field(default_factory=dict)
    penalties: Dict[Tuple[int, str], Dict[str, float]] = field(default_factory=dict)
    query_contexts: Dict[str, str] = field(default_factory=dict)


class DataRepairer:
    """数据修复器

    结合贝叶斯网络推理和共现矩阵的补偿分数进行数据修复。

    修复流程：
    1. 对于每个单元格，计算基于 BN 的条件概率
    2. 计算基于共现矩阵的补偿分数
    3. 综合两个分数选择最佳修复值
    """

    def __init__(
        self,
        partitions: Dict[str, BayesianNetwork],
        occurrence_matrix: Dict,
        null_marker: str = "[NULL Cell]",
        use_compensation_score: bool = True,
    ):
        """初始化数据修复器

        Args:
            partitions: 每个列的局部贝叶斯网络模型
            occurrence_matrix: 值共现矩阵
            null_marker: 空值标记
            use_compensation_score: 是否使用补偿分数
        """
        self.partitions = partitions
        self.occurrence_matrix = occurrence_matrix
        self.null_marker = null_marker
        self.use_compensation_score = use_compensation_score
        self.logger = get_logger()

    def repair(
        self,
        dirty_data: pd.DataFrame,
        target_columns: Optional[List[str]] = None,
        n_jobs: int = -1,
        chunk_size: int = 50,
    ) -> RepairResult:
        """修复脏数据

        Args:
            dirty_data: 待修复的脏数据
            target_columns: 需要修复的目标列，默认为所有列
            n_jobs: 并行作业数，-1 表示使用所有核心
            chunk_size: 每个处理块的行数

        Returns:
            修复结果
        """
        if target_columns is None:
            target_columns = dirty_data.columns.tolist()

        repaired_data = dirty_data.copy()
        all_repairs = {}
        all_probabilities = {}
        all_penalties = {}
        all_query_contexts = {}

        # 按块分割数据
        chunks = []
        total_rows = len(dirty_data)
        for i in range(0, total_rows, chunk_size):
            end = min(i + chunk_size, total_rows)
            chunks.append((i, end))

        self.logger.info(f"开始修复 {total_rows} 行数据，分 {len(chunks)} 个块处理")

        # 并行处理
        if n_jobs != 1 and len(chunks) > 1:
            with ProcessPoolExecutor(max_workers=None if n_jobs == -1 else n_jobs) as executor:
                futures = []
                for start, end in tqdm(chunks, desc="提交任务", unit="块"):
                    future = executor.submit(
                        self._repair_chunk,
                        dirty_data,
                        start,
                        end,
                        target_columns,
                    )
                    futures.append(future)

                for future in tqdm(futures, desc="处理结果", unit="块"):
                    chunk_result = future.result()
                    self._merge_chunk_result(
                        repaired_data,
                        chunk_result,
                        all_repairs,
                        all_probabilities,
                        all_penalties,
                        all_query_contexts,
                    )
        else:
            # 串行处理
            for start, end in tqdm(chunks, desc="修复数据", unit="块"):
                chunk_result = self._repair_chunk(dirty_data, start, end, target_columns)
                self._merge_chunk_result(
                    repaired_data,
                    chunk_result,
                    all_repairs,
                    all_probabilities,
                    all_penalties,
                    all_query_contexts,
                )

        return RepairResult(
            repaired_data=repaired_data,
            repairs=all_repairs,
            probabilities=all_probabilities,
            penalties=all_penalties,
            query_contexts=all_query_contexts,
        )

    def _repair_chunk(
        self,
        dirty_data: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        target_columns: List[str],
    ) -> Dict:
        """处理数据块"""
        chunk_data = dirty_data.iloc[start_idx:end_idx].copy()
        result = {
            "data": {},
            "repairs": {},
            "probabilities": {},
            "penalties": {},
            "query_contexts": {},
        }

        for i in range(len(chunk_data)):
            row_idx = start_idx + i
            row_result = self._repair_row(
                chunk_data.iloc[[i]],
                row_idx,
                target_columns,
                dirty_data,
            )
            result["data"][row_idx] = row_result["data"]
            result["repairs"].update(row_result["repairs"])
            result["probabilities"].update(row_result["probabilities"])
            result["penalties"].update(row_result["penalties"])
            result["query_contexts"].update(row_result["query_contexts"])

        return result

    def _repair_row(
        self,
        row_data: pd.DataFrame,
        row_idx: int,
        target_columns: List[str],
        full_data: pd.DataFrame,
    ) -> Dict:
        """修复单行数据"""
        repairs = {}
        row_copy = row_data.copy()
        prob_mappings = {}
        penalty_mappings = {}
        query_contexts = {}

        for column in target_columns:
            model = self.partitions.get(column)
            if model is None:
                continue

            # 获取当前值
            observed_value = row_copy.iloc[0][column]

            # 孤立节点跳过
            parents = list(model.get_parents(column))
            children = list(model.get_children(column))
            if len(parents) == 0 and len(children) == 0:
                continue

            # 获取值域
            domain = full_data[column].unique().tolist()

            # 计算补偿分数
            penalties = self._calculate_penalties(
                network=model,
                attribute=column,
                row_idx=row_idx,
                row_data=row_copy,
                domain=domain,
            )
            penalty_mappings[(row_idx, column)] = penalties.copy()

            # 计算条件概率并生成查询上下文
            probabilities, contexts = self._calculate_probabilities_with_context(
                model=model,
                node=column,
                row_data=row_copy,
                domain=domain,
            )
            prob_mappings[(row_idx, column)] = probabilities.copy()

            # 选择最佳修复值
            best_value = self._select_best_value(domain, penalties, probabilities)

            # 更新修复结果
            if best_value and best_value != observed_value:
                row_copy.iloc[0, row_copy.columns.get_loc(column)] = best_value
                repairs[(row_idx, column)] = best_value
                query_contexts[f"({row_idx}, {column})"] = contexts.get(best_value, "")

        return {
            "id": row_idx,
            "data": row_copy,
            "repairs": repairs,
            "probabilities": prob_mappings,
            "penalties": penalty_mappings,
            "query_contexts": query_contexts,
        }

    def _calculate_penalties(
        self,
        network: BayesianNetwork,
        attribute: str,
        row_idx: int,
        row_data: pd.DataFrame,
        domain: List[str],
    ) -> Dict[str, float]:
        """计算补偿惩罚得分

        基于共现矩阵计算每个候选值的补偿分数。
        """
        penalties = {}
        cooccurrence_dist = {}
        total_cooccur = 0

        for value in domain:
            if value == self.null_marker:
                penalties[value] = 0
                continue

            # 计算与其他属性的共现得分
            cooccur_vector = []
            for other_attr in row_data.columns:
                # 跳过当前属性和相关节点
                if (
                    other_attr == attribute
                    or other_attr in network.get_parents(attribute)
                    or other_attr in network.get_children(attribute)
                ):
                    continue

                other_value = row_data.iloc[0][other_attr]

                # 查找共现次数
                try:
                    score = self.occurrence_matrix[attribute][value][other_attr].get(other_value, 0)
                except KeyError:
                    score = 0

                cooccur_vector.append(score)

            # 计算 L2 范数
            cooccurrence_dist[value] = np.linalg.norm(cooccur_vector, ord=2) + 1
            total_cooccur += cooccurrence_dist[value]

        # 归一化
        for value in domain:
            if value != self.null_marker:
                penalties[value] = cooccurrence_dist[value] / total_cooccur

        return penalties

    def _calculate_probabilities_with_context(
        self,
        model: BayesianNetwork,
        node: str,
        row_data: pd.DataFrame,
        domain: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, str]]:
        """计算条件概率并生成 LLM 查询上下文"""
        probabilities = {}
        query_contexts = {}
        node_row_data = row_data.iloc[0].copy()

        def get_prob_from_cpd(cpd, node_value=None):
            """从 CPD 表中获取概率"""
            indices = []
            for attr in cpd.state_names:
                if attr == node:
                    value_to_use = node_value if node_value else node_row_data[attr]
                    indices.append(cpd.state_names[attr].index(value_to_use))
                else:
                    indices.append(cpd.state_names[attr].index(node_row_data[attr]))
            return cpd.values[tuple(indices)]

        for value in domain:
            cpd = model.get_cpds(node)
            parents = cpd.variables[1:]

            # 父节点只有一个且为 NULL 时跳过
            parent_is_null = len(parents) == 1 and node_row_data[parents[0]] == self.null_marker

            if parent_is_null:
                prob = 1.0
            else:
                prob = get_prob_from_cpd(cpd, value)

            # 计算子节点的概率
            for child in model.get_children(node):
                child_value = node_row_data[child]
                if child_value == self.null_marker:
                    continue

                child_cpd = model.get_cpds(child)
                child_prob = get_prob_from_cpd(child_cpd, value)
                prob *= child_prob

            probabilities[value] = prob

            # 生成 LLM 查询上下文
            query_contexts[value] = self._generate_query_context(model, node, row_data, value, prob)

        return probabilities, query_contexts

    def _generate_query_context(
        self,
        model: BayesianNetwork,
        node: str,
        row_data: pd.DataFrame,
        value: str,
        probability: float,
    ) -> str:
        """为特定单元格生成 LLM 查询上下文"""
        node_row_data = row_data.iloc[0]

        parents = list(model.get_parents(node))
        children = list(model.get_children(node))

        # 构建条件部分（父节点）
        conditions = []
        for parent in parents:
            parent_value = node_row_data[parent]
            if parent_value != self.null_marker:
                conditions.append(f"{parent} is {parent_value}")

        # 构建结果部分（子节点）
        results = []
        for child in children:
            child_value = node_row_data[child]
            if child_value != self.null_marker:
                results.append(f"{child} is {child_value}")

        # 概率字符串
        prob_str = f" (approximately {probability * 100:.1f}%)"

        # 生成查询文本
        if conditions and results:
            condition_str = " and ".join(conditions)
            result_str = " and ".join(results)
            query = (
                f"Given that {condition_str}, the probability that "
                f"{node} is {value} and {result_str} is{prob_str}"
            )
        elif conditions:
            condition_str = " and ".join(conditions)
            query = (
                f"Given that {condition_str}, the probability that {node} is {value} is{prob_str}"
            )
        elif results:
            result_str = " and ".join(results)
            query = f"The probability that {node} is {value} and {result_str} is{prob_str}"
        else:
            query = f"The probability that {node} is {value} is{prob_str}"

        return query

    def _select_best_value(
        self,
        domain: List[str],
        penalties: Dict[str, float],
        probabilities: Dict[str, float],
    ) -> Optional[str]:
        """选择最佳修复值"""
        best_score = -1
        best_value = None

        for value in domain:
            if value == self.null_marker:
                continue

            penalty = penalties.get(value, 0)
            prob = probabilities.get(value, 0)

            if self.use_compensation_score:
                score = penalty * prob
            else:
                score = prob

            if score > best_score:
                best_score = score
                best_value = value

        return best_value

    def _merge_chunk_result(
        self,
        repaired_data: pd.DataFrame,
        chunk_result: Dict,
        all_repairs: Dict,
        all_probabilities: Dict,
        all_penalties: Dict,
        all_query_contexts: Dict,
    ) -> None:
        """合并块处理结果"""
        for idx, row_data in chunk_result["data"].items():
            repaired_data.iloc[idx] = row_data.iloc[0]

        all_repairs.update(chunk_result["repairs"])
        all_probabilities.update(chunk_result["probabilities"])
        all_penalties.update(chunk_result["penalties"])
        all_query_contexts.update(chunk_result["query_contexts"])


def repair_dirty_data(
    dirty_data: pd.DataFrame,
    partitions: Dict[str, BayesianNetwork],
    occurrence_matrix: Dict,
    target_columns: Optional[List[str]] = None,
    n_jobs: int = -1,
    null_marker: str = "[NULL Cell]",
    use_compensation_score: bool = True,
) -> RepairResult:
    """修复脏数据的便捷函数

    Args:
        dirty_data: 待修复的脏数据
        partitions: 每个列的局部贝叶斯网络模型
        occurrence_matrix: 值共现矩阵
        target_columns: 需要修复的目标列
        n_jobs: 并行作业数
        null_marker: 空值标记
        use_compensation_score: 是否使用补偿分数

    Returns:
        修复结果
    """
    repairer = DataRepairer(
        partitions=partitions,
        occurrence_matrix=occurrence_matrix,
        null_marker=null_marker,
        use_compensation_score=use_compensation_score,
    )

    return repairer.repair(
        dirty_data=dirty_data,
        target_columns=target_columns,
        n_jobs=n_jobs,
    )


__all__ = [
    "RepairResult",
    "DataRepairer",
    "repair_dirty_data",
]
