"""一致性修复模块

基于等价节点分区进行一致性修复，通过多数投票策略推断正确值。
"""

from typing import List

import pandas as pd

from ..utils.logger import get_logger


class ConsistencyRepairer:
    """一致性修复器

    对于具有一致性约束的属性组（等价节点分区），使用多数投票
    策略进行修复。当多个属性之间存在强依赖关系时，可以通过
    其他属性的值来推断当前属性的正确值。
    """

    def __init__(self, null_marker: str = "[NULL Cell]"):
        """初始化一致性修复器

        Args:
            null_marker: 空值标记
        """
        self.null_marker = null_marker
        self.logger = get_logger()

    def repair(self, data: pd.DataFrame, consistencies: List[List[str]]) -> pd.DataFrame:
        """执行一致性修复

        对每个一致性组执行修复，返回修复后的数据。

        Args:
            data: 待修复的数据
            consistencies: 一致性组列表，每个组是一个属性名列表

        Returns:
            修复后的数据
        """
        if not consistencies:
            return data.copy()

        repaired_data = data.copy()

        for consistency_group in consistencies:
            if len(consistency_group) <= 1:
                continue

            self.logger.debug(f"处理一致性组: {consistency_group}")
            repaired_data = self._repair_consistency_group(repaired_data, consistency_group)

        return repaired_data

    def _repair_consistency_group(
        self, data: pd.DataFrame, consistencies: List[str]
    ) -> pd.DataFrame:
        """修复单个一致性组

        使用多数投票策略：对于每个属性，根据其他属性的值
        来推断最可能的正确值。

        Args:
            data: 数据
            consistencies: 一致性组中的属性列表

        Returns:
            修复后的数据
        """
        repaired_data = data.copy()

        for current_col in consistencies:
            other_cols = [c for c in consistencies if c != current_col]

            # 预计算每个条件列的频率表
            freq_tables = {}
            for other_col in other_cols:
                # 过滤掉 NULL 值
                valid_data = data[data[other_col] != self.null_marker]
                if len(valid_data) == 0:
                    continue

                # 按 (条件列, 目标列) 分组统计频率
                freq = valid_data.groupby([other_col, current_col]).size().reset_index(name="count")

                # 找出每个条件值对应的目标列最常见值
                idx_max = freq.groupby(other_col)["count"].idxmax()
                mode_table = freq.loc[idx_max].set_index(other_col)
                freq_tables[other_col] = mode_table

            if not freq_tables:
                continue

            # 准备投票数据
            vote_data = []
            for other_col, mode_table in freq_tables.items():
                recommended_values = data[other_col].map(mode_table[current_col].to_dict())
                vote_weights = data[other_col].map(mode_table["count"].to_dict())
                vote_data.append({"values": recommended_values, "weights": vote_weights})

            # 批量处理投票汇总
            def aggregate_votes(row_idx: int) -> str:
                vote_counts = {}

                for vote_dict in vote_data:
                    val = vote_dict["values"].iloc[row_idx]
                    weight = vote_dict["weights"].iloc[row_idx]

                    # 只统计有效的投票
                    if pd.notna(val) and pd.notna(weight):
                        vote_counts[val] = vote_counts.get(val, 0) + weight

                if not vote_counts:
                    return None

                # 找出得票最高的值
                max_count = max(vote_counts.values())
                max_keys = [k for k, v in vote_counts.items() if v == max_count]

                # 只有唯一最大值且不是 NULL 才使用
                if len(max_keys) == 1 and max_keys[0] != self.null_marker:
                    return max_keys[0]

                return None

            # 批量计算所有行的推断值
            inferred_values = pd.Series(range(len(data))).apply(aggregate_votes)

            # 批量更新（只更新非 None 的值）
            mask = inferred_values.notna()
            repaired_data.loc[mask, current_col] = inferred_values[mask]

        return repaired_data

    def get_consistency_columns(self, consistencies: List[List[str]]) -> List[str]:
        """获取所有一致性修复涉及的列（除每组第一个外）

        一致性组中，第一个列通常是"主列"，其余列会被一致性修复。
        这些被修复的列后续不需要通过贝叶斯网络推理。

        Args:
            consistencies: 一致性组列表

        Returns:
            需要排除的列名列表
        """
        excluded_cols = []
        for group in consistencies:
            if len(group) > 1:
                excluded_cols.extend(group[1:])
        return excluded_cols


def repair_with_consistency(
    data: pd.DataFrame,
    consistencies: List[List[str]],
    null_marker: str = "[NULL Cell]",
) -> pd.DataFrame:
    """便捷函数：执行一致性修复

    Args:
        data: 待修复的数据
        consistencies: 一致性组列表
        null_marker: 空值标记

    Returns:
        修复后的数据
    """
    repairer = ConsistencyRepairer(null_marker=null_marker)
    return repairer.repair(data, consistencies)


__all__ = [
    "ConsistencyRepairer",
    "repair_with_consistency",
]
