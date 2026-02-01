"""BN 结构构建器模块

基于 LLM 生成的函数依赖和 AFD 分数构建贝叶斯网络结构。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import pandas as pd
from scipy.spatial.distance import euclidean

from ..utils.logger import get_logger


class LLMBDEUConstructor:
    """基于 LLM + BDEU 的 BN 结构构建器

    结合 LLM 生成的函数依赖矩阵和近似函数依赖（AFD）分数，
    构建贝叶斯网络的初始结构。

    构建流程：
    1. 等价节点分区：将具有强相互依赖的属性分组
    2. 确定初始边：基于混合分数选择最佳父节点
    3. 环检测：确保生成的结构是 DAG
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.1,
    ):
        """初始化构建器

        Args:
            alpha: 等价分区的欧氏距离阈值
            beta: 边选择的分数阈值
        """
        self.alpha = alpha
        self.beta = beta
        self.logger = get_logger()

    def construct(
        self,
        score_afd: pd.DataFrame,
        score_llm: pd.DataFrame,
    ) -> Dict:
        """构建 BN 结构

        Args:
            score_afd: AFD 分数矩阵（mu_plus）
            score_llm: LLM FD 分数矩阵

        Returns:
            {
                "nodes": List[str],           # 节点列表
                "consistencies": List[List[str]],  # 一致性组列表
                "initial_edges": List[List[str]]   # 初始边列表
            }
        """
        result = {
            "nodes": score_afd.columns.tolist(),
            "consistencies": [],
            "initial_edges": [],
        }

        # Step 1: 等价节点分区
        groups = self._equivalence_node_partition(score_afd, score_llm)
        self.logger.info(f"等价节点分区: {groups}")

        # 记录一致性组（长度 > 1 的组）
        for group in groups:
            if len(group) > 1:
                result["consistencies"].append(group)

        # Step 2: 计算混合分数
        filtered_score = self._filter_positive_values(score_afd, score_llm)
        hybrid_score = self._compute_hybrid_score(score_afd, score_llm)

        # Step 3: 为非一致性组的节点选择初始边
        first_nodes = [g[0] for g in groups if g]

        for group in groups:
            if len(group) == 1:
                col = group[0]
                # 找最佳父节点
                max_idx = filtered_score[col].idxmax()

                if hybrid_score.loc[max_idx, col] == 0:
                    continue

                # 如果最佳父节点在一致性组中，使用组的第一个节点
                parent = max_idx
                for cons_group in result["consistencies"]:
                    if parent in cons_group:
                        parent = cons_group[0]
                        break

                # 检查是否会形成环
                new_edge = [parent, col]
                if not self._would_create_cycle(result["initial_edges"], new_edge):
                    result["initial_edges"].append(new_edge)

        self.logger.info(f"初始边: {result['initial_edges']}")
        return result

    def _equivalence_node_partition(
        self,
        score_afd: pd.DataFrame,
        score_llm: pd.DataFrame,
    ) -> List[List[str]]:
        """等价节点分区

        将具有强相互依赖关系的属性分到同一组。

        分组条件：
        1. 两个属性之间存在相互的正向 FD
        2. AFD 分数向量的欧氏距离不超过 alpha
        """
        filtered_score = self._filter_positive_values(score_afd, score_llm)
        all_columns = score_afd.columns.tolist()

        if not all_columns:
            return []

        groups = []
        processed = set()

        for col in all_columns:
            if col in processed:
                continue

            # 开始新组
            current_group = [col]
            processed.add(col)

            # 尝试扩展组
            expanded = True
            while expanded:
                expanded = False

                for other_col in all_columns:
                    if other_col in processed:
                        continue

                    # 检查是否与组内所有成员有相互依赖
                    can_add = True
                    for group_col in current_group:
                        # 条件 1: 相互正向 FD
                        if (
                            filtered_score[group_col][other_col] == 0
                            or filtered_score[other_col][group_col] == 0
                        ):
                            can_add = False
                            break

                        # 条件 2: 欧氏距离
                        distance = self._euclidean_distance(
                            score_afd.loc[group_col], score_afd.loc[other_col]
                        )
                        if distance > self.alpha:
                            can_add = False
                            break

                    if can_add:
                        current_group.append(other_col)
                        processed.add(other_col)
                        expanded = True

            groups.append(current_group)

        return groups

    def _filter_positive_values(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
    ) -> pd.DataFrame:
        """过滤正值：只保留两个矩阵都为正的值"""
        mask = (df1 != 0) & (df2 > 0)
        return df1.where(mask, 0)

    def _compute_hybrid_score(
        self,
        score_afd: pd.DataFrame,
        score_llm: pd.DataFrame,
    ) -> pd.DataFrame:
        """计算混合分数

        当 LLM 认为没有 FD（负值）但 AFD 分数较高时，
        应用惩罚系数。
        """
        hybrid = score_afd.copy()

        for col in score_afd.columns:
            for idx in score_afd.index:
                if score_llm[col][idx] < 0:  # LLM 认为没有 FD
                    if score_afd[col][idx] > self.beta:  # 但 AFD 分数高
                        # 应用惩罚
                        hybrid[col][idx] = score_afd[col][idx] * (1 + score_llm[col][idx])

        return hybrid

    def _euclidean_distance(self, row1: pd.Series, row2: pd.Series) -> float:
        """计算两行之间的欧氏距离"""
        valid_idx = ~row1.isna() & ~row2.isna()
        if valid_idx.sum() == 0:
            return float("inf")
        return euclidean(row1[valid_idx], row2[valid_idx])

    def _would_create_cycle(
        self,
        edges: List[List[str]],
        new_edge: List[str],
    ) -> bool:
        """检查添加新边是否会创建环"""
        G = nx.DiGraph()

        for edge in edges:
            G.add_edge(edge[0], edge[1])

        G.add_edge(new_edge[0], new_edge[1])

        return not nx.is_directed_acyclic_graph(G)


def generate_bn_structure(
    dataset_name: str,
    model_name: str,
    score_dir: Path = Path("data/score"),
    fd_dir: Path = Path("data/llm_generated_func_deps"),
    discrete_cols_path: Optional[Path] = None,
    alpha: float = 0.1,
    beta: float = 0.1,
) -> Dict:
    """便捷函数：生成 BN 结构

    Args:
        dataset_name: 数据集名称
        model_name: 模型名称
        score_dir: AFD 分数目录
        fd_dir: LLM FD 目录
        discrete_cols_path: 离散列配置路径
        alpha: 等价分区阈值
        beta: 边选择阈值

    Returns:
        BN 结构字典
    """
    # 默认路径
    afd_path = score_dir / dataset_name / "mu_plus.csv"
    llm_path = fd_dir / dataset_name / f"DFD_{model_name}.csv"

    if discrete_cols_path is None:
        discrete_cols_path = Path(f"data/datasets/{dataset_name}/discrete_cols.json")

    # 加载离散列
    with open(discrete_cols_path, "r", encoding="utf-8") as f:
        discrete_cols = json.load(f)

    # 加载分数矩阵
    score_afd = pd.read_csv(afd_path, index_col=0)[discrete_cols]
    score_llm = pd.read_csv(llm_path, index_col=0)[discrete_cols]

    # 构建结构
    constructor = LLMBDEUConstructor(alpha=alpha, beta=beta)
    return constructor.construct(score_afd, score_llm)


def load_bn_structure(path: Path) -> Dict:
    """加载已保存的 BN 结构"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_bn_structure(structure: Dict, path: Path) -> None:
    """保存 BN 结构"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(structure, f, indent=4, ensure_ascii=False)


__all__ = [
    "LLMBDEUConstructor",
    "generate_bn_structure",
    "load_bn_structure",
    "save_bn_structure",
]
