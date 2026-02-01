"""贝叶斯网络构建模块

提供贝叶斯网络结构学习和局部网络管理功能。
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pgmpy.estimators import BayesianEstimator, BDeuScore, HillClimbSearch
from pgmpy.models import BayesianNetwork

from ..utils.logger import get_logger


class BuildMethod(str, Enum):
    """贝叶斯网络构建方法枚举

    Attributes:
        HYBRID: LLM + BDEU 混合构建（默认，最完整的方法）
        HYBRID_NO_CONSISTENCY: 混合构建但不做一致性修复
        BDEU_ONLY: 仅使用 BDEU 评分 + HillClimb 搜索
        LLM_ONLY: 仅使用 LLM 生成的边结构
        USER_DEFINED: 用户自定义结构
    """

    HYBRID = "HYBRID"
    HYBRID_NO_CONSISTENCY = "HYBRID_NO_CONSISTENCY"
    BDEU_ONLY = "BDEU_ONLY"
    LLM_ONLY = "LLM_ONLY"
    USER_DEFINED = "USER_DEFINED"


class BNBuilder(ABC):
    """贝叶斯网络构建器抽象基类"""

    @abstractmethod
    def build(
        self, data: pd.DataFrame, initial_edges: Optional[List[Tuple[str, str]]] = None
    ) -> BayesianNetwork:
        """构建贝叶斯网络结构

        Args:
            data: 训练数据
            initial_edges: 初始边列表（可选）

        Returns:
            构建好的贝叶斯网络
        """
        pass


class BDEUBNBuilder(BNBuilder):
    """基于 BDEU 评分的贝叶斯网络构建器

    使用 HillClimbSearch 和 BDeuScore 进行结构学习。
    """

    def __init__(self, equivalent_sample_size: int = 50):
        """初始化构建器

        Args:
            equivalent_sample_size: BDEU 评分的等效样本大小
        """
        self.equivalent_sample_size = equivalent_sample_size
        self.logger = get_logger()

    def build(
        self, data: pd.DataFrame, initial_edges: Optional[List[Tuple[str, str]]] = None
    ) -> BayesianNetwork:
        """使用 HillClimbSearch + BDeuScore 构建网络

        Args:
            data: 训练数据
            initial_edges: 固定边列表，这些边将被保留

        Returns:
            构建好的贝叶斯网络
        """
        fixed_edges = []
        if initial_edges:
            fixed_edges = [tuple(edge) for edge in initial_edges]

        self.logger.info(f"使用 BDEU 构建贝叶斯网络 (ESS={self.equivalent_sample_size})")

        hc = HillClimbSearch(data)
        model = hc.estimate(
            scoring_method=BDeuScore(data, equivalent_sample_size=self.equivalent_sample_size),
            fixed_edges=fixed_edges if fixed_edges else None,
        )

        self.logger.info(f"网络边: {list(model.edges())}")
        return model


class EdgeBasedBNBuilder(BNBuilder):
    """基于预定义边的贝叶斯网络构建器

    直接使用给定的边列表构建网络，不进行结构学习。
    """

    def __init__(self):
        self.logger = get_logger()

    def build(
        self, data: pd.DataFrame, initial_edges: Optional[List[Tuple[str, str]]] = None
    ) -> BayesianNetwork:
        """使用给定边列表构建网络

        Args:
            data: 用于获取节点列表的数据
            initial_edges: 边列表

        Returns:
            构建好的贝叶斯网络
        """
        network = BayesianNetwork()

        # 添加所有节点
        for col in data.columns:
            network.add_node(col)

        # 添加边
        if initial_edges:
            for src, tgt in initial_edges:
                network.add_edge(src, tgt)

        self.logger.info(f"基于边列表构建网络，共 {len(initial_edges or [])} 条边")
        return network


class HybridBNBuilder(BNBuilder):
    """混合贝叶斯网络构建器

    结合 LLM 生成的初始结构和 BDEU 优化。
    这是默认的推荐方法。
    """

    def __init__(
        self,
        equivalent_sample_size: int = 50,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ):
        """初始化混合构建器

        Args:
            equivalent_sample_size: BDEU 评分的等效样本大小
            alpha: LLM 结构权重参数
            beta: BDEU 结构权重参数
        """
        self.equivalent_sample_size = equivalent_sample_size
        self.alpha = alpha
        self.beta = beta
        self.logger = get_logger()

    def build(
        self, data: pd.DataFrame, initial_edges: Optional[List[Tuple[str, str]]] = None
    ) -> BayesianNetwork:
        """构建混合贝叶斯网络

        首先使用 LLM 生成的边作为初始结构，然后通过 BDEU 优化。

        Args:
            data: 训练数据
            initial_edges: LLM 生成的初始边列表

        Returns:
            优化后的贝叶斯网络
        """
        self.logger.info("使用混合方法构建贝叶斯网络")

        # 使用 BDEU 优化，将 LLM 边作为固定边
        bdeu_builder = BDEUBNBuilder(self.equivalent_sample_size)
        return bdeu_builder.build(data, initial_edges)


class LocalNetworkManager:
    """局部网络管理器

    为每个节点创建和管理局部子网络，用于推理阶段。
    """

    def __init__(self):
        self.logger = get_logger()

    def create_local_subnetworks(self, network: BayesianNetwork) -> Dict[str, BayesianNetwork]:
        """为每个节点创建局部子网络

        每个局部网络包含节点本身、其父节点和子节点。

        Args:
            network: 原始贝叶斯网络

        Returns:
            节点名到局部网络的映射字典
        """
        local_subnetworks = {}

        for node in network.nodes():
            subnetwork = BayesianNetwork()

            # 获取相关节点
            parents = list(network.get_parents(node))
            children = list(network.get_children(node))

            # 添加当前节点
            subnetwork.add_node(node)

            # 添加父节点和边
            for parent in parents:
                subnetwork.add_node(parent)
                subnetwork.add_edge(parent, node)

            # 添加子节点和边
            for child in children:
                subnetwork.add_node(child)
                subnetwork.add_edge(node, child)

            local_subnetworks[node] = subnetwork

        self.logger.debug(f"创建了 {len(local_subnetworks)} 个局部网络")
        return local_subnetworks

    def create_local_networks_from_edges(
        self, edge_dict: Dict[str, List[Tuple[str, str]]]
    ) -> Dict[str, BayesianNetwork]:
        """从边字典创建局部网络

        Args:
            edge_dict: 节点到相关边列表的映射

        Returns:
            节点名到局部网络的映射字典
        """
        local_networks = {}

        for node, edges in edge_dict.items():
            network = BayesianNetwork()
            network.add_node(node)

            for src, dst in edges:
                if src == node:
                    network.add_node(dst)
                    network.add_edge(src, dst)
                elif dst == node:
                    network.add_node(src)
                    network.add_edge(src, dst)

            local_networks[node] = network

        return local_networks

    def train_local_networks(
        self, data: pd.DataFrame, local_networks: Dict[str, BayesianNetwork]
    ) -> None:
        """训练局部网络的条件概率分布

        使用贝叶斯估计器为每个局部网络学习 CPD。

        Args:
            data: 训练数据
            local_networks: 局部网络字典（会被原地修改）
        """
        self.logger.info(f"训练 {len(local_networks)} 个局部网络的 CPD")

        for node, network in local_networks.items():
            # 获取局部网络需要的列
            parents = list(network.get_parents(node))
            children = list(network.get_children(node))
            relevant_cols = parents + children + [node]

            # 提取相关数据并训练
            relevant_data = data[relevant_cols]
            network.fit(relevant_data, estimator=BayesianEstimator)


def filter_weak_partitions(
    partitions: Dict[str, BayesianNetwork],
    strength_matrix: Optional[pd.DataFrame] = None,
    gamma: float = 0.25,
) -> Dict[str, BayesianNetwork]:
    """过滤掉弱依赖的网络分区

    当推断目标只有一条边且依赖强度不足时，舍弃该网络。

    Args:
        partitions: 局部网络字典
        strength_matrix: 依赖强度矩阵（mu_plus），如果为 None 则不过滤
        gamma: 强度阈值

    Returns:
        过滤后的局部网络字典
    """
    if strength_matrix is None:
        return partitions

    logger = get_logger()
    result = {}

    for key, subnetwork in partitions.items():
        edges = list(subnetwork.edges())
        should_keep = True

        # 只有一条边时检查强度
        if len(edges) == 1:
            src, dst = edges[0]
            if src == key:
                # 当前节点是源，检查 dst -> src 的强度
                if strength_matrix.loc[dst, src] < gamma:
                    should_keep = False
                    logger.debug(
                        f"过滤弱分区: {key} (强度 {strength_matrix.loc[dst, src]:.3f} < {gamma})"
                    )
            elif dst == key:
                # 当前节点是目标，检查 src -> dst 的强度
                if strength_matrix.loc[src, dst] < gamma:
                    should_keep = False
                    logger.debug(
                        f"过滤弱分区: {key} (强度 {strength_matrix.loc[src, dst]:.3f} < {gamma})"
                    )

        if should_keep:
            result[key] = subnetwork

    logger.info(f"过滤后保留 {len(result)}/{len(partitions)} 个分区")
    return result


def create_bn_builder(
    method: BuildMethod,
    equivalent_sample_size: int = 50,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
) -> BNBuilder:
    """工厂函数：根据方法创建对应的构建器

    Args:
        method: 构建方法
        equivalent_sample_size: BDEU 等效样本大小
        alpha: LLM 结构权重参数
        beta: BDEU 结构权重参数

    Returns:
        对应的 BNBuilder 实例
    """
    # 处理字符串输入
    if isinstance(method, str):
        # 兼容老方法名（已废弃，将在未来版本移除）
        legacy_method_map = {
            "LLM_BDEU_BN": BuildMethod.HYBRID,
            "LLM_BDEU_BN_NO_CONSIS": BuildMethod.HYBRID_NO_CONSISTENCY,
            "BDEU_BN": BuildMethod.BDEU_ONLY,
            "LLM_BN": BuildMethod.LLM_ONLY,
            "USER_BN": BuildMethod.USER_DEFINED,
        }
        if method in legacy_method_map:
            logger = get_logger()
            new_name = legacy_method_map[method].value
            logger.warning(f"方法名 '{method}' 已废弃，请使用 '{new_name}'")
            method = legacy_method_map[method]
        else:
            method = BuildMethod(method)

    if method in (BuildMethod.HYBRID, BuildMethod.HYBRID_NO_CONSISTENCY):
        return HybridBNBuilder(equivalent_sample_size, alpha, beta)
    elif method == BuildMethod.BDEU_ONLY:
        return BDEUBNBuilder(equivalent_sample_size)
    elif method in (BuildMethod.LLM_ONLY, BuildMethod.USER_DEFINED):
        return EdgeBasedBNBuilder()
    else:
        raise ValueError(f"未知的构建方法: {method}")


__all__ = [
    "BuildMethod",
    "BNBuilder",
    "BDEUBNBuilder",
    "EdgeBasedBNBuilder",
    "HybridBNBuilder",
    "LocalNetworkManager",
    "filter_weak_partitions",
    "create_bn_builder",
]
