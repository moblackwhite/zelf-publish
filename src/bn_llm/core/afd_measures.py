"""近似函数依赖 (AFD) 度量模块

基于论文: "Approximately Measuring Functional Dependencies: a Comparative Study"
参考实现: https://github.com/UHasselt-DSI-Data-Systems-Lab/paper-afd-comparative-study

提供多种 AFD 度量方法，用于评估属性间的函数依赖强度。
"""

from typing import List, Optional, Union

import pandas as pd

from ..utils.logger import get_logger


def pdep_self(df: pd.DataFrame, y: str) -> float:
    """计算 pdep(Y)，由 Piatetsky-Shapiro & Matheus, 1993 定义

    Args:
        df: 数据框
        y: 单个属性名

    Returns:
        属性的 pdep 值
    """
    return (df[y].value_counts() / df.shape[0]).pow(2).sum()


def pdep(df: pd.DataFrame, lhs: Union[str, List[str]], rhs: str) -> float:
    """计算 pdep(X,Y)，由 Piatetsky-Shapiro & Matheus, 1993 定义

    支持左侧多属性。

    Args:
        df: 数据框
        lhs: 左侧属性名或属性名列表
        rhs: 右侧单个属性名

    Returns:
        lhs 和 rhs 之间的 pdep 值
    """
    # 确保 lhs 是列表
    lhs_cols = lhs if isinstance(lhs, list) else [lhs]

    # 计算 X,Y 联合计数
    all_cols = lhs_cols + [rhs]
    xy_groups = df.groupby(all_cols).size().reset_index(name="xy_count")

    # 计算 X 计数
    x_groups = df.groupby(lhs_cols).size().reset_index(name="x_count")

    # 合并计数
    counts = pd.merge(xy_groups, x_groups, on=lhs_cols)

    # 计算并返回 pdep 值
    return (1 / df.shape[0]) * (counts["xy_count"].pow(2) / counts["x_count"]).sum()


def mu(df: pd.DataFrame, lhs: Union[str, List[str]], rhs: str) -> float:
    """计算 mu 度量，由 Piatetsky-Shapiro & Matheus, 1993 定义

    支持左侧多属性。

    Args:
        df: 数据框
        lhs: 左侧属性名或属性名列表
        rhs: 右侧单个属性名

    Returns:
        衡量函数依赖强度的 mu 值
    """
    pdepXY = pdep(df, lhs, rhs)
    pdepY = pdep_self(df, rhs)
    r_size = df.shape[0]

    # 计算 X 的域大小（唯一值组合数）
    lhs_cols = lhs if isinstance(lhs, list) else [lhs]
    domX_size = df.groupby(lhs_cols).ngroups

    # 避免除零错误
    if r_size == domX_size or pdepY == 1.0:
        return 0.0

    return 1.0 - ((1 - pdepXY) / (1 - pdepY)) * ((r_size - 1) / (r_size - domX_size))


def mu_plus(df: pd.DataFrame, lhs: Union[str, List[str]], rhs: str) -> float:
    """修改后的 mu 度量，确保结果在 [0, 1] 范围内

    支持左侧多属性。

    Args:
        df: 数据框
        lhs: 左侧属性名或属性名列表
        rhs: 右侧单个属性名

    Returns:
        范围在 [0, 1] 内的非负 mu 值
    """
    return max(0.0, mu(df, lhs, rhs))


def compute_mu_plus_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """计算所有属性对的 mu_plus 矩阵

    Args:
        df: 数据框
        columns: 要计算的列名列表，默认为 df 的所有列
        show_progress: 是否显示进度

    Returns:
        mu_plus 分数矩阵，行为 LHS，列为 RHS
    """
    logger = get_logger()

    if columns is None:
        columns = list(df.columns)

    n_cols = len(columns)
    total_pairs = n_cols * n_cols

    if show_progress:
        logger.info(f"计算 mu_plus 矩阵: {n_cols} 列, {total_pairs} 对")

    # 初始化结果矩阵
    result = pd.DataFrame(index=columns, columns=columns, dtype=float)

    # 计算所有属性对
    computed = 0
    for lhs in columns:
        for rhs in columns:
            if lhs == rhs:
                # 对角线设为 1.0（属性对自身完全函数依赖）
                result.loc[lhs, rhs] = 1.0
            else:
                result.loc[lhs, rhs] = mu_plus(df, lhs, rhs)

            computed += 1

        if show_progress and computed % (n_cols * 2) == 0:
            progress = computed / total_pairs * 100
            logger.info(f"mu_plus 矩阵计算进度: {progress:.1f}%")

    if show_progress:
        logger.info("mu_plus 矩阵计算完成")

    return result


__all__ = [
    "pdep_self",
    "pdep",
    "mu",
    "mu_plus",
    "compute_mu_plus_matrix",
]
