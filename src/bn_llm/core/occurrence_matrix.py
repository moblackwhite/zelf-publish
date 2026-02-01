"""共现矩阵计算模块

计算数据集中属性值之间的共现关系，用于贝叶斯网络推理的补偿分数计算。
"""

import hashlib
import multiprocessing as mp
from functools import partial
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm


def compute_occurrence_matrix(
    data: pd.DataFrame, 
    n_processes: int = None,
    show_progress: bool = True,
) -> Dict[str, Dict[str, Dict[str, Dict[str, int]]]]:
    """计算属性值共现矩阵
    
    计算数据集中每对属性值的共现次数，用于后续的补偿分数计算。
    
    Args:
        data: 输入数据 DataFrame
        n_processes: 并行进程数，默认使用 CPU 核心数
        show_progress: 是否显示进度条
    
    Returns:
        嵌套字典结构:
        {
            attr1: {
                value1: {
                    attr2: {value2: count, ...},
                    ...
                },
                ...
            },
            ...
        }
    
    Example:
        >>> df = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['x', 'y', 'x']})
        >>> matrix = compute_occurrence_matrix(df)
        >>> matrix['A']['a']['B']['x']  # 'A=a' 与 'B=x' 共现次数
        1
    """
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    # 生成所有 (行索引, 属性) 对
    attribute_pairs = [
        (row_idx, attr)
        for row_idx in range(len(data))
        for attr in data.columns
    ]
    
    # 将工作分成若干块
    chunk_size = max(1, len(attribute_pairs) // n_processes)
    chunks = [
        attribute_pairs[i:i + chunk_size]
        for i in range(0, len(attribute_pairs), chunk_size)
    ]
    
    # 创建进程池并行处理
    with mp.Pool(processes=n_processes) as pool:
        process_chunk_fn = partial(_process_chunk, df=data)
        
        if show_progress:
            results = list(tqdm(
                pool.imap(process_chunk_fn, chunks),
                total=len(chunks),
                desc="计算共现矩阵"
            ))
        else:
            results = pool.map(process_chunk_fn, chunks)
    
    # 合并所有结果
    occurrences: Dict = {}
    for chunk_result in results:
        _merge_dictionaries(occurrences, chunk_result)
    
    return occurrences


def compute_occurrence_matrix_hash(data: pd.DataFrame) -> str:
    """计算数据的哈希值，用于缓存键
    
    Args:
        data: 输入数据 DataFrame
    
    Returns:
        12位哈希字符串
    """
    df_hash = hashlib.md5(
        pd.util.hash_pandas_object(data).values
    ).hexdigest()
    return df_hash[:12]


def _process_chunk(
    pairs: List[Tuple[int, str]], 
    df: pd.DataFrame
) -> Dict:
    """处理一组 (行索引, 属性) 对
    
    Args:
        pairs: (行索引, 属性名) 列表
        df: 数据 DataFrame
    
    Returns:
        局部共现字典
    """
    occurrences: Dict = {}
    
    for row_idx, main_attr in pairs:
        _process_pair(df, row_idx, main_attr, occurrences)
    
    return occurrences


def _process_pair(
    df: pd.DataFrame,
    row_idx: int,
    main_attr: str,
    occurrences: Dict
) -> None:
    """处理单个 (行, 属性) 对
    
    Args:
        df: 数据 DataFrame
        row_idx: 行索引
        main_attr: 主属性名
        occurrences: 共现字典（会被修改）
    """
    # 计算权重（使用列数的平方）
    weight = len(df.columns) ** 2
    
    # 初始化属性字典
    if main_attr not in occurrences:
        occurrences[main_attr] = {}
    
    # 获取主属性值
    main_value = df.loc[row_idx, main_attr]
    
    # 初始化值字典
    if main_value not in occurrences[main_attr]:
        occurrences[main_attr][main_value] = {}
    
    # 处理与其他属性的关系
    for other_attr in df.columns:
        if main_attr == other_attr:
            continue
        
        # 初始化其他属性字典
        if other_attr not in occurrences[main_attr][main_value]:
            occurrences[main_attr][main_value][other_attr] = {}
        
        # 获取其他属性值
        other_value = df.loc[row_idx, other_attr]
        
        # 初始化计数
        if other_value not in occurrences[main_attr][main_value][other_attr]:
            occurrences[main_attr][main_value][other_attr][other_value] = 0
        
        # 更新加权计数
        occurrences[main_attr][main_value][other_attr][other_value] += weight


def _merge_dictionaries(target: Dict, source: Dict) -> None:
    """合并嵌套字典结构
    
    递归地将 source 合并到 target 中。对于数值，执行累加操作。
    
    Args:
        target: 目标字典（会被修改）
        source: 源字典
    """
    for key, value in source.items():
        if key not in target:
            target[key] = value
        elif isinstance(value, dict):
            _merge_dictionaries(target[key], value)
        else:
            # 对于相同键的数值，累加
            target[key] += value


def get_co_occurrence_count(
    occurrences: Dict,
    attr1: str,
    value1: str,
    attr2: str,
    value2: str,
) -> int:
    """获取两个属性值的共现次数
    
    Args:
        occurrences: 共现矩阵
        attr1: 第一个属性名
        value1: 第一个属性值
        attr2: 第二个属性名
        value2: 第二个属性值
    
    Returns:
        共现次数，如果不存在则返回 0
    """
    try:
        return occurrences[attr1][value1][attr2][value2]
    except KeyError:
        return 0


def get_value_total_occurrences(
    occurrences: Dict,
    attr: str,
    value: str,
) -> int:
    """获取某个属性值的总出现次数
    
    Args:
        occurrences: 共现矩阵
        attr: 属性名
        value: 属性值
    
    Returns:
        总出现次数
    """
    if attr not in occurrences or value not in occurrences[attr]:
        return 0
    
    total = 0
    for other_attr in occurrences[attr][value]:
        for other_value, count in occurrences[attr][value][other_attr].items():
            total += count
    
    return total


__all__ = [
    "compute_occurrence_matrix",
    "compute_occurrence_matrix_hash",
    "get_co_occurrence_count",
    "get_value_total_occurrences",
]

