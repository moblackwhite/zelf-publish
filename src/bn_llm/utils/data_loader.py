"""BN-LLM 数据加载模块

提供数据集加载和预处理功能。
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from ..config import DatasetConfig, DatasetPaths
from .logger import get_logger


class DataLoader:
    """数据加载器
    
    负责加载和预处理数据集，支持多种数据集格式和变体。
    
    数据集目录结构:
        data/datasets/
        ├── hospital/
        │   ├── clean.csv
        │   ├── dirty.csv
        │   ├── discrete_cols.json
        │   └── variants/
        │       ├── error_10/
        │       │   ├── clean.csv
        │       │   └── dirty.csv
        │       └── error_20/
        │           └── ...
        └── flights/
            └── ...
    """
    
    def __init__(
        self, 
        data_root: Path = Path("data/datasets"),
        null_marker: str = "[NULL Cell]"
    ):
        """初始化数据加载器
        
        Args:
            data_root: 数据集根目录
            null_marker: 空值标记
        """
        self.data_root = Path(data_root)
        self.null_marker = null_marker
        self.logger = get_logger()
    
    def get_dataset_paths(self, config: DatasetConfig) -> DatasetPaths:
        """获取数据集路径
        
        Args:
            config: 数据集配置
        
        Returns:
            数据集路径对象
        """
        return config.get_paths(self.data_root)
    
    def load_dirty_data(
        self, 
        config: DatasetConfig,
        discrete_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """加载脏数据
        
        Args:
            config: 数据集配置
            discrete_cols: 离散列列表，如果为 None 则自动加载
        
        Returns:
            预处理后的脏数据 DataFrame
        """
        paths = self.get_dataset_paths(config)
        
        if discrete_cols is None:
            discrete_cols = self.load_discrete_cols(config)
        
        self.logger.info(f"加载脏数据: {paths.dirty_csv}")
        return self._load_and_preprocess(paths.dirty_csv, discrete_cols)
    
    def load_clean_data(
        self, 
        config: DatasetConfig,
        discrete_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """加载干净数据（Ground Truth）
        
        Args:
            config: 数据集配置
            discrete_cols: 离散列列表，如果为 None 则自动加载
        
        Returns:
            预处理后的干净数据 DataFrame
        """
        paths = self.get_dataset_paths(config)
        
        if discrete_cols is None:
            discrete_cols = self.load_discrete_cols(config)
        
        self.logger.info(f"加载干净数据: {paths.clean_csv}")
        return self._load_and_preprocess(paths.clean_csv, discrete_cols)
    
    def load_discrete_cols(self, config: DatasetConfig) -> List[str]:
        """加载离散列配置
        
        Args:
            config: 数据集配置
        
        Returns:
            离散列名称列表
        """
        paths = self.get_dataset_paths(config)
        
        if not paths.discrete_cols.exists():
            raise FileNotFoundError(
                f"离散列配置文件不存在: {paths.discrete_cols}"
            )
        
        with open(paths.discrete_cols, "r", encoding="utf-8") as f:
            discrete_cols = json.load(f)
        
        self.logger.debug(f"离散列: {discrete_cols}")
        return discrete_cols
    
    def load_both(
        self,
        config: DatasetConfig,
        discrete_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """同时加载脏数据和干净数据
        
        Args:
            config: 数据集配置
            discrete_cols: 离散列列表
        
        Returns:
            (脏数据, 干净数据) 元组
        """
        if discrete_cols is None:
            discrete_cols = self.load_discrete_cols(config)
        
        dirty_df = self.load_dirty_data(config, discrete_cols)
        clean_df = self.load_clean_data(config, discrete_cols)
        
        return dirty_df, clean_df
    
    def _load_and_preprocess(
        self, 
        path: Path, 
        discrete_cols: List[str]
    ) -> pd.DataFrame:
        """加载并预处理数据
        
        Args:
            path: 数据文件路径
            discrete_cols: 离散列列表
        
        Returns:
            预处理后的 DataFrame
        """
        # 读取 CSV，所有列作为字符串
        df = pd.read_csv(path, dtype=str)
        
        # 只保留指定的离散列
        df = df[discrete_cols]
        
        # 用空值标记填充缺失值
        df = df.fillna(self.null_marker)
        
        # 确保所有值都是字符串
        for col in df.columns:
            df[col] = df[col].astype(str)
        
        return df
    
    def list_datasets(self) -> List[str]:
        """列出所有可用数据集
        
        Returns:
            数据集名称列表
        """
        if not self.data_root.exists():
            return []
        
        datasets = []
        for item in self.data_root.iterdir():
            if item.is_dir() and (item / "clean.csv").exists():
                datasets.append(item.name)
        
        return sorted(datasets)
    
    def list_variants(self, dataset_name: str) -> List[str]:
        """列出数据集的所有变体
        
        Args:
            dataset_name: 数据集名称
        
        Returns:
            变体名称列表
        """
        variants_dir = self.data_root / dataset_name / "variants"
        
        if not variants_dir.exists():
            return []
        
        variants = []
        for item in variants_dir.iterdir():
            if item.is_dir() and (item / "dirty.csv").exists():
                variants.append(item.name)
        
        return sorted(variants)
    
    def verify_dataset(self, config: DatasetConfig) -> dict:
        """验证数据集完整性
        
        Args:
            config: 数据集配置
        
        Returns:
            验证结果字典
        """
        paths = self.get_dataset_paths(config)
        
        result = {
            "dataset": config.full_name,
            "clean_csv": paths.clean_csv.exists(),
            "dirty_csv": paths.dirty_csv.exists(),
            "discrete_cols": paths.discrete_cols.exists(),
            "valid": True,
            "errors": [],
        }
        
        if not paths.clean_csv.exists():
            result["errors"].append(f"干净数据文件不存在: {paths.clean_csv}")
            result["valid"] = False
        
        if not paths.dirty_csv.exists():
            result["errors"].append(f"脏数据文件不存在: {paths.dirty_csv}")
            result["valid"] = False
        
        if not paths.discrete_cols.exists():
            result["errors"].append(f"离散列配置不存在: {paths.discrete_cols}")
            result["valid"] = False
        
        # 验证数据一致性
        if result["valid"]:
            try:
                discrete_cols = self.load_discrete_cols(config)
                dirty_df = pd.read_csv(paths.dirty_csv, dtype=str, nrows=5)
                clean_df = pd.read_csv(paths.clean_csv, dtype=str, nrows=5)
                
                # 检查列是否存在
                for col in discrete_cols:
                    if col not in dirty_df.columns:
                        result["errors"].append(f"脏数据缺少列: {col}")
                        result["valid"] = False
                    if col not in clean_df.columns:
                        result["errors"].append(f"干净数据缺少列: {col}")
                        result["valid"] = False
                
                result["row_count_dirty"] = len(pd.read_csv(paths.dirty_csv))
                result["row_count_clean"] = len(pd.read_csv(paths.clean_csv))
                result["column_count"] = len(discrete_cols)
                
            except Exception as e:
                result["errors"].append(f"数据验证出错: {str(e)}")
                result["valid"] = False
        
        return result


__all__ = ["DataLoader"]

