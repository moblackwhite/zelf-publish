"""BN-LLM 缓存管理模块

提供缓存存储、加载和清理功能，支持多种数据格式。
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import pandas as pd

from .logger import get_logger

T = TypeVar("T")


class CacheCategory:
    """缓存类别常量"""

    BN_STRUCTURES = "bn_structures"
    FD_RESULTS = "fd_results"
    INFERRED_DF = "inferred_df"
    LLM_RESPONSES = "llm_responses"
    OCCURRENCE_MATRIX = "occurrence_matrix"
    LOCAL_NETWORKS = "local_networks"
    REPAIR_RESULT = "repair_result"  # BN 推理修复结果


class CacheManager:
    """缓存管理器

    负责管理项目中的各类缓存数据，支持自动序列化/反序列化。

    缓存目录结构:
        cache_dir/
        ├── bn_structures/
        │   └── hospital_a1b2c3d4.json
        ├── fd_results/
        │   └── hospital_DFD_a1b2c3d4.csv
        ├── inferred_df/
        │   └── hospital_a1b2c3d4.csv
        └── llm_responses/
            └── hospital_cell_cleaning_a1b2c3d4.json
    """

    def __init__(self, cache_dir: Path):
        """初始化缓存管理器

        Args:
            cache_dir: 缓存根目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()

    @staticmethod
    def compute_hash(config_dict: dict) -> str:
        """计算配置哈希

        Args:
            config_dict: 配置字典

        Returns:
            12位哈希字符串
        """
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]

    def get_cache_path(self, category: str, key: str, suffix: str = ".pkl") -> Path:
        """获取缓存文件路径

        Args:
            category: 缓存类别
            key: 缓存键名
            suffix: 文件后缀

        Returns:
            缓存文件完整路径
        """
        category_dir = self.cache_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        return category_dir / f"{key}{suffix}"

    def exists(self, category: str, key: str, suffix: str = ".pkl") -> bool:
        """检查缓存是否存在

        Args:
            category: 缓存类别
            key: 缓存键名
            suffix: 文件后缀

        Returns:
            缓存是否存在
        """
        cache_path = self.get_cache_path(category, key, suffix)
        return cache_path.exists()

    def load(
        self,
        category: str,
        key: str,
        suffix: str = ".pkl",
        loader: Optional[Callable[[], T]] = None,
    ) -> Optional[T]:
        """加载缓存

        如果缓存存在，加载并返回；如果不存在但提供了 loader，则执行 loader 并缓存结果。

        Args:
            category: 缓存类别
            key: 缓存键名
            suffix: 文件后缀
            loader: 可选的数据加载函数

        Returns:
            缓存数据或 None
        """
        cache_path = self.get_cache_path(category, key, suffix)

        if cache_path.exists():
            self.logger.debug(f"加载缓存: {cache_path}")
            return self._load_file(cache_path)

        if loader is not None:
            self.logger.debug(f"缓存未命中，执行 loader: {category}/{key}")
            data = loader()
            self.save(category, key, data, suffix)
            return data

        return None

    def save(
        self,
        category: str,
        key: str,
        data: Any,
        suffix: str = ".pkl",
    ) -> Path:
        """保存缓存

        Args:
            category: 缓存类别
            key: 缓存键名
            data: 要缓存的数据
            suffix: 文件后缀

        Returns:
            缓存文件路径
        """
        cache_path = self.get_cache_path(category, key, suffix)
        self.logger.debug(f"保存缓存: {cache_path}")
        self._save_file(cache_path, data)
        return cache_path

    def _load_file(self, path: Path) -> Any:
        """根据文件后缀加载数据"""
        suffix = path.suffix.lower()

        if suffix == ".pkl":
            with open(path, "rb") as f:
                return pickle.load(f)
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif suffix == ".csv":
            return pd.read_csv(path, index_col=0)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")

    def _save_file(self, path: Path, data: Any) -> None:
        """根据文件后缀保存数据"""
        suffix = path.suffix.lower()

        if suffix == ".pkl":
            with open(path, "wb") as f:
                pickle.dump(data, f)
        elif suffix == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        elif suffix == ".csv":
            if isinstance(data, pd.DataFrame):
                data.to_csv(path)
            else:
                raise TypeError("CSV 格式只支持 DataFrame")
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")

    def clear(self, category: Optional[str] = None) -> int:
        """清除缓存

        Args:
            category: 指定清除的类别，None 表示清除所有

        Returns:
            清除的文件数量
        """
        count = 0

        if category:
            category_dir = self.cache_dir / category
            if category_dir.exists():
                for file in category_dir.iterdir():
                    if file.is_file():
                        file.unlink()
                        count += 1
        else:
            for category_dir in self.cache_dir.iterdir():
                if category_dir.is_dir():
                    for file in category_dir.iterdir():
                        if file.is_file():
                            file.unlink()
                            count += 1

        self.logger.info(f"清除了 {count} 个缓存文件")
        return count

    def list_cache(self, category: Optional[str] = None) -> dict:
        """列出缓存文件

        Args:
            category: 指定列出的类别，None 表示列出所有

        Returns:
            {category: [file_names]} 格式的字典
        """
        result = {}

        if category:
            category_dir = self.cache_dir / category
            if category_dir.exists():
                result[category] = [f.name for f in category_dir.iterdir() if f.is_file()]
        else:
            for category_dir in self.cache_dir.iterdir():
                if category_dir.is_dir():
                    files = [f.name for f in category_dir.iterdir() if f.is_file()]
                    if files:
                        result[category_dir.name] = files

        return result

    def get_cache_size(self, category: Optional[str] = None) -> int:
        """获取缓存大小（字节）

        Args:
            category: 指定类别，None 表示所有缓存

        Returns:
            缓存大小（字节）
        """
        total_size = 0

        if category:
            category_dir = self.cache_dir / category
            if category_dir.exists():
                for file in category_dir.iterdir():
                    if file.is_file():
                        total_size += file.stat().st_size
        else:
            for category_dir in self.cache_dir.iterdir():
                if category_dir.is_dir():
                    for file in category_dir.iterdir():
                        if file.is_file():
                            total_size += file.stat().st_size

        return total_size


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


__all__ = ["CacheManager", "CacheCategory", "format_size"]
