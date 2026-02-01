"""实验结果管理模块

提供结果存储、查询、导出等功能。

结果目录结构：
outputs/results/
├── 20260107_143052/               # 时间戳（一次实验）
│   ├── config.yaml                # 本次实验配置快照
│   ├── hospital.json              # 单数据集结果
│   ├── flights.json
│   └── summary.json               # 汇总结果
└── latest -> 20260107_143052/     # 符号链接到最新
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .logger import get_logger

class ResultsManager:
    """结果管理器
    
    负责实验结果的存储、查询和导出。
    
    Example:
        >>> manager = ResultsManager(Path("outputs/results"))
        >>> exp_dir = manager.create_experiment_dir()
        >>> manager.save_result(exp_dir, "hospital", {"f1": 0.82})
        >>> manager.summarize(exp_dir)
    """
    
    def __init__(self, results_dir: Path):
        """初始化结果管理器
        
        Args:
            results_dir: 结果目录根路径
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()
    
    def create_experiment_dir(
        self,
        name: Optional[str] = None,
    ) -> Path:
        """创建新实验目录（时间戳命名）
        
        Args:
            name: 可选的实验名称后缀
        
        Returns:
            实验目录路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if name:
            dir_name = f"{timestamp}_{name}"
        else:
            dir_name = timestamp
        
        exp_dir = self.results_dir / dir_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 更新 latest 符号链接
        latest_link = self.results_dir / "latest"
        if latest_link.is_symlink():
            latest_link.unlink()
        elif latest_link.exists():
            # 如果是普通目录，不处理
            pass
        
        try:
            latest_link.symlink_to(dir_name)
        except OSError:
            # Windows 可能不支持符号链接
            pass
        
        self.logger.info(f"创建实验目录: {exp_dir}")
        return exp_dir
    
    def save_result(
        self,
        exp_dir: Path,
        dataset: str,
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """保存单数据集结果
        
        Args:
            exp_dir: 实验目录
            dataset: 数据集名称
            metrics: 评估指标
            metadata: 元数据
        
        Returns:
            结果文件路径
        """
        result = {
            "metadata": {
                "dataset": dataset,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {}),
            },
            "metrics": metrics,
        }
        
        output_path = Path(exp_dir) / f"{dataset}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"保存结果: {output_path}")
        return output_path
    
    def save_config_snapshot(
        self,
        exp_dir: Path,
        config: Dict[str, Any],
    ) -> Path:
        """保存配置快照
        
        Args:
            exp_dir: 实验目录
            config: 配置字典
        
        Returns:
            配置文件路径
        """
        import yaml
        
        output_path = Path(exp_dir) / "config.yaml"
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return output_path
    
    def load_result(
        self,
        exp_dir: Path,
        dataset: str,
    ) -> Optional[Dict]:
        """加载单数据集结果
        
        Args:
            exp_dir: 实验目录
            dataset: 数据集名称
        
        Returns:
            结果字典或 None
        """
        result_path = Path(exp_dir) / f"{dataset}.json"
        if not result_path.exists():
            return None
        
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def summarize(self, exp_dir: Path) -> Dict:
        """汇总实验结果
        
        Args:
            exp_dir: 实验目录
        
        Returns:
            汇总字典
        """
        exp_dir = Path(exp_dir)
        
        # 收集所有数据集结果
        results = []
        for result_file in exp_dir.glob("*.json"):
            if result_file.name == "summary.json":
                continue
            
            with open(result_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            dataset = data.get("metadata", {}).get("dataset", result_file.stem)
            metrics = data.get("metrics", {})
            
            results.append({
                "dataset": dataset,
                **metrics,
            })
        
        if not results:
            return {
                "experiment_id": exp_dir.name,
                "datasets": [],
                "aggregate": {},
                "results": [],
            }
        
        # 计算聚合指标
        df = pd.DataFrame(results)
        aggregate = {}
        
        for metric in ["precision", "recall", "f1", "detection_rate", "repair_accuracy"]:
            if metric in df.columns:
                aggregate[f"mean_{metric}"] = df[metric].mean()
                aggregate[f"std_{metric}"] = df[metric].std()
        
        summary = {
            "experiment_id": exp_dir.name,
            "datasets": [r["dataset"] for r in results],
            "aggregate": aggregate,
            "results": results,
        }
        
        # 保存汇总文件
        summary_path = exp_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary
    
    def export(
        self,
        exp_dir: Path,
        format: str = "csv",
        output_path: Optional[Path] = None,
    ) -> Path:
        """导出结果
        
        Args:
            exp_dir: 实验目录
            format: 导出格式 (csv/json)
            output_path: 输出文件路径
        
        Returns:
            输出文件路径
        """
        summary = self.summarize(exp_dir)
        results = summary.get("results", [])
        
        if not results:
            raise ValueError("没有可导出的结果")
        
        exp_dir = Path(exp_dir)
        
        if output_path is None:
            if format == "csv":
                output_path = exp_dir / "results.csv"
            else:
                output_path = exp_dir / "results.json"
        
        output_path = Path(output_path)
        
        if format == "csv":
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"导出结果到: {output_path}")
        return output_path
    
    def get_latest_experiment(self) -> Optional[Path]:
        """获取最新实验目录
        
        Returns:
            最新实验目录或 None
        """
        # 先尝试 latest 链接
        latest_link = self.results_dir / "latest"
        if latest_link.is_symlink() or latest_link.exists():
            target = latest_link.resolve()
            if target.is_dir():
                return target
        
        # 否则找最新的时间戳目录
        exp_dirs = [
            d for d in self.results_dir.iterdir()
            if d.is_dir() and d.name != "latest"
        ]
        
        if not exp_dirs:
            return None
        
        return max(exp_dirs, key=lambda d: d.name)
    
    def list_experiments(
        self,
        limit: int = 10,
    ) -> List[Dict]:
        """列出实验
        
        Args:
            limit: 返回数量限制
        
        Returns:
            实验信息列表
        """
        exp_dirs = [
            d for d in self.results_dir.iterdir()
            if d.is_dir() and d.name != "latest"
        ]
        
        # 按名称倒序排序（时间戳命名）
        exp_dirs.sort(key=lambda d: d.name, reverse=True)
        exp_dirs = exp_dirs[:limit]
        
        experiments = []
        for exp_dir in exp_dirs:
            # 读取汇总
            summary_path = exp_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                datasets = summary.get("datasets", [])
                aggregate = summary.get("aggregate", {})
            else:
                # 计算数据集数量
                datasets = [
                    f.stem for f in exp_dir.glob("*.json")
                    if f.name != "summary.json"
                ]
                aggregate = {}
            
            experiments.append({
                "id": exp_dir.name,
                "path": str(exp_dir),
                "datasets": datasets,
                "mean_f1": aggregate.get("mean_f1"),
            })
        
        return experiments
    
    def delete_experiment(
        self,
        exp_id: str,
        force: bool = False,
    ) -> bool:
        """删除实验
        
        Args:
            exp_id: 实验 ID
            force: 强制删除，不检查
        
        Returns:
            是否删除成功
        """
        exp_dir = self.results_dir / exp_id
        if not exp_dir.exists():
            self.logger.warning(f"实验不存在: {exp_id}")
            return False
        
        if not force:
            # 检查是否为 latest
            latest_link = self.results_dir / "latest"
            if latest_link.is_symlink():
                if latest_link.resolve() == exp_dir.resolve():
                    self.logger.warning("不能删除最新实验，请使用 force=True")
                    return False
        
        shutil.rmtree(exp_dir)
        self.logger.info(f"已删除实验: {exp_id}")
        return True


def format_metrics(metrics: Dict[str, float]) -> str:
    """格式化指标为字符串
    
    Args:
        metrics: 指标字典
    
    Returns:
        格式化字符串
    """
    parts = []
    for key in ["precision", "recall", "f1"]:
        if key in metrics:
            parts.append(f"{key.capitalize()}: {metrics[key]:.4f}")
    return ", ".join(parts)


def create_results_manager(
    results_dir: Optional[Path] = None,
) -> ResultsManager:
    """创建结果管理器
    
    Args:
        results_dir: 结果目录，默认为 outputs/results
    
    Returns:
        结果管理器实例
    """
    if results_dir is None:
        results_dir = Path("outputs/results")
    return ResultsManager(results_dir)


__all__ = [
    "ResultsManager",
    "format_metrics",
    "create_results_manager",
]

