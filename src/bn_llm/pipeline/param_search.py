"""参数搜索实验模块

提供参数网格搜索功能，用于寻找最优的 alpha、beta 参数组合。

参数说明：
- alpha: LLM FD 分数的权重
- beta: AFD 分数的权重
"""

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from ..config import Config, BNConfig, create_config
from ..utils.logger import get_logger
from .cleaning import CleaningPipeline, CleaningPipelineResult


@dataclass
class ParamSearchResult:
    """参数搜索结果"""
    
    best_params: Dict[str, float]
    """最优参数组合"""
    
    best_f1: float
    """最优 F1 分数"""
    
    all_results: List[Dict]
    """所有参数组合的结果"""
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame"""
        return pd.DataFrame(self.all_results)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "best_params": self.best_params,
            "best_f1": self.best_f1,
            "results_count": len(self.all_results),
            "all_results": self.all_results,
        }


@dataclass
class SingleParamResult:
    """单参数搜索结果"""
    
    param_name: str
    """参数名"""
    
    values: List[float]
    """搜索的参数值列表"""
    
    metrics: List[Dict[str, float]]
    """各参数值对应的指标"""
    
    best_value: float = 0.0
    """最优参数值"""
    
    best_f1: float = 0.0
    """最优 F1 分数"""
    
    def __post_init__(self):
        """找到最优值"""
        if self.metrics:
            best_idx = max(
                range(len(self.metrics)),
                key=lambda i: self.metrics[i].get("f1", 0.0)
            )
            self.best_value = self.values[best_idx]
            self.best_f1 = self.metrics[best_idx].get("f1", 0.0)
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame"""
        data = []
        for v, m in zip(self.values, self.metrics):
            row = {self.param_name: v, **m}
            data.append(row)
        return pd.DataFrame(data)


class ParamSearchExperiment:
    """参数搜索实验
    
    支持单参数搜索和多参数网格搜索。
    
    Example:
        >>> config = create_config("hospital")
        >>> experiment = ParamSearchExperiment(config)
        >>> result = experiment.search_alpha([0.05, 0.1, 0.15, 0.2])
        >>> print(f"最优 alpha: {result.best_value}")
    """
    
    DEFAULT_ALPHA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25]
    DEFAULT_BETA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25]
    
    def __init__(self, config: Config):
        """初始化参数搜索实验
        
        Args:
            config: 基础配置
        """
        self.config = config
        self.logger = get_logger()
    
    def grid_search(
        self,
        alpha_values: Optional[List[float]] = None,
        beta_values: Optional[List[float]] = None,
        use_cache: bool = True,
    ) -> ParamSearchResult:
        """网格搜索最优参数组合
        
        Args:
            alpha_values: alpha 参数值列表
            beta_values: beta 参数值列表
            use_cache: 是否使用缓存
        
        Returns:
            参数搜索结果
        """
        # 使用默认值
        if alpha_values is None:
            alpha_values = [self.config.bn.alpha or 0.1]
        if beta_values is None:
            beta_values = [self.config.bn.beta or 0.1]
        
        total = len(alpha_values) * len(beta_values)
        self.logger.info(
            f"开始参数网格搜索: {self.config.dataset.full_name}, "
            f"共 {total} 种组合"
        )
        
        all_results = []
        best_params = {}
        best_f1 = -1.0
        
        for i, (alpha, beta) in enumerate(
            itertools.product(alpha_values, beta_values)
        ):
            self.logger.info(
                f"[{i+1}/{total}] alpha={alpha}, beta={beta}"
            )
            
            # 创建配置
            config = self._create_config(alpha, beta)
            
            # 运行实验
            try:
                result = self._run_with_config(config, use_cache)
                metrics = self._extract_metrics(result)
            except Exception as e:
                self.logger.error(f"实验失败: {e}")
                metrics = {"precision": 0, "recall": 0, "f1": 0}
            
            # 记录结果
            result_dict = {
                "alpha": alpha,
                "beta": beta,
                **metrics,
            }
            all_results.append(result_dict)
            
            # 更新最优
            f1 = metrics.get("f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_params = {"alpha": alpha, "beta": beta}
        
        self.logger.info(
            f"参数搜索完成，最优: alpha={best_params.get('alpha')}, "
            f"beta={best_params.get('beta')}, F1={best_f1:.4f}"
        )
        
        return ParamSearchResult(
            best_params=best_params,
            best_f1=best_f1,
            all_results=all_results,
        )
    
    def search_alpha(
        self,
        values: Optional[List[float]] = None,
        use_cache: bool = True,
    ) -> SingleParamResult:
        """搜索最优 alpha 参数
        
        Args:
            values: alpha 参数值列表
            use_cache: 是否使用缓存
        
        Returns:
            单参数搜索结果
        """
        if values is None:
            values = self.DEFAULT_ALPHA_VALUES
        
        return self._search_single_param("alpha", values, use_cache)
    
    def search_beta(
        self,
        values: Optional[List[float]] = None,
        use_cache: bool = True,
    ) -> SingleParamResult:
        """搜索最优 beta 参数
        
        Args:
            values: beta 参数值列表
            use_cache: 是否使用缓存
        
        Returns:
            单参数搜索结果
        """
        if values is None:
            values = self.DEFAULT_BETA_VALUES
        
        return self._search_single_param("beta", values, use_cache)
    
    def _search_single_param(
        self,
        param_name: str,
        values: List[float],
        use_cache: bool,
    ) -> SingleParamResult:
        """搜索单个参数
        
        Args:
            param_name: 参数名
            values: 参数值列表
            use_cache: 是否使用缓存
        
        Returns:
            单参数搜索结果
        """
        self.logger.info(
            f"开始 {param_name} 参数搜索: {self.config.dataset.full_name}, "
            f"值: {values}"
        )
        
        metrics_list = []
        
        for i, value in enumerate(values):
            self.logger.info(f"[{i+1}/{len(values)}] {param_name}={value}")
            
            # 创建配置
            if param_name == "alpha":
                config = self._create_config(alpha=value)
            else:  # beta
                config = self._create_config(beta=value)
            
            # 运行实验
            try:
                result = self._run_with_config(config, use_cache)
                metrics = self._extract_metrics(result)
            except Exception as e:
                self.logger.error(f"实验失败: {e}")
                metrics = {"precision": 0, "recall": 0, "f1": 0}
            
            metrics_list.append(metrics)
        
        return SingleParamResult(
            param_name=param_name,
            values=values,
            metrics=metrics_list,
        )
    
    def _create_config(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> Config:
        """创建配置
        
        Args:
            alpha: alpha 参数
            beta: beta 参数
        
        Returns:
            新配置对象
        """
        bn_dict = self.config.bn.model_dump()
        
        if alpha is not None:
            bn_dict["alpha"] = alpha
        if beta is not None:
            bn_dict["beta"] = beta
        
        new_bn = BNConfig(**bn_dict)
        
        return Config(
            dataset=self.config.dataset,
            model=self.config.model,
            bn=new_bn,
            experiment=self.config.experiment,
            output=self.config.output,
            data_root=self.config.data_root,
        )
    
    def _run_with_config(
        self,
        config: Config,
        use_cache: bool,
    ) -> CleaningPipelineResult:
        """使用指定配置运行清洗流程
        
        Args:
            config: 配置对象
            use_cache: 是否使用缓存
        
        Returns:
            清洗结果
        """
        pipeline = CleaningPipeline(config)
        return pipeline.run(use_cache=use_cache)
    
    def _extract_metrics(self, result: CleaningPipelineResult) -> Dict[str, float]:
        """从结果中提取指标
        
        Args:
            result: 清洗结果
        
        Returns:
            指标字典
        """
        if result.evaluation is None:
            return {}
        
        return {
            "precision": result.evaluation.metrics.precision,
            "recall": result.evaluation.metrics.recall,
            "f1": result.evaluation.metrics.f1,
            "detection_rate": result.evaluation.metrics.detection_rate,
            "repair_accuracy": result.evaluation.metrics.repair_accuracy,
        }


def run_param_search(
    dataset_name: str,
    variant: Optional[str] = None,
    alpha_values: Optional[List[float]] = None,
    beta_values: Optional[List[float]] = None,
    model_name: str = "gpt-4o-mini-2024-07-18",
    use_cache: bool = True,
) -> ParamSearchResult:
    """便捷函数：运行参数网格搜索
    
    Args:
        dataset_name: 数据集名称
        variant: 数据集变体
        alpha_values: alpha 参数值列表
        beta_values: beta 参数值列表
        model_name: LLM 模型名称
        use_cache: 是否使用缓存
    
    Returns:
        参数搜索结果
    """
    config = create_config(
        dataset_name=dataset_name,
        variant=variant,
        model_name=model_name,
    )
    
    experiment = ParamSearchExperiment(config)
    return experiment.grid_search(
        alpha_values=alpha_values,
        beta_values=beta_values,
        use_cache=use_cache,
    )


__all__ = [
    "ParamSearchResult",
    "SingleParamResult",
    "ParamSearchExperiment",
    "run_param_search",
]

