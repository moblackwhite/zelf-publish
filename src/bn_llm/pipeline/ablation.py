"""消融实验模块

提供消融实验功能，用于评估各组件对整体性能的贡献。

消融参数：
- use_compensation_score: 是否使用补偿分数
- use_llm_construct: 是否使用 LLM 构建 BN 结构
- use_hierarchical: 是否使用层次化结构（一致性修复）
- use_llm_inference: 是否使用 LLM 辅助推理
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config import Config, BNConfig
from ..utils.logger import get_logger
from .cleaning import CleaningPipeline, CleaningPipelineResult


@dataclass
class AblationResult:
    """消融实验结果"""
    
    param_name: str
    """消融的参数名"""
    
    enabled_metrics: Dict[str, float]
    """参数启用时的指标"""
    
    disabled_metrics: Dict[str, float]
    """参数禁用时的指标"""
    
    impact: Dict[str, float] = field(default_factory=dict)
    """影响（启用 - 禁用）"""
    
    def __post_init__(self):
        """计算影响"""
        for key in self.enabled_metrics:
            if key in self.disabled_metrics:
                self.impact[key] = (
                    self.enabled_metrics[key] - self.disabled_metrics[key]
                )


@dataclass
class AblationExperimentResult:
    """消融实验整体结果"""
    
    dataset_name: str
    results: List[AblationResult]
    baseline_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "dataset_name": self.dataset_name,
            "baseline": self.baseline_metrics,
            "ablations": [
                {
                    "param": r.param_name,
                    "enabled": r.enabled_metrics,
                    "disabled": r.disabled_metrics,
                    "impact": r.impact,
                }
                for r in self.results
            ],
        }


class AblationExperiment:
    """消融实验
    
    通过逐个禁用组件来评估各组件对整体性能的贡献。
    
    Example:
        >>> config = create_config("hospital")
        >>> experiment = AblationExperiment(config)
        >>> results = experiment.run()
        >>> for r in results.results:
        ...     print(f"{r.param_name}: F1 影响 = {r.impact['f1']:.4f}")
    """
    
    ABLATION_PARAMS = [
        "use_compensation_score",
        "use_llm_construct",
        "use_hierarchical",
        "use_llm_inference",
    ]
    
    def __init__(self, config: Config):
        """初始化消融实验
        
        Args:
            config: 基础配置（所有参数启用）
        """
        self.config = config
        self.logger = get_logger()
    
    def run(
        self,
        params: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> AblationExperimentResult:
        """运行消融实验
        
        Args:
            params: 要消融的参数列表，默认为所有参数
            use_cache: 是否使用缓存
        
        Returns:
            消融实验结果
        """
        if params is None:
            params = self.ABLATION_PARAMS
        
        self.logger.info(
            f"开始消融实验: {self.config.dataset.full_name}, "
            f"参数: {params}"
        )
        
        # 运行基线（所有参数启用）
        self.logger.info("运行基线实验...")
        baseline_result = self._run_with_config(self.config, use_cache)
        baseline_metrics = self._extract_metrics(baseline_result)
        
        # 运行各参数的消融
        ablation_results = []
        
        for param in params:
            if param not in self.ABLATION_PARAMS:
                self.logger.warning(f"未知的消融参数: {param}")
                continue
            
            self.logger.info(f"消融参数: {param}")
            
            # 创建禁用该参数的配置
            disabled_config = self._create_ablated_config(param)
            disabled_result = self._run_with_config(disabled_config, use_cache)
            disabled_metrics = self._extract_metrics(disabled_result)
            
            ablation_results.append(AblationResult(
                param_name=param,
                enabled_metrics=baseline_metrics.copy(),
                disabled_metrics=disabled_metrics,
            ))
        
        return AblationExperimentResult(
            dataset_name=self.config.dataset.full_name,
            results=ablation_results,
            baseline_metrics=baseline_metrics,
        )
    
    def _create_ablated_config(self, param: str) -> Config:
        """创建消融配置
        
        Args:
            param: 要禁用的参数名
        
        Returns:
            新配置对象
        """
        # 复制 BN 配置
        bn_dict = self.config.bn.model_dump()
        bn_dict[param] = False
        
        # 特殊处理：如果禁用 use_llm_construct，切换到 BDEU_ONLY
        if param == "use_llm_construct":
            bn_dict["build_method"] = "BDEU_ONLY"
        
        # 特殊处理：如果禁用 use_hierarchical，使用 NO_CONSISTENCY 方法
        if param == "use_hierarchical":
            if self.config.bn.build_method == "HYBRID":
                bn_dict["build_method"] = "HYBRID_NO_CONSISTENCY"
        
        new_bn = BNConfig(**bn_dict)
        
        # 创建新配置
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


def run_ablation_experiment(
    dataset_name: str,
    variant: Optional[str] = None,
    params: Optional[List[str]] = None,
    model_name: str = "gpt-4o-mini-2024-07-18",
    use_cache: bool = True,
) -> AblationExperimentResult:
    """便捷函数：运行消融实验
    
    Args:
        dataset_name: 数据集名称
        variant: 数据集变体
        params: 要消融的参数列表
        model_name: LLM 模型名称
        use_cache: 是否使用缓存
    
    Returns:
        消融实验结果
    """
    from ..config import create_config
    
    config = create_config(
        dataset_name=dataset_name,
        variant=variant,
        model_name=model_name,
    )
    
    experiment = AblationExperiment(config)
    return experiment.run(params=params, use_cache=use_cache)


__all__ = [
    "AblationResult",
    "AblationExperimentResult",
    "AblationExperiment",
    "run_ablation_experiment",
]

