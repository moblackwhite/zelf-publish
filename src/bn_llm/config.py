"""BN-LLM 配置模块

使用 Pydantic 进行配置管理，支持 YAML 文件加载和环境变量覆盖。
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DatasetPaths(BaseModel):
    """数据集路径集合"""

    clean_csv: Path
    dirty_csv: Path
    discrete_cols: Path


class DatasetConfig(BaseModel):
    """数据集配置"""

    name: str = Field(..., description="数据集名称")
    variant: Optional[str] = Field(None, description="变体名称，如 'error_10'")

    @property
    def full_name(self) -> str:
        """返回完整数据集名称"""
        if self.variant:
            return f"{self.name}_{self.variant}"
        return self.name

    def get_paths(self, data_root: Path) -> DatasetPaths:
        """获取数据集相关路径"""
        if self.variant:
            # 变体数据集路径
            variant_dir = data_root / self.name / "variants" / self.variant
            base_dir = data_root / self.name
            return DatasetPaths(
                clean_csv=variant_dir / "clean.csv",
                dirty_csv=variant_dir / "dirty.csv",
                discrete_cols=base_dir / "discrete_cols.json",
            )
        else:
            # 基础数据集路径
            base_dir = data_root / self.name
            return DatasetPaths(
                clean_csv=base_dir / "clean.csv",
                dirty_csv=base_dir / "dirty.csv",
                discrete_cols=base_dir / "discrete_cols.json",
            )


class ModelConfig(BaseModel):
    """LLM 模型配置"""

    name: str = Field(default="gpt-4o-mini-2024-07-18")
    api_base: Optional[str] = Field(default=None, description="API 基础 URL")
    api_key: Optional[str] = Field(default=None, description="API 密钥")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    def model_post_init(self, __context) -> None:
        """加载环境变量"""
        if self.api_base is None:
            self.api_base = os.environ.get("LLM_API_BASE_URL")
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")

    @property
    def short_name(self) -> str:
        """返回适用于文件名的简短模型名称

        从完整名称中提取最后一段，例如：
        - 'openrouter/google/gemini-3-flash-preview' -> 'gemini-3-flash-preview'
        - 'gpt-4o-mini' -> 'gpt-4o-mini'
        """
        return self.name.rsplit("/", 1)[-1]


class BNConfig(BaseModel):
    """贝叶斯网络配置"""

    build_method: Literal[
        "HYBRID",  # LLM + BDEU 混合构建
        "HYBRID_NO_CONSISTENCY",  # 混合构建但不做一致性修复
        "BDEU_ONLY",  # 仅使用 BDEU 评分
        "LLM_ONLY",  # 仅使用 LLM 生成的边
        "USER_DEFINED",  # 用户自定义结构
    ] = Field(default="HYBRID")

    # 消融实验参数
    use_compensation_score: bool = Field(default=True, description="是否使用补偿分数")
    use_llm_construct: bool = Field(default=True, description="是否使用 LLM 构建")
    use_hierarchical: bool = Field(default=True, description="是否使用层次化结构")
    use_llm_inference: bool = Field(default=True, description="是否使用 LLM 推理")

    # 参数实验
    alpha: Optional[float] = Field(None, ge=0.0, le=1.0, description="Alpha 参数")
    beta: Optional[float] = Field(None, ge=0.0, le=1.0, description="Beta 参数")

    # BDEU 参数
    equivalent_sample_size: int = Field(default=50, description="等效样本大小")


class ExperimentConfig(BaseModel):
    """实验配置"""

    infer_limit: int = Field(default=-1, description="推理数量限制，-1 表示无限制")
    worker_count: int = Field(default=32, ge=1)
    batch_size: int = Field(default=250, ge=1)
    null_marker: str = Field(default="[NULL Cell]")
    random_state: int = Field(default=1234, description="随机种子")

    # LLM 修复预算配置
    llm_repair_budget_ratio: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="LLM 修复预算比例，基于总单元格数（如 0.01 表示 1%）",
    )
    llm_repair_max_budget: int = Field(
        default=-1, description="LLM 修复最大预算数量，-1 表示使用比例计算"
    )


class OutputConfig(BaseModel):
    """输出配置"""

    base_dir: Path = Field(default=Path("outputs"))
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    run_id: Optional[str] = Field(default=None, description="运行 ID（时间戳），用于区分不同运行")

    @property
    def cache_dir(self) -> Path:
        return self.base_dir / "cache"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def results_dir(self) -> Path:
        return self.base_dir / "results"

    @property
    def figures_dir(self) -> Path:
        return self.base_dir / "figures"

    def get_run_results_dir(self, dataset_name: str) -> Path:
        """获取当前运行的结果目录

        如果 run_id 已设置，返回 results_dir / dataset_name / run_id
        否则返回 results_dir / dataset_name
        """
        base = self.results_dir / dataset_name
        if self.run_id:
            return base / self.run_id
        return base


class Config(BaseSettings):
    """主配置类

    支持从 YAML 文件加载配置，并可通过环境变量覆盖。
    环境变量前缀：BN_LLM_
    """

    dataset: DatasetConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    bn: BNConfig = Field(default_factory=BNConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    # 数据根目录
    data_root: Path = Field(default=Path("data/datasets"))

    model_config = {
        "env_prefix": "BN_LLM_",
        "env_nested_delimiter": "__",
    }

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """从 YAML 文件加载配置"""
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # 处理环境变量引用
        data = cls._resolve_env_vars(data)

        return cls(**data)

    @classmethod
    def _resolve_env_vars(cls, data: dict) -> dict:
        """递归解析配置中的环境变量引用

        支持格式: ${env:VAR_NAME} 或 ${env:VAR_NAME:default}
        """
        import re

        env_pattern = re.compile(r"\$\{env:([^}:]+)(?::([^}]*))?\}")

        def resolve_value(value):
            if isinstance(value, str):
                match = env_pattern.match(value)
                if match:
                    var_name = match.group(1)
                    default = match.group(2)
                    return os.environ.get(var_name, default)
                return value
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            return value

        return resolve_value(data)

    def to_hash(self) -> str:
        """生成配置哈希，用于缓存键

        只包含影响结果的关键配置
        """
        relevant_config = {
            "dataset": self.dataset.full_name,
            "model": self.model.name,
            "bn_method": self.bn.build_method,
            "alpha": self.bn.alpha,
            "beta": self.bn.beta,
            "use_compensation_score": self.bn.use_compensation_score,
            "use_llm_construct": self.bn.use_llm_construct,
            "use_hierarchical": self.bn.use_hierarchical,
            "use_llm_inference": self.bn.use_llm_inference,
        }
        config_str = json.dumps(relevant_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]

    def get_dataset_paths(self) -> DatasetPaths:
        """获取当前数据集的路径"""
        return self.dataset.get_paths(self.data_root)

    def ensure_output_dirs(self) -> None:
        """确保输出目录存在"""
        self.output.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output.logs_dir.mkdir(parents=True, exist_ok=True)
        self.output.results_dir.mkdir(parents=True, exist_ok=True)
        self.output.figures_dir.mkdir(parents=True, exist_ok=True)


def create_config(
    dataset_name: str,
    variant: Optional[str] = None,
    model_name: str = "gpt-4o-mini-2024-07-18",
    build_method: str = "HYBRID",
    **kwargs,
) -> Config:
    """便捷函数：创建配置对象

    Args:
        dataset_name: 数据集名称
        variant: 数据集变体
        model_name: LLM 模型名称
        build_method: BN 构建方法
        **kwargs: 其他配置参数

    Returns:
        Config 对象
    """
    dataset_config = DatasetConfig(name=dataset_name, variant=variant)
    model_config = ModelConfig(name=model_name)
    bn_config = BNConfig(build_method=build_method)

    # 从 kwargs 中提取各子配置的参数
    bn_params = {k: v for k, v in kwargs.items() if k in BNConfig.model_fields}
    exp_params = {k: v for k, v in kwargs.items() if k in ExperimentConfig.model_fields}
    out_params = {k: v for k, v in kwargs.items() if k in OutputConfig.model_fields}

    if bn_params:
        bn_config = BNConfig(build_method=build_method, **bn_params)
    if exp_params:
        experiment_config = ExperimentConfig(**exp_params)
    else:
        experiment_config = ExperimentConfig()
    if out_params:
        output_config = OutputConfig(**out_params)
    else:
        output_config = OutputConfig()

    return Config(
        dataset=dataset_config,
        model=model_config,
        bn=bn_config,
        experiment=experiment_config,
        output=output_config,
    )
