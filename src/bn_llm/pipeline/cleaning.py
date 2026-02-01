"""数据清洗流水线模块

提供完整的数据清洗流程，包括：
1. 函数依赖检测
2. 贝叶斯网络结构构建
3. 一致性修复
4. BN 推理修复
5. LLM 辅助修复（可选）
6. 结果评估
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork

from ..config import Config, create_config
from ..core.afd_measures import compute_mu_plus_matrix
from ..core.bayesian_network import (
    LocalNetworkManager,
    create_bn_builder,
    filter_weak_partitions,
)
from ..core.cell_selector import LowConfidenceCellSelector, compute_llm_repair_budget
from ..core.consistency import ConsistencyRepairer
from ..core.evaluator import (
    DataCleaningEvaluator,
    EvaluationResult,
    save_evaluation_result,
)
from ..core.occurrence_matrix import compute_occurrence_matrix
from ..core.repairer import DataRepairer, RepairResult
from ..llm.bn_constructor import LLMBDEUConstructor, save_bn_structure
from ..llm.cell_cleaner import CellCleaner
from ..llm.context_builder import ContextType
from ..llm.fd_generator import FDGenerator, FDMethod
from ..utils.cache import CacheCategory, CacheManager
from ..utils.data_loader import DataLoader
from ..utils.logger import get_logger


@dataclass
class CleaningPipelineResult:
    """清洗流水线结果"""

    repaired_data: pd.DataFrame
    evaluation: Optional[EvaluationResult] = None
    duration_seconds: float = 0.0
    config_hash: str = ""

    # 中间结果
    fd_matrix: Optional[pd.DataFrame] = None
    bn_structure: Optional[Dict] = None
    repair_result: Optional[RepairResult] = None

    # 统计信息
    steps_completed: List[str] = field(default_factory=list)


class CleaningPipeline:
    """数据清洗流水线

    完整的数据清洗流程，支持分步执行和缓存。

    Example:
        >>> config = create_config("hospital", variant="error_10")
        >>> pipeline = CleaningPipeline(config)
        >>> result = pipeline.run()
        >>> print(f"F1 Score: {result.evaluation.metrics.f1:.4f}")
    """

    def __init__(self, config: Config):
        """初始化清洗流水线

        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = get_logger()

        # 确保输出目录存在
        config.ensure_output_dirs()

        # 初始化组件
        self.data_loader = DataLoader(
            data_root=config.data_root,
            null_marker=config.experiment.null_marker,
        )
        self.cache_manager = CacheManager(config.output.cache_dir)

        # 状态
        self._dirty_data: Optional[pd.DataFrame] = None
        self._clean_data: Optional[pd.DataFrame] = None
        self._discrete_cols: Optional[List[str]] = None
        self._fd_matrix: Optional[pd.DataFrame] = None
        self._bn_structure: Optional[Dict] = None
        self._partitions: Optional[Dict[str, BayesianNetwork]] = None
        self._occurrence_matrix: Optional[Dict] = None
        self._repaired_data: Optional[pd.DataFrame] = None
        self._afd_matrix: Optional[pd.DataFrame] = None  # AFD 分数矩阵缓存
        self._repair_result: Optional[RepairResult] = None  # BN 推理修复结果

    def run(self, use_cache: bool = True) -> CleaningPipelineResult:
        """执行完整清洗流程

        Args:
            use_cache: 是否使用缓存

        Returns:
            清洗流水线结果
        """
        start_time = time.time()
        steps_completed = []

        self.logger.info(f"开始数据清洗流程: {self.config.dataset.full_name}")
        self.logger.info(f"构建方法: {self.config.bn.build_method}")

        # Step 1: 加载数据
        self._load_data()
        steps_completed.append("load_data")

        # Step 2: 生成函数依赖（如果使用 LLM 构建）
        if self.config.bn.use_llm_construct:
            self._fd_matrix = self.run_step_fd(use_cache=use_cache)
            steps_completed.append("fd_generation")

        # Step 3: 构建贝叶斯网络
        self._bn_structure = self.run_step_bn(use_cache=use_cache)
        steps_completed.append("bn_construction")

        # Step 4: BN 推理修复
        self._repair_result = self.run_step_infer()
        self._repaired_data = self._repair_result.repaired_data
        steps_completed.append("bn_inference")

        # Step 5: LLM 辅助修复
        if self.config.bn.use_llm_inference:
            self._repaired_data = self.run_step_llm_repair()
            steps_completed.append("llm_repair")

        # Step 6: 评估结果
        evaluation = self.run_step_evaluate()
        steps_completed.append("evaluation")

        # 保存修复后的数据
        self._save_repaired_data()

        duration = time.time() - start_time
        self.logger.info(f"清洗流程完成，耗时: {duration:.2f}s")

        return CleaningPipelineResult(
            repaired_data=self._repaired_data,
            evaluation=evaluation,
            duration_seconds=duration,
            config_hash=self.config.to_hash(),
            fd_matrix=self._fd_matrix,
            bn_structure=self._bn_structure,
            repair_result=self._repair_result,
            steps_completed=steps_completed,
        )

    def _load_data(self) -> None:
        """加载数据"""
        self._discrete_cols = self.data_loader.load_discrete_cols(self.config.dataset)
        self._dirty_data, self._clean_data = self.data_loader.load_both(
            self.config.dataset,
            self._discrete_cols,
        )

        self.logger.info(f"数据加载完成: {len(self._dirty_data)} 行, {len(self._discrete_cols)} 列")

    def _save_repaired_data(self) -> None:
        """保存修复后的数据到结果目录"""
        if self._repaired_data is None:
            return

        output_path = (
            self.config.output.get_run_results_dir(self.config.dataset.full_name) / "repaired.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._repaired_data.to_csv(output_path, index=False)
        self.logger.info(f"修复数据已保存: {output_path}")

    def _load_or_compute_afd_matrix(self, use_cache: bool = True) -> pd.DataFrame:
        """加载或计算 AFD 分数矩阵 (mu_plus)

        如果缓存中存在则加载，否则从数据计算。

        Args:
            use_cache: 是否使用缓存

        Returns:
            AFD 分数矩阵
        """
        # 如果已经计算过，直接返回
        if self._afd_matrix is not None:
            return self._afd_matrix

        if self._dirty_data is None:
            self._load_data()

        cache_key = f"{self.config.dataset.name}_mu_plus"

        # 尝试从缓存加载
        if use_cache:
            cached = self.cache_manager.load(
                CacheCategory.FD_RESULTS,
                cache_key,
                suffix=".csv",
            )
            if cached is not None:
                self.logger.info("从缓存加载 AFD 分数矩阵")
                self._afd_matrix = cached[self._discrete_cols]
                return self._afd_matrix

        # 计算 AFD 分数矩阵
        self.logger.info("计算 AFD 分数矩阵 (mu_plus)...")
        self._afd_matrix = compute_mu_plus_matrix(
            self._dirty_data,
            columns=self._discrete_cols,
            show_progress=True,
        )

        # 对角线自回环值设为 NaN
        np.fill_diagonal(self._afd_matrix.values, np.nan)

        # 保存到缓存
        self.cache_manager.save(
            CacheCategory.FD_RESULTS,
            cache_key,
            self._afd_matrix,
            suffix=".csv",
        )

        return self._afd_matrix

    def run_step_fd(self, use_cache: bool = True) -> pd.DataFrame:
        """步骤1: 生成函数依赖

        Args:
            use_cache: 是否使用缓存

        Returns:
            函数依赖矩阵
        """
        if self._dirty_data is None:
            self._load_data()

        cache_key = f"{self.config.dataset.full_name}_DFD_{self.config.model.short_name}"

        # 尝试加载缓存
        if use_cache:
            cached = self.cache_manager.load(
                CacheCategory.FD_RESULTS,
                cache_key,
                suffix=".csv",
            )
            if cached is not None:
                self.logger.info("从缓存加载函数依赖矩阵")
                return cached

        self.logger.info("使用 LLM 生成函数依赖...")

        # 配置 DSPy（需要在调用前配置好 LM）
        import dspy

        lm = dspy.LM(
            self.config.model.name,
            api_base=self.config.model.api_base,
            api_key=self.config.model.api_key,
            temperature=self.config.model.temperature,
        )
        dspy.configure(lm=lm)

        # 生成 FD
        generator = FDGenerator(
            method=FDMethod.DFD,
            sample_size=10,
            random_state=self.config.experiment.random_state,
        )
        fd_matrix = generator.generate(
            self._dirty_data,
            self.config.dataset.name,
        )

        # 保存缓存
        self.cache_manager.save(
            CacheCategory.FD_RESULTS,
            cache_key,
            fd_matrix,
            suffix=".csv",
        )

        return fd_matrix

    def run_step_bn(self, use_cache: bool = True) -> Dict:
        """步骤2: 构建贝叶斯网络

        Args:
            use_cache: 是否使用缓存

        Returns:
            BN 结构字典
        """
        if self._dirty_data is None:
            self._load_data()

        cache_key = f"{self.config.dataset.full_name}_{self.config.to_hash()}"

        # 尝试加载缓存
        if use_cache:
            cached = self.cache_manager.load(
                CacheCategory.BN_STRUCTURES,
                cache_key,
                suffix=".json",
            )
            if cached is not None:
                self.logger.info("从缓存加载 BN 结构")
                self._bn_structure = cached
                return cached

        self.logger.info("构建贝叶斯网络结构...")

        # 根据构建方法生成结构
        build_method = self.config.bn.build_method

        if build_method in ("HYBRID", "HYBRID_NO_CONSISTENCY"):
            # 需要 LLM FD 和 AFD 分数
            if self._fd_matrix is None:
                self._fd_matrix = self.run_step_fd(use_cache=use_cache)

            # 加载或计算 AFD 分数（mu_plus）
            score_afd = self._load_or_compute_afd_matrix(use_cache=use_cache)

            # 使用 LLM + BDEU 构建
            constructor = LLMBDEUConstructor(
                alpha=self.config.bn.alpha or 0.1,
                beta=self.config.bn.beta or 0.1,
            )
            bn_structure = constructor.construct(score_afd, self._fd_matrix)

        elif build_method == "BDEU_ONLY":
            # 仅使用 BDEU，无初始边
            bn_structure = {
                "nodes": self._discrete_cols,
                "consistencies": [],
                "initial_edges": [],
            }

        elif build_method == "LLM_ONLY":
            # 仅使用 LLM 边
            if self._fd_matrix is None:
                self._fd_matrix = self.run_step_fd(use_cache=use_cache)

            # 从 FD 矩阵提取边
            edges = []
            for i, row in enumerate(self._fd_matrix.index):
                for j, col in enumerate(self._fd_matrix.columns):
                    if row != col and self._fd_matrix.loc[row, col] > 0.5:
                        edges.append([row, col])

            bn_structure = {
                "nodes": self._discrete_cols,
                "consistencies": [],
                "initial_edges": edges,
            }
        else:
            # USER_DEFINED 或其他
            bn_structure = {
                "nodes": self._discrete_cols,
                "consistencies": [],
                "initial_edges": [],
            }

        # 保存缓存
        self.cache_manager.save(
            CacheCategory.BN_STRUCTURES,
            cache_key,
            bn_structure,
            suffix=".json",
        )

        # 保存到结果目录（使用运行 ID 子目录）
        output_path = (
            self.config.output.get_run_results_dir(self.config.dataset.full_name)
            / "bn_structure.json"
        )
        save_bn_structure(bn_structure, output_path)

        self._bn_structure = bn_structure
        return bn_structure

    def run_step_infer(self) -> RepairResult:
        """步骤3: BN 推理修复

        Returns:
            修复结果
        """
        if self._dirty_data is None:
            self._load_data()
        if self._bn_structure is None:
            self._bn_structure = self.run_step_bn()

        self.logger.info("执行 BN 推理修复...")

        # 1. 一致性修复
        consistencies = self._bn_structure.get("consistencies", [])
        build_method = self.config.bn.build_method

        if consistencies and build_method != "HYBRID_NO_CONSISTENCY":
            self.logger.info(f"执行一致性修复，{len(consistencies)} 个一致性组")
            consistency_repairer = ConsistencyRepairer(
                null_marker=self.config.experiment.null_marker,
            )
            repaired_data = consistency_repairer.repair(self._dirty_data, consistencies)

            # 获取被一致性修复的列（后续 BN 推理时排除）
            excluded_cols = consistency_repairer.get_consistency_columns(consistencies)
        else:
            repaired_data = self._dirty_data.copy()
            excluded_cols = []

        # 2. 构建贝叶斯网络
        initial_edges = self._bn_structure.get("initial_edges", [])
        initial_edges = [tuple(e) for e in initial_edges]

        builder = create_bn_builder(
            method=build_method,
            equivalent_sample_size=self.config.bn.equivalent_sample_size,
            alpha=self.config.bn.alpha,
            beta=self.config.bn.beta,
        )

        bn_model = builder.build(repaired_data, initial_edges)

        # 3. 创建局部网络
        network_manager = LocalNetworkManager()
        local_networks = network_manager.create_local_subnetworks(bn_model)

        # 过滤弱分区
        WEAK_PARTITION_THRESHOLD = 0.25
        strength_matrix = self._load_or_compute_afd_matrix(use_cache=True)
        local_networks = filter_weak_partitions(
            local_networks,
            strength_matrix,
            gamma=WEAK_PARTITION_THRESHOLD,
        )

        # 训练局部网络
        network_manager.train_local_networks(repaired_data, local_networks)

        # 排除一致性修复的列
        target_columns = [c for c in self._discrete_cols if c not in excluded_cols]

        # 4. 计算共现矩阵
        self.logger.info("计算共现矩阵...")
        self._occurrence_matrix = compute_occurrence_matrix(
            repaired_data,
            show_progress=True,
        )

        # 5. 执行修复
        self.logger.info("执行数据修复...")

        # 处理推理限制
        infer_limit = self.config.experiment.infer_limit
        if infer_limit > 0:
            data_to_repair = repaired_data.iloc[:infer_limit].copy()
        else:
            data_to_repair = repaired_data

        repairer = DataRepairer(
            partitions=local_networks,
            occurrence_matrix=self._occurrence_matrix,
            null_marker=self.config.experiment.null_marker,
            use_compensation_score=self.config.bn.use_compensation_score,
        )

        repair_result = repairer.repair(
            dirty_data=data_to_repair,
            target_columns=target_columns,
            # n_jobs=1,  # 串行执行以避免多进程问题
            chunk_size=self.config.experiment.batch_size,
        )

        self._repaired_data = repair_result.repaired_data
        self._partitions = local_networks

        # 保存缓存以支持单独运行 LLM 修复
        self._save_infer_result(repair_result)

        return repair_result

    def _get_infer_cache_key(self) -> str:
        """获取推理结果缓存键"""
        return f"{self.config.dataset.full_name}_{self.config.to_hash()}"

    def _save_infer_result(self, repair_result: RepairResult) -> None:
        """保存 BN 推理修复结果到缓存

        Args:
            repair_result: 修复结果
        """
        cache_key = self._get_infer_cache_key()

        # 1. 保存修复后的数据
        self.cache_manager.save(
            CacheCategory.REPAIR_RESULT,
            f"{cache_key}_repaired_data",
            repair_result.repaired_data,
            suffix=".csv",
        )

        # 2. 保存 repairs (元组键转换为字符串)
        repairs_serializable = {f"({k[0]}, {k[1]})": v for k, v in repair_result.repairs.items()}
        self.cache_manager.save(
            CacheCategory.REPAIR_RESULT,
            f"{cache_key}_repairs",
            repairs_serializable,
            suffix=".json",
        )

        # 3. 保存 query_contexts
        self.cache_manager.save(
            CacheCategory.REPAIR_RESULT,
            f"{cache_key}_query_contexts",
            repair_result.query_contexts,
            suffix=".json",
        )

        # 4. 保存 probabilities (元组键转换为字符串)
        probs_serializable = {
            f"({k[0]}, {k[1]})": v for k, v in repair_result.probabilities.items()
        }
        self.cache_manager.save(
            CacheCategory.REPAIR_RESULT,
            f"{cache_key}_probabilities",
            probs_serializable,
            suffix=".json",
        )

        # 5. 保存共现矩阵
        if self._occurrence_matrix is not None:
            self.cache_manager.save(
                CacheCategory.OCCURRENCE_MATRIX,
                cache_key,
                self._occurrence_matrix,
                suffix=".pkl",
            )

        self.logger.info(f"BN 推理结果已缓存: {cache_key}")

    def load_infer_result(self, use_cache: bool = True) -> bool:
        """加载 BN 推理修复结果缓存

        用于单独运行 run_step_llm_repair() 时加载之前的推理结果。

        Args:
            use_cache: 是否使用缓存

        Returns:
            是否成功加载缓存
        """
        if not use_cache:
            return False

        # 确保数据已加载（需要 _dirty_data 和 _discrete_cols）
        if self._dirty_data is None:
            self._load_data()

        cache_key = self._get_infer_cache_key()

        # 1. 加载修复后的数据
        repaired_data = self.cache_manager.load(
            CacheCategory.REPAIR_RESULT,
            f"{cache_key}_repaired_data",
            suffix=".csv",
        )
        if repaired_data is None:
            return False

        # 2. 加载 repairs
        repairs_raw = self.cache_manager.load(
            CacheCategory.REPAIR_RESULT,
            f"{cache_key}_repairs",
            suffix=".json",
        )
        if repairs_raw is None:
            return False

        # 转换字符串键回元组
        repairs = {}
        for key, value in repairs_raw.items():
            # 解析 "(row_idx, col_name)" 格式
            key = key.strip("()")
            parts = key.split(", ", 1)
            row_idx = int(parts[0])
            col_name = parts[1].strip("'\"")
            repairs[(row_idx, col_name)] = value

        # 3. 加载 query_contexts
        query_contexts = self.cache_manager.load(
            CacheCategory.REPAIR_RESULT,
            f"{cache_key}_query_contexts",
            suffix=".json",
        )
        if query_contexts is None:
            query_contexts = {}

        # 4. 加载 probabilities
        probs_raw = self.cache_manager.load(
            CacheCategory.REPAIR_RESULT,
            f"{cache_key}_probabilities",
            suffix=".json",
        )
        probabilities = {}
        if probs_raw is not None:
            for key, value in probs_raw.items():
                key = key.strip("()")
                parts = key.split(", ", 1)
                row_idx = int(parts[0])
                col_name = parts[1].strip("'\"")
                probabilities[(row_idx, col_name)] = value

        # 5. 加载共现矩阵
        occurrence_matrix = self.cache_manager.load(
            CacheCategory.OCCURRENCE_MATRIX,
            cache_key,
            suffix=".pkl",
        )

        # 6. 加载 BN 结构
        if self._bn_structure is None:
            self._bn_structure = self.run_step_bn(use_cache=True)

        # 构建 RepairResult
        self._repair_result = RepairResult(
            repaired_data=repaired_data,
            repairs=repairs,
            probabilities=probabilities,
            penalties={},  # penalties 不是必需的
            query_contexts=query_contexts,
        )
        self._repaired_data = repaired_data
        self._occurrence_matrix = occurrence_matrix

        self.logger.info(f"从缓存加载 BN 推理结果: {cache_key}")
        return True

    def run_step_llm_repair(self, use_cache: bool = True) -> pd.DataFrame:
        """步骤4: LLM 修复

        使用 LLM 对 BN 推理修复的单元格进行二次确认和清洗。
        基于置信度筛选低置信度单元格，控制 LLM 调用成本。

        支持单独运行：如果没有运行过 run_step_infer()，会尝试从缓存加载。

        Args:
            use_cache: 是否使用缓存加载之前的推理结果

        Returns:
            修复后的数据
        """
        # 尝试从缓存加载推理结果（支持单独运行）
        if self._repaired_data is None or self._repair_result is None:
            if use_cache and self.load_infer_result(use_cache=True):
                self.logger.info("已从缓存加载推理结果，继续执行 LLM 修复")
            else:
                raise ValueError(
                    "请先执行 run_step_infer() 或确保存在推理结果缓存。\n"
                    "可通过 pipeline.load_infer_result() 手动加载缓存。"
                )

        self.logger.info("执行 LLM 辅助修复...")

        # 1. 获取 BN 推理修复的单元格列表
        all_repair_cells = list(self._repair_result.repairs.keys())

        if not all_repair_cells:
            self.logger.info("没有需要 LLM 修复的单元格")
            return self._repaired_data

        self.logger.info(f"BN 推理修复的单元格总数: {len(all_repair_cells)}")

        # 2. 计算 LLM 修复预算
        total_cells = self._repaired_data.shape[0] * self._repaired_data.shape[1]
        budget = compute_llm_repair_budget(
            total_cells=total_cells,
            budget_ratio=self.config.experiment.llm_repair_budget_ratio,
            max_budget=self.config.experiment.llm_repair_max_budget,
            candidate_count=len(all_repair_cells),
        )

        self.logger.info(
            f"LLM 修复预算: {budget} "
            f"(比例: {self.config.experiment.llm_repair_budget_ratio:.1%}, "
            f"总单元格: {total_cells})"
        )

        # 3. 基于置信度筛选低置信度单元格
        edges = self._bn_structure.get("initial_edges", []) if self._bn_structure else []
        selector = LowConfidenceCellSelector(
            data=self._repaired_data,
            occurrence_matrix=self._occurrence_matrix or {},
            edges=edges,
            probabilities=self._repair_result.probabilities,
        )
        selection_result = selector.select(all_repair_cells, budget)
        check_list = selection_result.selected_cells

        if len(check_list) < len(all_repair_cells):
            self.logger.info(f"筛选低置信度单元格: {len(check_list)}/{len(all_repair_cells)}")
        else:
            self.logger.info("预算充足，检查所有修复单元格")

        # 4. 从 BN 结构构建字段关系字典
        relationship_dict = self._build_relationship_dict()

        # 5. 获取 BN 推理查询上下文
        bn_query_contexts = self._repair_result.query_contexts

        # 6. 初始化单元格清洗器
        cleaner = CellCleaner(
            model_name=self.config.model.name,
            temperature=self.config.model.temperature,
            max_workers=self.config.experiment.worker_count,
            api_base=self.config.model.api_base,
            api_key=self.config.model.api_key,
        )

        # 7. 执行批量清洗
        cleaned_df, cleaning_results = cleaner.clean_batch(
            df=self._repaired_data,
            check_list=check_list,
            relationship_dict=relationship_dict,
            bn_query_contexts=bn_query_contexts,
            context_type=ContextType.FULL,
        )

        # 8. 统计清洗结果
        applied_count = sum(1 for r in cleaning_results if r.applied)
        high_conf_count = sum(1 for r in cleaning_results if r.confidence.lower() == "high")
        medium_conf_count = sum(1 for r in cleaning_results if r.confidence.lower() == "medium")

        self.logger.info(
            f"LLM 清洗完成: 应用修复 {applied_count}/{len(cleaning_results)}, "
            f"高置信度: {high_conf_count}, 中置信度: {medium_conf_count}"
        )

        self._repaired_data = cleaned_df
        return self._repaired_data

    def _build_relationship_dict(self) -> Dict[str, List[str]]:
        """从 BN 结构构建字段关系字典

        Returns:
            字段关系字典，键为字段名，值为相关字段列表
        """
        relationship_dict: Dict[str, List[str]] = {}

        if self._bn_structure is None:
            return relationship_dict

        # 从 BN 初始边构建关系
        initial_edges = self._bn_structure.get("initial_edges", [])
        for edge in initial_edges:
            if len(edge) >= 2:
                parent, child = edge[0], edge[1]
                # 双向添加关系
                if parent not in relationship_dict:
                    relationship_dict[parent] = []
                if child not in relationship_dict[parent]:
                    relationship_dict[parent].append(child)

                if child not in relationship_dict:
                    relationship_dict[child] = []
                if parent not in relationship_dict[child]:
                    relationship_dict[child].append(parent)

        # 从局部网络分区添加关系
        if self._partitions:
            for col_name, bn_model in self._partitions.items():
                if col_name not in relationship_dict:
                    relationship_dict[col_name] = []

                # 添加父节点和子节点作为相关字段
                parents = list(bn_model.get_parents(col_name))
                children = list(bn_model.get_children(col_name))

                for parent in parents:
                    if parent not in relationship_dict[col_name]:
                        relationship_dict[col_name].append(parent)

                for child in children:
                    if child not in relationship_dict[col_name]:
                        relationship_dict[col_name].append(child)

        return relationship_dict

    def run_step_evaluate(self) -> EvaluationResult:
        """步骤5: 评估结果

        Returns:
            评估结果
        """
        if self._repaired_data is None:
            raise ValueError("请先执行修复步骤")
        if self._clean_data is None:
            self._load_data()

        self.logger.info("评估清洗结果...")

        # 处理推理限制
        infer_limit = self.config.experiment.infer_limit
        if infer_limit > 0:
            dirty_eval = self._dirty_data.iloc[:infer_limit]
            clean_eval = self._clean_data.iloc[:infer_limit]
            repaired_eval = self._repaired_data.iloc[:infer_limit]
        else:
            dirty_eval = self._dirty_data
            clean_eval = self._clean_data
            repaired_eval = self._repaired_data

        evaluator = DataCleaningEvaluator()
        result = evaluator.evaluate(dirty_eval, repaired_eval, clean_eval)
        evaluator.log_summary(result)

        # 保存评估结果（使用运行 ID 子目录）
        output_path = (
            self.config.output.get_run_results_dir(self.config.dataset.full_name)
            / "evaluation.json"
        )
        save_evaluation_result(result, output_path)

        return result

    @property
    def dirty_data(self) -> Optional[pd.DataFrame]:
        """获取脏数据"""
        return self._dirty_data

    @property
    def clean_data(self) -> Optional[pd.DataFrame]:
        """获取干净数据"""
        return self._clean_data

    @property
    def repaired_data(self) -> Optional[pd.DataFrame]:
        """获取修复后的数据"""
        return self._repaired_data


def run_cleaning_pipeline(
    dataset_name: str,
    variant: Optional[str] = None,
    model_name: str = "gpt-4o-mini-2024-07-18",
    build_method: str = "HYBRID",
    use_cache: bool = True,
    **kwargs,
) -> CleaningPipelineResult:
    """便捷函数：运行数据清洗流水线

    Args:
        dataset_name: 数据集名称
        variant: 数据集变体
        model_name: LLM 模型名称
        build_method: BN 构建方法
        use_cache: 是否使用缓存
        **kwargs: 其他配置参数

    Returns:
        清洗流水线结果
    """
    config = create_config(
        dataset_name=dataset_name,
        variant=variant,
        model_name=model_name,
        build_method=build_method,
        **kwargs,
    )

    pipeline = CleaningPipeline(config)
    return pipeline.run(use_cache=use_cache)


__all__ = [
    "CleaningPipelineResult",
    "CleaningPipeline",
    "run_cleaning_pipeline",
]
