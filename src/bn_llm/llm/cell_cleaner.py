"""单元格清洗器模块

使用 LangChain 框架通过 LLM 清洗数据单元格。
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from ..utils.logger import get_logger
from .context_builder import ContextBuilder, ContextType
from .prompts import CELL_CLEANING_USER_TEMPLATE


def _fix_json_output(text: str) -> str:
    """修复 LLM 输出中的常见 JSON 格式问题

    处理以下问题：
    1. Markdown 代码块标记（```json ... ```）
    2. 末尾多余的逗号（trailing comma）

    Args:
        text: LLM 输出的原始文本

    Returns:
        修复后的 JSON 文本
    """
    if not isinstance(text, str):
        return text

    result = text

    # 1. 移除 markdown 代码块标记
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, result)
    if match:
        result = match.group(1).strip()

    # 2. 移除 JSON 对象/数组中末尾多余的逗号
    # 匹配 ",}" 或 ",]"（允许中间有空白字符）
    result = re.sub(r",(\s*[}\]])", r"\1", result)

    return result


def _preprocess_llm_output(output):
    """预处理 LLM 输出，修复常见的 JSON 格式问题

    此函数作为 LangChain 链中的中间步骤，在解析器之前处理输出。
    兼容 AIMessage 和纯字符串两种输出格式。

    Args:
        output: LLM 输出（AIMessage 或 str）

    Returns:
        处理后的输出，保持原有类型
    """
    if isinstance(output, AIMessage):
        output.content = _fix_json_output(output.content)
    elif isinstance(output, str):
        output = _fix_json_output(output)
    return output


class CleaningResult(BaseModel):
    """LLM 清洗结果"""

    correct_value: str = Field(description="The corrected value")
    confidence: str = Field(description="Confidence in correction (high, medium, low)")
    reasoning: str = Field(description="Explanation of the correction reasoning")


@dataclass
class CellCleaningResult:
    """单元格清洗结果"""

    row: int
    column: str
    original: str
    corrected: str
    confidence: str
    reasoning: str
    applied: bool
    success: bool
    error: Optional[str] = None


class CellCleaner:
    """单元格清洗器

    使用 LangChain 和 LLM 进行单元格级别的数据清洗。

    Example:
        >>> cleaner = CellCleaner(model_name="gpt-4o-mini")
        >>> result = cleaner.clean_cell(df, 0, "City")
        >>> print(result.correct_value)
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 3,
        max_workers: int = 32,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """初始化单元格清洗器

        Args:
            model_name: LLM 模型名称
            temperature: 生成温度
            max_retries: 最大重试次数
            max_workers: 最大并行线程数
            api_base: API 基础 URL
            api_key: API 密钥
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.api_base = api_base
        self.api_key = api_key
        self.logger = get_logger()

        self._llm = None
        self._chain = None

    @property
    def llm(self) -> ChatLiteLLM:
        """懒加载 LLM 实例"""
        if self._llm is None:
            kwargs = {
                "model": self.model_name,
                "temperature": self.temperature,
            }
            if self.api_base:
                kwargs["api_base"] = self.api_base
            if self.api_key:
                kwargs["api_key"] = self.api_key

            self._llm = ChatLiteLLM(**kwargs)
        return self._llm

    @property
    def chain(self):
        """懒加载清洗链"""
        if self._chain is None:
            parser = PydanticOutputParser(pydantic_object=CleaningResult)

            prompt = PromptTemplate(
                input_variables=["context", "curr_value"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
                template=CELL_CLEANING_USER_TEMPLATE,
            )

            # 添加预处理步骤，清理可能的 markdown 代码块标记
            # 兼容所有模型：GPT 等模型的输出不受影响，DeepSeek 等模型的输出会被正确清理
            preprocessor = RunnableLambda(_preprocess_llm_output)
            self._chain = prompt | self.llm | preprocessor | parser

        return self._chain

    def clean_cell(
        self,
        df: pd.DataFrame,
        row_idx: int,
        col_name: str,
        relationship_dict: Optional[Dict[str, List[str]]] = None,
        bn_query_contexts: Optional[Dict[str, str]] = None,
        context_type: ContextType = ContextType.FULL,
    ) -> CleaningResult:
        """清洗单个单元格

        Args:
            df: 数据 DataFrame
            row_idx: 行索引
            col_name: 列名
            relationship_dict: 字段关系字典
            bn_query_contexts: BN 推理查询上下文
            context_type: 上下文类型

        Returns:
            清洗结果
        """
        curr_value = str(df.loc[row_idx, col_name])

        # 构建上下文
        context_builder = ContextBuilder(context_type=context_type)
        context = context_builder.build(df, row_idx, col_name, relationship_dict, bn_query_contexts)

        # 重试机制
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self.chain.invoke(
                    {
                        "context": context,
                        "curr_value": curr_value,
                    }
                )

                if attempt > 0:
                    self.logger.info(f"成功处理 ({row_idx}, {col_name})，第 {attempt + 1} 次尝试")

                return result

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = 2**attempt
                    self.logger.warning(
                        f"处理 ({row_idx}, {col_name}) 失败，"
                        f"第 {attempt + 1} 次尝试: {e}，"
                        f"{delay} 秒后重试..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"处理 ({row_idx}, {col_name}) 失败，"
                        f"已用完所有 {self.max_retries + 1} 次尝试"
                    )

        # 所有尝试都失败
        return CleaningResult(
            correct_value=curr_value,
            confidence="low",
            reasoning=f"处理失败: {str(last_error)}",
        )

    def clean_batch(
        self,
        df: pd.DataFrame,
        check_list: List[Tuple[int, str]],
        relationship_dict: Optional[Dict[str, List[str]]] = None,
        bn_query_contexts: Optional[Dict[str, str]] = None,
        context_type: ContextType = ContextType.FULL,
    ) -> Tuple[pd.DataFrame, List[CellCleaningResult]]:
        """批量清洗单元格

        Args:
            df: 数据 DataFrame
            check_list: 待清洗的 (row_idx, col_name) 列表
            relationship_dict: 字段关系字典
            bn_query_contexts: BN 推理查询上下文
            context_type: 上下文类型

        Returns:
            (清洗后的 DataFrame, 清洗结果列表)
        """
        cleaned_df = df.copy()
        results = []

        # 过滤有效的检查项
        valid_check_list = [
            (idx, col) for idx, col in check_list if idx < len(df) and col in df.columns
        ]

        if not valid_check_list:
            self.logger.warning("没有有效的单元格需要处理")
            return cleaned_df, []

        self.logger.info(f"使用 {self.max_workers} 个线程处理 {len(valid_check_list)} 个单元格")

        def process_cell(idx: int, col: str) -> CellCleaningResult:
            try:
                curr_value = str(df.loc[idx, col])
                result = self.clean_cell(
                    df, idx, col, relationship_dict, bn_query_contexts, context_type
                )

                should_apply = result.confidence.lower() in ["high", "medium"]

                return CellCleaningResult(
                    row=idx,
                    column=col,
                    original=curr_value,
                    corrected=result.correct_value,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    applied=should_apply,
                    success=True,
                )
            except Exception as e:
                return CellCleaningResult(
                    row=idx,
                    column=col,
                    original=str(df.loc[idx, col]) if idx < len(df) else "N/A",
                    corrected=str(df.loc[idx, col]) if idx < len(df) else "N/A",
                    confidence="low",
                    reasoning=f"处理错误: {str(e)}",
                    applied=False,
                    success=False,
                    error=str(e),
                )

        # 并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_cell = {
                executor.submit(process_cell, idx, col): (idx, col) for idx, col in valid_check_list
            }

            for future in as_completed(future_to_cell):
                idx, col = future_to_cell[future]
                try:
                    result = future.result()
                    results.append(result)

                    if result.success and result.applied:
                        cleaned_df.loc[idx, col] = result.corrected

                    self.logger.debug(
                        f"处理 ({idx}, {col}): "
                        f"'{result.original}' -> '{result.corrected}' "
                        f"[置信度: {result.confidence}]"
                    )

                except Exception as e:
                    self.logger.error(f"处理 ({idx}, {col}) 时出错: {e}")
                    results.append(
                        CellCleaningResult(
                            row=idx,
                            column=col,
                            original=str(df.loc[idx, col]),
                            corrected=str(df.loc[idx, col]),
                            confidence="low",
                            reasoning=f"处理错误: {str(e)}",
                            applied=False,
                            success=False,
                            error=str(e),
                        )
                    )

        self.logger.info(f"完成处理 {len(results)} 个单元格")
        return cleaned_df, results


__all__ = [
    "CleaningResult",
    "CellCleaningResult",
    "CellCleaner",
]
