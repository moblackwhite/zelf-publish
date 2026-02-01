"""提示词模板模块

集中管理所有 LLM 提示词模板，便于维护和修改。
"""

# =============================================================================
# 单元格清洗提示词
# =============================================================================

CELL_CLEANING_SYSTEM_PROMPT = """You are a data cleaning expert. Your task is to clean and correct potentially dirty data values."""

CELL_CLEANING_USER_TEMPLATE = """The following context contains information about the data, including
row context with related fields, and similar rows from the dataset:

{context}

The current value that needs correction is: {curr_value}

Please analyze the context thoroughly to determine the most likely correct value. Pay special attention to:
1. Patterns in the column data
2. Relationships between fields in the same row
3. Values in similar rows that might indicate the correct value
4. Common data entry errors (typos, transpositions, missing digits, etc.)

Return your answer in the following format:

{format_instructions}

If the value appears correct already, keep it as is but still provide your confidence assessment.
Explain your reasoning clearly, referring to specific evidence from the context."""


# =============================================================================
# 上下文构建提示词片段
# =============================================================================

CONTEXT_HEADER = """You are an expert at cleaning tabular data. Your task is to clean and correct potentially dirty data values."""

CONTEXT_CELL_INFO = (
    """CELL TO CLEAN: Row {row_idx}, Column '{col_name}', Current Value: '{current_value}'"""
)

CONTEXT_BN_INFO_HEADER = """## BN INFERENCE INFORMATION"""

CONTEXT_RELATED_FIELDS_HINT = """The {col_name} may be related to fields: {related_fields}"""

CONTEXT_NEARBY_ROWS_HEADER = """## RELATED FIELDS IN CURRENT AND SURROUNDING ROWS"""

CONTEXT_SIMILAR_ROWS_HEADER = """## SIMILAR ROWS FROM THE DATASET"""

CONTEXT_RANDOM_ROWS_HEADER = """## SOME ROWS FROM THE DATASET"""

CONTEXT_WARNING = """IMPORTANT: The surrounding context values may also contain errors. Focus on pattern recognition and data consistency rather than exact matches."""


# =============================================================================
# 函数依赖生成提示词
# =============================================================================

FD_GENERATION_SYSTEM_PROMPT = """You are an expert in data analysis and functional dependencies. Your task is to analyze column relationships in tabular data."""

FD_GENERATION_USER_TEMPLATE = """Analyze the following columns and determine if there are functional dependencies between them.

Dataset: {dataset_name}
Columns: {columns}

Sample data:
{sample_data}

For each pair of columns, determine if one column functionally determines another (A -> B means if we know A, we can determine B).

Return your analysis in JSON format with a matrix of dependency scores (0 to 1)."""


__all__ = [
    # 单元格清洗
    "CELL_CLEANING_SYSTEM_PROMPT",
    "CELL_CLEANING_USER_TEMPLATE",
    # 上下文构建
    "CONTEXT_HEADER",
    "CONTEXT_CELL_INFO",
    "CONTEXT_BN_INFO_HEADER",
    "CONTEXT_RELATED_FIELDS_HINT",
    "CONTEXT_NEARBY_ROWS_HEADER",
    "CONTEXT_SIMILAR_ROWS_HEADER",
    "CONTEXT_RANDOM_ROWS_HEADER",
    "CONTEXT_WARNING",
    # 函数依赖
    "FD_GENERATION_SYSTEM_PROMPT",
    "FD_GENERATION_USER_TEMPLATE",
]
