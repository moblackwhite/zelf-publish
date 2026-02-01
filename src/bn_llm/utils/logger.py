"""BN-LLM 日志模块

提供统一的日志配置和管理功能。
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# 全局日志器实例
_logger: Optional[logging.Logger] = None


def setup_logger(
    name: str = "bn_llm",
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    dataset_name: Optional[str] = None,
    model_name: Optional[str] = None,
    to_console: bool = True,
    to_file: bool = True,
    run_id: Optional[str] = None,
) -> tuple[logging.Logger, Optional[str]]:
    """配置并返回日志器

    Args:
        name: 日志器名称
        log_dir: 日志目录，默认为 outputs/logs
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        dataset_name: 数据集名称，用于日志文件命名
        model_name: 模型名称，用于日志文件路径
        to_console: 是否输出到控制台
        to_file: 是否输出到文件
        run_id: 运行 ID（时间戳），如果未提供则自动生成

    Returns:
        元组 (配置好的日志器, 运行ID时间戳)
    """
    global _logger

    # 如果已存在同名日志器且已配置，直接返回
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger, run_id

    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False

    # 日志格式
    log_format = "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # 控制台处理器
    if to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 生成运行 ID（时间戳）
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 文件处理器
    if to_file:
        if log_dir is None:
            log_dir = Path("outputs/logs")

        # 构建日志路径
        if dataset_name:
            if model_name:
                # 提取简短名称用于文件路径（处理如 openrouter/google/gemini-3-flash-preview）
                short_model_name = model_name.rsplit("/", 1)[-1]
                log_path = log_dir / dataset_name / short_model_name
            else:
                log_path = log_dir / dataset_name
        else:
            log_path = log_dir

        log_path.mkdir(parents=True, exist_ok=True)

        # 使用 run_id 作为日志文件名
        log_file = log_path / f"{run_id}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"日志文件: {log_file}")

    _logger = logger
    return logger, run_id


def get_logger(name: str = "bn_llm") -> logging.Logger:
    """获取日志器

    如果全局日志器已配置，返回全局日志器。
    否则返回指定名称的日志器（可能未配置）。

    Args:
        name: 日志器名称

    Returns:
        日志器实例
    """
    global _logger

    if _logger is not None:
        return _logger

    # 返回基础日志器
    logger = logging.getLogger(name)
    if not logger.handlers:
        # 添加基础控制台处理器
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


class LoggerContext:
    """日志器上下文管理器

    用于在特定代码块中临时改变日志级别。

    Example:
        >>> with LoggerContext("bn_llm", "DEBUG"):
        ...     logger.debug("This will be logged")
    """

    def __init__(self, name: str, level: str):
        self.name = name
        self.level = level
        self._original_level = None

    def __enter__(self):
        logger = logging.getLogger(self.name)
        self._original_level = logger.level
        logger.setLevel(getattr(logging, self.level.upper()))
        return logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._original_level is not None:
            logger = logging.getLogger(self.name)
            logger.setLevel(self._original_level)
        return False


__all__ = ["setup_logger", "get_logger", "LoggerContext"]
