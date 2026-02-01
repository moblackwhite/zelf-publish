"""Functional Dependency Generator Module

NOTE: This is a placeholder file. The full implementation will be released after paper acceptance.
"""

from enum import Enum
from typing import Literal, Optional

import dspy
import pandas as pd

from ..utils.logger import get_logger


class FDMethod(str, Enum):
    FD = "FD"
    DFD = "DFD"
    FDS = "FDS"


class ClassifyFunctionalDependency(dspy.Signature):
    dataset_name: str = dspy.InputField()
    determinant: str = dspy.InputField()
    dependent: str = dspy.InputField()
    sample: str = dspy.InputField()
    functional_dependency: Literal["yes", "no"] = dspy.OutputField()
    confidence: float = dspy.OutputField()


class DetectFunctionalDependency(dspy.Signature):
    dataset_name: str = dspy.InputField()
    determinant_attr: str = dspy.InputField()
    dependent_attr: str = dspy.InputField()
    data_sample: str = dspy.InputField()
    has_dependency: Literal["yes", "no"] = dspy.OutputField()
    confidence_score: float = dspy.OutputField()


class FunctionalDependencyStrength(dspy.Signature):
    dataset_name: str = dspy.InputField()
    determinant: str = dspy.InputField()
    dependent: str = dspy.InputField()
    sample: str = dspy.InputField()
    fd_score: float = dspy.OutputField()


class FDGenerator:
    def __init__(
        self,
        method: FDMethod = FDMethod.DFD,
        sample_size: int = 10,
        random_state: int = 1234,
    ):
        self.method = FDMethod(method) if isinstance(method, str) else method
        self.sample_size = sample_size
        self.random_state = random_state
        self.logger = get_logger()

    def generate(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        max_workers: Optional[int] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError("Implementation hidden")

    def _detect_fd(
        self,
        determinant: str,
        dependent: str,
        dataset_name: str,
        sample_json: str,
    ) -> float:
        raise NotImplementedError("Implementation hidden")


def extract_functional_dependencies(
    data: pd.DataFrame,
    dataset_name: str,
    method: FDMethod = FDMethod.DFD,
    sample_size: int = 10,
    random_state: int = 1234,
) -> pd.DataFrame:
    generator = FDGenerator(
        method=method,
        sample_size=sample_size,
        random_state=random_state,
    )
    return generator.generate(data, dataset_name)


__all__ = [
    "FDMethod",
    "ClassifyFunctionalDependency",
    "DetectFunctionalDependency",
    "FunctionalDependencyStrength",
    "FDGenerator",
    "extract_functional_dependencies",
]
