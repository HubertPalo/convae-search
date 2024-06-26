# Python imports
from dataclasses import dataclass
from typing import List, Optional, Union

@dataclass
class ResourceConfig:
    gpu: float = 1
    cpu: float = 1

@dataclass
class SearchSpaceUnit:
    identifier: str
    tune_function: str
    tune_parameters: list
    route: str
    extra_features: Optional[List] = None

@dataclass
class ValidationConfig:
    validation_type: str
    validation_subtype: str
    validation_parameters: Optional[List] = None
    exception_value: Optional[Union[int, float, str]] = None

@dataclass
class ExplorationConfig:
    resources: ResourceConfig
    search_space: List[SearchSpaceUnit]
    initial_params: Optional[dict] = None
    additional_info: Optional[dict] = None
    validation: Optional[List[ValidationConfig]] = None

