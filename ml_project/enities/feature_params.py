from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class FeatureParams:
    categorical_features: Optional[List[str]]
    simple_categorical_features: Optional[List[str]]
    numerical_features: Optional[List[str]]
    target_col: Optional[str]
    transformer_features_log: Optional[List[str]] = field(default=tuple())
    transformer_features_square: Optional[List[str]] = field(default=tuple())
