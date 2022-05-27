from typing import List, Union

import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, conlist, validator

# количество признаков
FEATURE_NUMBER = 13

# описание признаков:
# * порядковый номер,
# * название,
# * тип,
# * мин/макс значения,
FEATURES = [
    (0, "age", "num", 29, 77),
    (1, "sex", "cat", 0, 1),
    (2, "cp", "cat", 0, 3),
    (3, "trestbps", "num", 94, 200),
    (4, "chol", "num", 126, 564),
    (5, "fbs", "cat", 0, 1),
    (6, "restecg", "cat", 0, 2),
    (7, "thalach", "num", 71, 202),
    (8, "exang", "cat", 0, 1),
    (9, "oldpeak", "num", 0.0, 6.2),
    (10, "slope", "cat", 0, 2),
    (11, "ca", "cat", 0, 3),
    (12, "thal", "cat", 0, 2)
]
features_df = pd.DataFrame(
    FEATURES,
    columns=["id", "feature", "type", "min", "max"]
)


class HeartDiseaseModel(BaseModel):
    data: List[conlist(Union[float, int], min_items=13, max_items=13)]
    features: List[str]

    @validator("features")
    def check_feature_number(cls, v):
        """Проверка на количество колонок."""
        if FEATURE_NUMBER != len(v):
            raise HTTPException(
                status_code=400,
                detail=f"should be {FEATURE_NUMBER} features, got {len(v)}")
        return v

    @validator("features")
    def check_feature_order(cls, v):
        """Проверка последовательности признаков."""
        if v != features_df.feature.tolist():
            raise HTTPException(
                status_code=400,
                detail=f"feature order should be following {features_df.feature.tolist()}, got {v}"
            )
        return v

    @validator("data")
    def check_feature_type(cls, v):
        """Проверка типа признаков."""
        for i, value in enumerate(v[0]):
            if not isinstance(value, int) and not isinstance(value, float):
                print(value)
                raise HTTPException(
                    status_code=400,
                    detail=f"feature #{i} ({features_df.loc[i, 'feature']}) "
                           f"should be float or int, got {type(value)}"
                )
        return v

    @validator("data")
    def check_range_value(cls, v):
        """Проверка диапазона значений признаков."""
        for i, value in enumerate(v[0]):

            if features_df.loc[i, "type"] == "cat":
                value_max = int(features_df.loc[i, "max"])
                value_min = int(features_df.loc[i, "min"])
                value_possible = list(range(value_min, value_max + 1))
                if value not in value_possible:
                    raise HTTPException(
                        status_code=400,
                        detail=f"feature #{i} ({features_df.loc[i, 'feature']}) "
                               f"should have following value: {value_possible}, "
                               f"got {value}"
                    )

            if features_df.loc[i, "type"] == "num":
                delta = 0.1  # коэффициент запаса
                value_max = features_df.loc[i, "max"]
                value_min = features_df.loc[i, "min"]
                value_max += value_max * delta
                value_min -= value_min * delta
                if not value_min <= value <= value_max:
                    raise HTTPException(
                        status_code=400,
                        detail=f"feature #{i} ({features_df.loc[i, 'feature']}) "
                               f"should be in interval [{value_min}, {value_max}], "
                               f"got {value}"
                    )
        return v


class HeartDiseaseModelResponse(BaseModel):
    condition: int
    condition_proba: float
