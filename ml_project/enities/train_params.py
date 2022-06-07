from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model_type: str = field(default="LogisticRegression")
    random_state: int = field(default=17)
    n_estimators: int = field(default=100)
