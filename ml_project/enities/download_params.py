from dataclasses import dataclass


@dataclass()
class DownloadParams:
    train_path: str
    test_path: str
    train_url: str
    test_url: str
    output_folder: str
