from dataclasses import dataclass


@dataclass
class DataValidationArtifact:
    good_data_dir: str
    archived_data_dir: str


@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str


@dataclass
class DataPreparationArtifact:
    preprocessor_path: str
    clusterer_path: str
    transformed_feature_store_file_path: str

