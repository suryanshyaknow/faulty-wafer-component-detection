import os
from wafer.logger import lg
from datetime import datetime
from dataclasses import dataclass


RAW_DATA_DIR = "training_batch_files"
TRAINING_SCHEMA = "schema_training.json"
PREDICTION_SCHEMA = "schema_prediction.json"
FEATURE_STORE_FILE = "wafers.csv"
PREPROCESSOR = "preprocessor.pkl"
CLUSTERER = "clusterer.pkl"
ELBOW_PLOT = "kmeans_elbow.png"
MODELS_PERFORMANCE_REPORT = "report.yaml"


@dataclass
class BaseConfig:
    project: str = "wafer-fault-detection"
    target: str = "Good/Bad"


@dataclass
class DataSourceConfig:
    raw_data_dir = os.path.join(os.getcwd(), RAW_DATA_DIR)
    training_schema = os.path.join(os.getcwd(), TRAINING_SCHEMA)
    prediciton_schema = os.path.join(os.getcwd(), PREDICTION_SCHEMA)


@dataclass
class TrainingArtifactsConfig:
    try:
        # Training Pipeline's artifacts dir
        artifacts_dir: str = os.path.join(
            os.getcwd(), "artifacts", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        ...
    except Exception as e:
        lg.exception(e)
        raise e


class DataValidationConfig:
    def __init__(self):
        try:
            training_artifacts_config = TrainingArtifactsConfig()
            # Data Validation's artifacts dir
            self.data_validation_dir = os.path.join(
                training_artifacts_config.artifacts_dir, "data_validation")

            # Good Raw Data dir
            self.good_data_dir = os.path.join(
                self.data_validation_dir, "good_raw_data")
            # Bad Raw Data dir
            self.bad_data_dir = os.path.join(
                self.data_validation_dir, "bad_raw_data")
            # Archived Data dir
            self.archived_data_dir = os.path.join(
                self.data_validation_dir, "archived_data")
            ...
        except Exception as e:
            lg.exception(e)
            raise e


class DataIngestionConfig:
    def __init__(self):
        try:
            training_artifacts_config = TrainingArtifactsConfig()
            # Data Ingestion's artifacts dir
            self.data_ingestion_dir = os.path.join(
                training_artifacts_config.artifacts_dir, "data_ingestion")

            # Feature Store Dataset dir
            self.feature_store_file_path = os.path.join(
                self.data_ingestion_dir, "feature_store", FEATURE_STORE_FILE)
            ...
        except Exception as e:
            lg.exception(e)
            raise e


class DataPreparationConfig:
    def __init__(self):
        try:
            training_artifacts_config = TrainingArtifactsConfig()
            # Data Preparation's artifacts dir
            self.data_preparation_dir = os.path.join(
                training_artifacts_config.artifacts_dir, "data_preparation")

            # Preprocessor path
            self.preprocessor_path = os.path.join(
                self.data_preparation_dir, "preprocessor", PREPROCESSOR)
            # Clusterer path
            self.clusterer_path = os.path.join(
                self.data_preparation_dir, "clusterer", CLUSTERER)
            # Elbow Plot path
            self.elbow_plot_path = os.path.join(
                self.data_preparation_dir, "plots", ELBOW_PLOT)
            # Transformed Feature Store dataset path
            self.transformed_feature_store_file_path = os.path.join(
                self.data_preparation_dir, "preprocessed", FEATURE_STORE_FILE.replace(".csv", ".npz"))
            ...
        except Exception as e:
            lg.exception


class ModelTrainingConfig:
    def __init__(self):
        try:
            training_artifacts_config = TrainingArtifactsConfig()
            # Model Training's Artifacts dir
            self.model_training_dir = os.path.join(
                training_artifacts_config.artifacts_dir, "model_training")

            # Train-Test Split params
            self.test_size = 1/3
            self.random_state = 42
            # Cluster based Models dir
            self.cluster_based_models_dir = os.path.join(
                self.model_training_dir, "cluster_based_models")
            # Cross-Validation params
            self.cv_for_eval = 5
            self.cv_for_hypertuning = 2
            # Models' Performance Report
            self.performance_report_path = os.path.join(
                self.model_training_dir, "models_performace_report", MODELS_PERFORMANCE_REPORT)
            # Model Evaluation params
            self.expected_score = .85
            self.overfit_thresh = .1
            ...
        except Exception as e:
            lg.exception(e)
            raise e
