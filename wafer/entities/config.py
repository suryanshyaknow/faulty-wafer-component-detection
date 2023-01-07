import os
from wafer.logger import lg
from datetime import datetime
from dataclasses import dataclass


RAW_DATA_DIR = "training_batch_files"
PREDICTION_BATCHES_DIR = "prediction_batch_files"
TRAINING_SCHEMA = "schema_training.json"
PREDICTION_SCHEMA = "schema_prediction.json"
FEATURE_STORE_FILE = "wafers.csv"
TRAINING_SET = "train.csv"
TEST_SET = "test.csv"
PREPROCESSOR = "preprocessor.pkl"
CLUSTERER = "clusterer.pkl"
ELBOW_PLOT = "kmeans_elbow.png"
MODELS_PERFORMANCE_REPORT = "report.json"
PREDICTION_DIR = "predictions"


@dataclass
class BaseConfig:
    project: str = "wafer-fault-detection"
    target: str = "Good/Bad"


@dataclass
class DataSourceConfig:
    raw_data_dir = os.path.join(os.getcwd(), RAW_DATA_DIR)
    training_schema = os.path.join(os.getcwd(), TRAINING_SCHEMA)
    prediction_schema = os.path.join(os.getcwd(), PREDICTION_SCHEMA)
    prediction_batches_dir = os.path.join(
        os.getcwd(), PREDICTION_BATCHES_DIR)


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

            # Feature Store Dataset path
            self.feature_store_file_path = os.path.join(
                self.data_ingestion_dir, "feature_store", FEATURE_STORE_FILE)
            self.test_size = 0.2
            self.random_state = 42
            # Training Set path
            self.training_set_path = os.path.join(
                self.data_ingestion_dir, "datasets", TRAINING_SET)
            # Test Set path
            self.test_set_path = os.path.join(
                self.data_ingestion_dir, "datasets", TEST_SET)
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
            self.prepared_training_set_path = os.path.join(
                self.data_preparation_dir, "preprocessed", TRAINING_SET.replace(".csv", ".npz"))
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
            self.overfit_thresh = .1
            ...
        except Exception as e:
            lg.exception(e)
            raise e


class ModelEvaluationConfig:
    def __init__(self) -> None:
        try:
            self.replace_models_thresh = 0.01
        except Exception as e:
            lg.exception(e)
            raise e


class ModelPushingConfig:
    def __init__(self) -> None:
        try:
            training_artifacts_config = TrainingArtifactsConfig()
            self.model_pushing_dir = os.path.join(
                training_artifacts_config.artifacts_dir, "model_pushing")

            self.saved_models_dir = os.path.join(
                self.model_pushing_dir, "saved_models")
            self.to_be_pushed_models_dir = os.path.join(
                self.model_pushing_dir, "models")
            self.to_be_pushed_processor_path = os.path.join(
                self.model_pushing_dir, PREPROCESSOR)
            self.to_be_pushed_clusterer_path = os.path.join(
                self.model_pushing_dir, CLUSTERER)
        except Exception as e:
            lg.exception(e)
            raise e


class PredictionBatchesValidationConfig:
    def __init__(self) -> None:
        try:
            # Predictions Artifacts dir
            self.prediction_artifacts_dir = os.path.join(
                os.getcwd(), "prediction_artifacts", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")

            # Good Prediction Batches dir
            self.good_data_dir = os.path.join(
                self.prediction_artifacts_dir, "good_prediction_batches")
            # Bad Prediction Batches dir
            self.bad_data_dir = os.path.join(
                self.prediction_artifacts_dir, "bad_prediction_batches")
            # Archived Prediction Batches dir
            self.archived_data_dir = os.path.join(
                self.prediction_artifacts_dir, "archived_prediction_batches")
            ...
        except Exception as e:
            lg.exception(e)
