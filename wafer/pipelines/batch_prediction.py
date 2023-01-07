import os
from wafer.logger import lg
from wafer.CONFIG import ModelRegistryConfig
from wafer.entities.config import PredictionBatchesValidationConfig, DataSourceConfig
from wafer.components.data_validation import DataValidation
from wafer.utils.file_ops import BasicUtils
from datetime import datetime
from dataclasses import dataclass

PREDICTION_DIR = "predictions"


@dataclass
class BatchPredictionPipeline:
    """Shall be used for triggering the prediction pipeline.

    Args:
        prediction_batch_files_dir (str): Dir of prediction batches for which predictions gotta be made.
    """
    lg.info("Prediction Pipeline commences now..")
    lg.info(
        f"Entered the {os.path.basename(__file__)[:-3]}.BatchPredictionPipeline")

    prediction_batches_dir: str = DataSourceConfig().prediction_batches_dir
    prediction_file_name: str = "predictions"
    model_registry_config = ModelRegistryConfig()

    def get_predicition_file_path(self, prediction_file_name: str) -> str:
        """Returns the file path where the Predictions file is to be stored.
        
        Raises:
            e: Throws exception should any error or exception pops up while execution of this method.

        Returns:
            str: Path where prepared prediction file's gotta be stored.
        """
        try:
            # Configure Prediction dir (if not already there) followed by a `datetimestamp` named subdir
            prediction_dirs = os.path.join(PREDICTION_DIR, f"__{datetime.now().strftime('%m%d%Y__%H%M%S')}")
            # Configure Prediction file path            
            prediction_file_path = os.path.join(prediction_dirs, prediction_file_name+".csv")
            ...
        except Exception as e:
            lg.exception(e)
            raise e
        else:
            return prediction_file_path

    def commence(self):
        try:
            ############################## Validate Prediction Batches ##############################
            # and move them to `Bad Prediction Bacthes` and `Good Prediciton Bacthes` accordingly
            lg.info("VALIDATING PREDICTION BATCHES...")
            validate_prediction_batches = DataValidation(
                data_validation_config=PredictionBatchesValidationConfig(), 
                schema_path=DataSourceConfig().prediction_schema, schema_desc="prediction schema")
            validate_prediction_batches.initiate()

            ############################## Ingest into Database #####################################

            ########################### Preprocess Prediction Batches ###############################

            ################## Traverse through each Cluster and Make predictions ###################
            ...
        except Exception as e:
            lg.exception(e)
            raise e 


if __name__ == "__main__":
    prediction_pipeline = BatchPredictionPipeline()
    prediction_pipeline.commence()