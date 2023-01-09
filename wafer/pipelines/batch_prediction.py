import os
import pandas as pd
import json
from wafer.logger import lg
from wafer.CONFIG import ModelRegistryConfig
from wafer.entities.config import PredictionBatchesValidationConfig, DataSourceConfig
from wafer.components.data_validation import DataValidation
from wafer.CONFIG import DatabaseConfig
from wafer.utils.db_ops import MongoDBOperations
from wafer.utils.file_ops import BasicUtils
from datetime import datetime
from dataclasses import dataclass

PREDICTION_DIR = "predictions"
PREDICTION_OUTPUT_FILE = "predictions.csv"


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
    model_registry_config = ModelRegistryConfig()
    database_config = DatabaseConfig()

    def get_predicition_file_path(self, prediction_file_name: str) -> str:
        """Returns the file path where the Predictions file is to be stored.

        Raises:
            e: Throws exception should any error or exception pops up while execution of this method.

        Returns:
            str: Path where prepared prediction file's gotta be stored.
        """
        try:
            # Configure Prediction dir (if not already there) followed by a `datetimestamp` named subdir
            prediction_dirs = os.path.join(
                PREDICTION_DIR, f"__{datetime.now().strftime('%m%d%Y__%H%M%S')}")
            # Configure Prediction file path
            prediction_file_path = os.path.join(
                prediction_dirs, prediction_file_name+".csv")
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
            lg.info(f'VALIDATING PREDICITON BATCHES...')
            validate_prediction_batches = DataValidation(
                data_validation_config=PredictionBatchesValidationConfig(), files_dir=DataSourceConfig().prediction_batches_dir,
                schema_path=DataSourceConfig().prediction_schema, schema_desc="prediction schema")
            validation_artifacts = validate_prediction_batches.initiate()

            ############################## Ingest into Database #####################################
            lg.info(f'\n{"="*25} PREDICTION BATCHES INGESTION {"="*30}')
            lg.info("setting up MongoDB credentials and operations..")
            db_ops = MongoDBOperations(
                connection_string=self.database_config.mongodb_url,
                database_name=self.database_config.database_name,
                collection_name=self.database_config.prediction_collection)
            lg.info(f"setup done with success!")

            # Dump validated data into database
            lg.info(
                f'Before ingesting validated data into MongoDB\'s "{self.database_config.prediction_collection}" collection, stripping it off of all records..')
            db_ops.emptyCollection()
            lg.info(
                'Fetching validated data from "Good Data" dir..')
            all_records = []
            for csv_file in os.listdir(validation_artifacts.good_data_dir):
                # Read dataframe with na_values as "null"
                df = pd.read_csv(os.path.join(
                    validation_artifacts.good_data_dir, csv_file), na_values="null")
                # Rename the unnamed column to "Wafer"
                df.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                # Convert the df into "dumpable into MongoDB" format --json
                data = list(json.loads(df.T.to_json()).values())
                all_records += data
                lg.info(f'"{csv_file}"\'s data fetched successfully!')
            lg.info("..Records from all validated files fetched successfully!")
            lg.info("Now, dumping all fetched records from all validated files into MongoDB database..")
            db_ops.dumpData(records=all_records, data_desc="validated Prediction batches")
            lg.info(
                f'..successfully dumped all data from "Good Data" dir into database {self.database_config.database_name}!')

            ########################### Preprocess Prediction Batches ###############################

            ################## Traverse through each Cluster and Make predictions ###################
            ...
        except Exception as e:
            lg.exception(e)
            raise e


if __name__ == "__main__":
    prediction_pipeline = BatchPredictionPipeline()
    prediction_pipeline.commence()
