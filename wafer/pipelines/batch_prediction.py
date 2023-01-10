import os
import json
import pandas as pd
import numpy as np
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
PREDICTION_OUTPUT_FILE = "predictions"


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

    def get_prediction_file_path(self, prediction_file_name: str = PREDICTION_OUTPUT_FILE) -> str:
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

    def commence(self) -> str:
        """Commences the Prediction Pipeline starting off with validating prediciton batches followed by 
        ingestion of those batches into MongoDB, preprocessing, recording predictions made my cluster based 
        models and finally readying the `predictions output file` and apparently returns the path of that
        readied predictions file.

        Raises:
            e: Raises exception should any sort of exception/error pops up while execution of the prediction 
            pipeline flow.

        Returns:
            str: Path of the readied predictions output file.
        """
        try:
            ############################# PREDICTION BATCHES VALIDATION #############################
            # and move them to `Bad Prediction Bacthes` and `Good Prediction Bacthes` accordingly
            lg.info(f'VALIDATING PREDICTION BATCHES...')
            validate_prediction_batches = DataValidation(
                data_validation_config=PredictionBatchesValidationConfig(), files_dir=DataSourceConfig().prediction_batches_dir,
                schema_path=DataSourceConfig().prediction_schema, schema_desc="prediction schema")
            validation_artifacts = validate_prediction_batches.initiate()

            ############################# PREDICTION BATCHES INGESTION ##############################
            lg.info(f'\n{"="*25} PREDICTION BATCHES INGESTION {"="*30}')
            lg.info("setting up MongoDB credentials and operations..")
            db_ops = MongoDBOperations(
                connection_string=self.database_config.mongodb_url,
                database_name=self.database_config.database_name,
                collection_name=self.database_config.prediction_collection)
            lg.info(f"setup done with success!")

            # ************************** Dump Validated Data into MongoDB ***************************
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
            lg.info(
                "Now, dumping all fetched records from all validated files into MongoDB database..")
            db_ops.dumpData(records=all_records,
                            data_desc="validated prediction-batches")
            lg.info(
                f'..successfully dumped all data from "Good Data" dir into database {self.database_config.database_name}!')

            ########################### PREDICTION BATCHES PREPARATION ##############################
            lg.info(f'\n{"="*25} PREDICTION BATCHES PREPARATION {"="*30}')
            lg.info('Getting "Prediction Batches Data" from the MongoDB database..')
            PRED_BATCH = db_ops.getDataAsDataFrame()
            lg.info("loaded all prediction data from MonogDB database successfully!")

            # ************************* Preprocess Prediction Instances *****************************
            # Load Preprocessor from Model Registry
            preprocessor = BasicUtils.load_object(
                file_path=self.model_registry_config.get_latest_preprocessor_path(), obj_desc="preprocessor")
            # Select only the features that were used in the Training
            lg.info(
                'Keeping only the features in the "prediction data" that were used in the training..')
            input_feats = preprocessor.feature_names_in_
            wafer_names = PRED_BATCH.iloc[:, 0]
            PRED_BATCH = PRED_BATCH[input_feats]
            lg.info('..said features kept successfully!')
            # Preprocess the Prediction Instances
            lg.info("Preprocessing the prediction instances..")
            PRED_BATCH_TRANS = preprocessor.transform(PRED_BATCH)
            lg.info("..preprocessed the prediction instances successfully!")

            # ************************** Cluster Prediction Instances *******************************
            # Load Clusterer from Model Registry
            clusterer = BasicUtils.load_object(
                file_path=self.model_registry_config.get_latest_clusterer_path(), obj_desc="clusterer")
            # Cluster the Prediction Instances
            lg.info("Clustering the prediction instances..")
            y_clus = clusterer.predict(PRED_BATCH_TRANS)

            # Configure "Wafer feature" and "Cluster labels" and to the prepared array
            PRED_BATCH_CLUS = np.c_[wafer_names, PRED_BATCH_TRANS, y_clus]
            lg.info("..clustered the prediction instances successfully!")

            ################## Traverse through each Cluster and Make predictions ###################
            lg.info(f'\n{"="*25} PREDICTION FILE PREPARATION {"="*30}')
            lg.info("NOW, TRAVERSING THROUGH EACH CLUSTER AND MAKING PREDICTIONS...")
            n_clusters = len(np.unique(y_clus))
            lg.info(f'Number of clusters predictions batches got clustered into: {n_clusters}')

            # Create a Dataframe to hold "wafer names" and "predictions"
            results = pd.DataFrame(columns=["Wafers", "Predictions"])
            
            for i in range(n_clusters):

                # Filter cluster instances
                lg.info(f'filtering "Cluster {i}" instances..')
                clustered_instances = PRED_BATCH_CLUS[PRED_BATCH_CLUS[:, -1] == i]
                lg.info(f'..filtered "Cluster {i}" instances successfully!')

                # Fetch Wafers names
                wafer_names_clus = clustered_instances[:, 0]

                # Separate Input features
                X_batch = np.delete(clustered_instances, [0, -1], axis=1)

                # Load the cluster based Model
                lg.info(f'loading the cluster based model for "Cluster {i}"..')
                mod_name, mod = BasicUtils.load_cluster_based_model(
                    model_dir=self.model_registry_config.get_latest_models_dir(), cluster=i)
                lg.info(
                    f'..loaded the model "{mod_name}" built and trained for "Cluster {i}" with success!')

                # Make predictions and Record them
                lg.info(
                    f'Making predictions on "Cluster {i}" of the prediction data..')
                y_preds = mod.predict(X_batch)

                # Merge these predictions to the "Results" dataframe
                cluster_results = pd.DataFrame(np.c_[wafer_names_clus, y_preds], columns=[
                                               "Wafers", "Predictions"])
                results = results.append(cluster_results)
                lg.info("..recorded 'em predictions successfully!")

            # Prepare the "Predictions" file
            prediction_file_path = self.get_prediction_file_path()
            BasicUtils.save_dataframe_as_csv(
                file_path=prediction_file_path, df=results, desc="predictions")
            lg.info(f'Shape of the "Predictions Output file": {results.shape}')
            lg.info("Prediction Pipeline ran with success!")

            return prediction_file_path
            ...
        except Exception as e:
            lg.exception(e)
            raise e


if __name__ == "__main__":
    prediction_pipeline = BatchPredictionPipeline()
    prediction_pipeline.commence()
