import os
import pandas as pd
import json
import argparse
from wafer.logger import lg
from wafer.CONFIG import DatabaseConfig
from wafer.entities.config import DataIngestionConfig
from wafer.entities.artifact import DataIngestionArtifact, DataValidationArtifact
from wafer.utils.db_ops import MongoDBOperations
from wafer.utils.file_ops import BasicUtils
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestion:
    """Shall be used for ingesting the validated data from "Good Data" dir into MongoDB and 
    even extract and readies the consequent feature store file once the ingestion's been done.

    Args:
        new_data (bool): Whether there's any new data for dumping into the desired relational dB. 
        Defaults to False.
    """
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.DataIngestion" class')

    data_validation_artifact: DataValidationArtifact
    new_data: bool = False
    database_config = DatabaseConfig()
    data_ingestion_config = DataIngestionConfig()

    def initiate(self) -> DataIngestionArtifact:
        """Initiates the Data Ingestion stage of the training pipeline.

        Raises:
            e: Raises exception should any pops up while ingestion of data or extraction of same 
            after ingestion.

        Returns:
            DataIngestionArtifact: Contains configurations of all relevant artifacts that shall 
            be made during the Data Ingestion stage.
        """
        try:
            lg.info(f"\n{'='*27} DATA INGESTION {'='*40}")

            ######################## Setup MongoDB Credentials and Operations ##################################
            lg.info("setting up MongoDB credentials and operations..")
            db_ops = MongoDBOperations(
                connection_string=self.database_config.mongodb_url,
                database_name=self.database_config.database_name,
                collection_name=self.database_config.training_collection)
            lg.info(f"setup done with success!")

            if self.new_data:  # dump data to MongoDB only if there's new data

                ####################### Fetch GoodRawData and Dump into MongoDB ####################################
                lg.info('Fetching all validated data from "Good Data" dir..')
                good_data_dir = self.data_validation_artifact.good_data_dir
                all_records = []
                for csv_file in os.listdir(good_data_dir):
                    # read dataframe with na_values as "null"
                    df = pd.read_csv(os.path.join(
                        good_data_dir, csv_file), na_values="null")
                    # rename the unnamed column to "Wafer"
                    df.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                    # convert the df into "dumpable into MongoDB" format --json
                    data = list(json.loads(df.T.to_json()).values())
                    all_records += data
                    lg.info(f"\"{csv_file}\" data fetched successfully!")

                lg.info("..Records from all validated files fetched successfully!")
                lg.info("Now, dumping all fetched records from all validated files into MongoDB database..")
                db_ops.dumpData(records=all_records, data_desc="validated Training batches")    
                lg.info(
                    f'successfully dumped all data from "Good Data" dir into database {self.database_config.database_name}')

            ################## Read data from Database and Prepare "Feature Store" set #########################
            lg.info('readying the "feature store" set..')
            feature_store_df = db_ops.getDataAsDataFrame()
            lg.info(
                f'Shape of the "feature store" set: {feature_store_df.shape}')
            BasicUtils.save_dataframe_as_csv(
                file_path=self.data_ingestion_config.feature_store_file_path, df=feature_store_df, desc="feature store")
            lg.info('..prepared the "feature store" set successfully!')

            ############################### Perform Training-Test Split ########################################
            lg.info('Splitting the data into training and test subsets..')
            train, test = train_test_split(
                feature_store_df, test_size=self.data_ingestion_config.test_size, random_state=self.data_ingestion_config.random_state)
            lg.info("data split into test and training subsets successfully!")
            # Save the test and training sets to their respective dirs
            lg.info("Saving the test and training subsets to their respective dirs..")
            BasicUtils.save_dataframe_as_csv(
                file_path=self.data_ingestion_config.training_set_path, df=train, desc="training")
            BasicUtils.save_dataframe_as_csv(
                file_path=self.data_ingestion_config.test_set_path, df=test, desc="test")
            lg.info("test and training subsets saved succesfully!")

            ########################### Prepare the Data Ingestion Artifact ####################################
            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                training_set_path=self.data_ingestion_config.training_set_path, test_set_path=self.data_ingestion_config.test_set_path)
            lg.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            lg.info("DATA INGESTION completed!")

            return data_ingestion_artifact
            ...
        except Exception as e:
            lg.exception(e)
            raise e

if __name__ == "__main__":
    data_validation_artifact = DataValidationArtifact(
        good_data_dir=r'artifacts\01072023__012253\data_validation\good_raw_data',
        archived_data_dir=r'artifacts\01072023__012253\data_validation\archived_data')

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_validation_artifact", default=data_validation_artifact)
    parser.add_argument("--new_data", default=False)
    parsed_args = parser.parse_args()
    data_ingestion = DataIngestion(
        data_validation_artifact=parsed_args.data_validation_artifact, new_data=parsed_args.new_data)
    data_ingestion.initiate()
