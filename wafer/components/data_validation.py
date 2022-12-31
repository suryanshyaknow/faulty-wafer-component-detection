import os
import shutil
import pandas as pd
import re
from wafer.logger import lg
from wafer.entities.config import DataValidationConfig, DataSourceConfig
from wafer.entities.artifact import DataValidationArtifact
from wafer.utils.file_ops import BasicUtils
from typing import List
from dataclasses import dataclass


@dataclass
class DataValidation:
    """Gotta be used for validating the data before dumping it all into the desired database. First and foremost,
    the data shall be validated on basis of their file names followed by validation of raw data based on multiple
    constraints. In each stage, if data is validated, the file is moved to `Good Raw Data` dir and in the 
    `Bad Raw Data` dir for the other way around. At last, all the bad (rejected) data is sent over to `Archived Data` 
    dir. 
    """
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.DataValidation" class')

    data_source_config = DataSourceConfig()
    data_validation_config = DataValidationConfig()
    good_data_dir = data_validation_config.good_data_dir
    bad_data_dir = data_validation_config.bad_data_dir
    archived_data_dir = data_validation_config.archived_data_dir

    def validate_raw_files(self, expected_length_of_date_stamp: int, expected_length_of_time_stamp: int) -> None:
        """Validates the raw training files names by referencing a regular expression composed in regard to the 
        MDM (Master Data Management) by performing checks for expected date and time stamps' lengths and moves the
        validated files to `GoodRawData` dir and rejected ones to `BadRawData` dir.

        Args:
            expected_length_of_date_stamp (int): Expected length of datestamp in the file name in regard to MDM.
            expected_length_of_time_stamp (int): Expected length of timestamp in the file name in regard to MDM.

        Raises:
            e: Throws exception if any error pops up while validating raw files.
        """
        try:
            ############################ Configure Good/Bad Raw Data dirs #######################################
            os.makedirs(self.good_data_dir, exist_ok=True)
            os.makedirs(self.bad_data_dir, exist_ok=True)

            ###################### define Regex reference for validating files names ############################
            regex = "['wafer']+['\_'']+[\d_]+[\d]+\.csv"

            ######################### Validate Files Names using Regex reference ################################
            # and move them accordingly to GoodRawData and BadRawData dirs
            for file_name in os.listdir(self.data_source_config.raw_data_dir):
                if re.match(regex, file_name):
                    split_at_dot = re.split('.csv', file_name)[0]
                    date_stamp = re.split('_', split_at_dot)[1]
                    time_stamp = re.split('_', split_at_dot)[2]
                    if len(date_stamp) == expected_length_of_date_stamp:
                        if len(time_stamp) == expected_length_of_time_stamp:
                            lg.info(
                                f'"{file_name}" validated, moving to the `Good Raw Data` dir"..')
                            shutil.copy(os.path.join(
                                self.data_source_config.raw_data_dir, file_name), self.good_data_dir)
                        else:
                            lg.warning(
                                f'"{file_name}": "Timestamp" couldn\'t match to the expected, file rejected!')
                            shutil.copy(os.path.join(
                                self.data_source_config.raw_data_dir, file_name), self.bad_data_dir)
                    else:
                        lg.warning(
                            f'"{file_name}": "Datestamp" couldn\'t match to the expected, file rejected!')
                        shutil.copy(os.path.join(
                            self.data_source_config.raw_data_dir, file_name), self.bad_data_dir)
                else:
                    lg.warning(
                        f'"{file_name}": Invalid filename, file rejected!')
                    shutil.copy(os.path.join(
                        self.data_source_config.raw_data_dir, file_name), self.bad_data_dir)
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    def validate_raw_data(self, expected_number_of_columns: int, expected_names_of_columns: List) -> None:
        """Validates raw data in the raw training files by performing checks for expected number of columns and
        entries in those very columns, taken from the MDM (Master Data Management). And moves the validated files 
        to `GoodRawData` dir and rejected ones to `BadRawData` dir.

        Args:
            expected_number_of_columns (int): Expected number of columns for validating a file.

        Raises:
            e: Throws exception if any error pops up while validating raw data.
        """
        try:
            ############################# Validate Number of Columns ############################################
            # and move them accordingly to GoodRawData and BadRawData dirs
            lg.info("Validating `number of columns` in raw training files..")
            for csv_file in os.listdir(self.good_data_dir):
                # reading each csv file as pandas dataframe
                df = pd.read_csv(os.path.join(self.good_data_dir, csv_file))
                if df.shape[1] != expected_number_of_columns:
                    lg.warning(
                        f"\"{csv_file}\": Number of Columns Mismatched, file rejected!")
                    shutil.move(os.path.join(self.good_data_dir,
                                csv_file), self.bad_data_dir)
                else:
                    lg.info(
                        f'"{csv_file}" validated, staying in `Good Raw Data` dir..')
            lg.info("Check for expected number of columns completed with success!")

            ############################# Validate Columns' Names ###############################################
            # and move them accordingly to GoodRawData and BadRawData dirs
            lg.info("Validating `names of columns` in raw training files..")
            for csv_file in os.listdir(self.good_data_dir):
                # reading each csv file as pandas dataframe
                df = pd.read_csv(os.path.join(self.good_data_dir, csv_file))
                # rename the feature "Unnamed : 0" to "Wafer" of GoodRawData file
                df.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                current_cols_names = list(df.columns).sort()
                if current_cols_names != expected_names_of_columns.sort():
                    lg.warning(
                        f'"{csv_file}": Names of Columns Mismatched, file rejected!"')
                    shutil.move(os.path.join(self.good_data_dir,
                                csv_file), self.bad_data_dir)
                else:
                    lg.info(
                        f'"{csv_file}" validated, staying in `Good Raw Data` dir..')
            lg.info("Check for expected names of columns completed with success!")

            #################### See whether any Columns have all entries missing ###############################
            # and move them accordingly to GoodRawData and BadRawData dirs
            lg.info("Checking whether any of the columns has zero entries..")
            for csv_file in os.listdir(self.good_data_dir):
                lg.info(f"validating entries in {csv_file}'s columns..")
                # reading each csv file as pandas dataframe
                df = pd.read_csv(os.path.join(self.good_data_dir, csv_file))
                for col in df.columns:
                    # implies a column has zero entries
                    if (len(df[col]) - df[col].count() == len(df[col])):
                        lg.warning(
                            f'"{col}" of "{csv_file}" has zero entries, file rejected!')
                        shutil.move(os.path.join(self.good_data_dir,
                                    csv_file), self.bad_data_dir)
                        break
            lg.info("Check for columns' length completed with success!")
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    def initiate(self) -> DataValidationArtifact:
        """Initiates the Data Validation stage by performing relevant checks.

        Raises:
            e: Throws exception if any error pops up while in the Data Validation process. 

        Returns:
            DataValidationArtifact: Contains `Good Raw Data` and `Archived Data` dirs configurations.
        """
        try:
            lg.info(f"\n{'='*27} DATA VALIDATION {'='*40}")

            ########################## Create GoodRawData and BadRawData dirs ##################################
            lg.info(
                "Creating the respective `Good Raw Data`, `Bad Raw Data` and `Archived Data` dirs..")
            os.makedirs(self.good_data_dir, exist_ok=True)
            os.makedirs(self.bad_data_dir, exist_ok=True)
            os.makedirs(self.archived_data_dir, exist_ok=True)
            lg.info("..said dirs created successfully!")

            ######################### Fetch the relevants from the Training Schema #############################
            training_schema = BasicUtils.read_json_file(
                file_path=self.data_source_config.training_schema, file_desc="Training Schema")
            expected_len_of_datestamp = training_schema["LengthOfDateStampInFile"]
            expected_len_of_timestamp = training_schema["LengthOfTimeStampInFile"]
            expected_number_of_columns = training_schema["NumberofColumns"]
            expected_names_of_columns = list(training_schema["ColName"].keys())

            ################################### Validate Files Names ###########################################
            lg.info("Validating Raw Data Files' names..")
            self.validate_raw_files(expected_length_of_date_stamp=expected_len_of_datestamp,
                                    expected_length_of_time_stamp=expected_len_of_timestamp)
            lg.info("Raw files validated successfully!")

            ################################### Validate Raw Data ##############################################
            lg.info("Validating Raw data itself..")
            self.validate_raw_data(
                expected_number_of_columns=expected_number_of_columns,
                expected_names_of_columns=expected_names_of_columns)
            lg.info("Raw data validated successfully!")

            ########################## Move data from BadRawData dir to Archived dir ###########################
            lg.info(
                "moving all rejected data from `Bad Raw Data` dir to `Archived Data` dir..")
            # checking if BadRawData dir even exists
            if os.path.isdir(self.bad_data_dir):
                for rejected_file in os.listdir(self.bad_data_dir):
                    if rejected_file not in self.archived_data_dir:
                        shutil.move(os.path.join(self.bad_data_dir,
                                    rejected_file), self.archived_data_dir)
                        lg.info(
                            f"rejected file \"{rejected_file}\" moved to `Archived Data` dir with success!")
            lg.info("All rejected files moved to `Archived Data` dir successfully!")

            ############################# Delete BadRawData dir ################################################
            lg.info("Ultimatelty deleting the `Bad Raw Data` dir..")
            shutil.rmtree(self.bad_data_dir)
            lg.info("..said dir deleted successfully!")

            ############################# Prepare the Data Validation Artifact #################################
            data_validation_artifact = DataValidationArtifact(
                good_data_dir=self.good_data_dir, archived_data_dir=self.archived_data_dir)
            lg.info(f"Data Validation Artifact: {data_validation_artifact}")

            lg.info("DATA VALIDATION completed!")

            return data_validation_artifact
            ...
        except Exception as e:
            lg.exception(e)
            raise e
