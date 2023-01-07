import numpy as np
import os
from wafer.logger import lg
from wafer.utils.file_ops import BasicUtils
from wafer.entities.config import DataPreparationConfig, BaseConfig
from wafer.entities.artifact import DataIngestionArtifact, DataPreparationArtifact
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from wafer.components.data_preparation.clustering import ClusterDataInstances
from dataclasses import dataclass


@dataclass
class DataPreparation:
    """Shall be used for preparing the training data, before feeding into ML algorithms for training."""
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.DataTransformation" class')

    data_ingestion_artifact: DataIngestionArtifact
    data_prep_config = DataPreparationConfig()
    target = BaseConfig().target

    @classmethod
    def get_preprocessor(cls) -> Pipeline:
        """Returns customizable pipeline for numerical attributes of the said dataset.

        Raises:
            e: Throws exception if any error pops up while building or returning the preprocessor.

        Returns:
            Pipeline: Customizable Pipeline for the numerical features of the said dataset. 
        """
        try:
            ########################## Pipeline for Numerical Atts ############################################
            preprocessing_pipeline = Pipeline(
                steps=[('KNN IMputer', KNNImputer(n_neighbors=3))])

            return preprocessing_pipeline
            ...
        except Exception as e:
            lg.info(e)
            raise e

    def initiate(self) -> DataPreparationArtifact:
        """Initiates the Data Preparation stage of the training pipeline.

        Raises:
            e: Raises exception should any pops up while preparing the data.

        Returns:
            DataPreparationArtifact: Contains configurations of all the relevant artifacts that shall be made while
            preparing the data.
        """
        try:
            lg.info(f"\n{'='*27} DATA PREPARATION {'='*40}")

            ################################# Fetch the Training set ###########################################
            lg.info('fetching the "training" dataset for data preparation..')
            wafers = BasicUtils.load_csv_as_dataframe(
                self.data_ingestion_artifact.training_set_path, desc="training")
            lg.info("..said dataset fetched successfully!")

            ################################ Drop Redundant Features ###########################################
            # `Wafer` feature to be dropped
            cols_to_drop = ["Wafer"]
            # features with missing ratio more than 0.7
            cols_with_missing_ratio_70 = BasicUtils.get_columns_with_certain_missing_thresh(
                df=wafers, missing_thresh=0.7, desc="feature store")
            # features with "0 Standard Deviation"
            cols_with_zero_std = BasicUtils.get_columns_with_zero_std_dev(
                df=wafers, desc="feature store")

            cols_to_drop = cols_to_drop + cols_with_missing_ratio_70 + cols_with_zero_std
            # drop these Redundant features
            wafers = BasicUtils.drop_columns(
                wafers, cols_to_drop=cols_to_drop, desc="feature store")

            lg.info(f"Size of the dataset after dropping redundant features: {wafers.size}")

            ########################## Separate the Features and Labels out ####################################
            X, y = BasicUtils.get_features_and_labels(
                df=wafers, target=[self.target], desc="feature store")

            ############################## Transformation: Imputation ##########################################
            # fetch the transformer and fit to the training features
            lg.info("fetching the preprocessor (Imputer)..")
            preprocessor = DataPreparation.get_preprocessor()
            lg.info("..preprocessor fetched successfully!")
            lg.info("transforming the feature store dataset..")
            X_transformed = preprocessor.fit_transform(X)
            lg.info("..transformed the feature store dataset successfully!")

            # Saving the Preprocessor
            BasicUtils.save_object(
                file_path=self.data_prep_config.preprocessor_path,
                obj=preprocessor,
                obj_desc="preprocessor")

            ################################# Cluster Data Instances ###########################################
            cluster_train_data = ClusterDataInstances(
                X=X_transformed, desc="training", data_prep_config=self.data_prep_config)
            clusterer, X_clus = cluster_train_data.create_clusters()

            # Save the Clusterer
            lg.info("saving the Clusterer..")
            BasicUtils.save_object(
                file_path=self.data_prep_config.clusterer_path, obj=clusterer, obj_desc="KMeans Clusterer")
            lg.info("..said Clusterer saved with success!")

            ######################### Configure and Save the Feature Store array ###############################
            wafers_arr = np.c_[X_clus, y]
            BasicUtils.save_numpy_array(
                file_path=self.data_prep_config.prepared_training_set_path,
                arr=wafers_arr,
                desc="(transformed) feature store")

            ########################## Prepare the Data Preparation Artifact ###################################
            n_clusters = len(np.unique(wafers_arr[:, -2]))  # Number of Optimal Clusters made
            data_prep_artifact = DataPreparationArtifact(
                preprocessor_path=self.data_prep_config.preprocessor_path,
                clusterer_path=self.data_prep_config.clusterer_path,
                prepared_training_set_path=self.data_prep_config.prepared_training_set_path,
                n_clusters=n_clusters)
            lg.info(f"Data Preparation Artifact: {data_prep_artifact}")
            lg.info("Data Preparation completed!")

            return data_prep_artifact
            ...
        except Exception as e:
            lg.exception(e)
            raise e
