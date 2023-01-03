import os
import argparse
import numpy as np
from wafer.logger import lg
from wafer.entities.artifact import DataPreparationArtifact, ModelTrainingArtifact
from wafer.entities.config import ModelTrainingConfig
from wafer.components.model_training.model_selection import BestModelSelection
from wafer.utils.file_ops import BasicUtils
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class ModelTraining:
    """Shall be used for training the shortlisted models, finetuning them and apparently returning configurations 
    of the built (and finetuned) models and their peformance measures.

    Args:    
        data_prep_artifact (DataPreparationArtifact): Takes in a `DataPreparationArtifact` object to have access 
        to all relevant configs of Data Preparation stage.
    """
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.ModelTraining" class')

    data_prep_artifact: DataPreparationArtifact
    model_training_config = ModelTrainingConfig()

    def initiate(self) -> ModelTrainingArtifact:
        try:
            lg.info(f"\n{'='*27} MODEL TRAINING {'='*40}")

            ############################### Fetch the Feature Store set #########################################
            lg.info('fetching the transformed "feature store" set..')
            wafers = BasicUtils.load_numpy_array(
                file_path=self.data_prep_artifact.transformed_feature_store_file_path, desc="feature store")
            lg.info('transformed "feature store" set fetched successfully..')
            lg.info(f'Shape of the "feature store" set: {wafers.shape}')

            ########################### Select and Train models based on clusters ##############################
            # Configure unique clusters
            n_clusters = np.unique(wafers[:, -2]).astype(int)

            # Traverse through each cluster and find a best model for it
            for i in n_clusters:
                lg.info(f"\n{'*'*27} CLUSTER {i} {'*'*40}")

                ############################### Filter Cluster data ############################################
                # filter cluster data
                lg.info(f'filtering "Cluster {i}" instances..')
                wafers_clus = wafers[wafers[:, -2] == i]

                ################### Configure Features and Labels for Cluster-filtered instances ###############
                X, y = np.delete(
                    wafers_clus, [-2, -1], axis=1), wafers_clus[:, -1]

                ################################ Training-Test Split ###########################################
                lg.info(f'Performing Training-Test split on "Cluster {i}"..')
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=1/3, random_state=42)
                lg.info("..training-test split done successfully!")

                ######################### Fetch best Model for given Cluster ###################################
                lg.info(
                    f'QUEST FOR FINDING THE BEST MODEL FOR "Cluster {i}" BEGINS NOW..')
                model_selection = BestModelSelection(
                    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, cv_for_eval=self.model_training_config.cv_for_eval,
                    cv_for_hypertuning=self.model_training_config.cv_for_hypertuning)
                best_mod_name, best_mod = model_selection.get_best_model()
                lg.info(
                    f'BEST MODEL FOR "Cluster {i}" TURNS OUTTA BE: "{best_mod_name}"')

                ########################## Save best Model for given Cluster ###################################
                lg.info(
                    f'Saving the best model "{best_mod_name}" built for "Cluster {i}"..')
                BasicUtils.save_cluster_based_model(
                    model=best_mod, model_name=best_mod_name, model_dir=self.model_training_config.model_training_dir, cluster=i)
                lg.info(f'..best model "{best_mod_name}" saved successfully!')

            ########################### Prepare the Model Training Artifact ####################################

            ...
        except Exception as e:
            lg.exception(e)
            raise e


if __name__ == "__main__":
    data_prep_artifact = DataPreparationArtifact(
        preprocessor_path=r'artifacts\01012023__190435\data_preparation\preprocessor\preprocessor.pkl',
        clusterer_path=r'artifacts\01012023__190435\data_preparation\clusterer\clusterer.pkl',
        transformed_feature_store_file_path=r'artifacts\01012023__190435\data_preparation\preprocessed\wafers.npz')

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_prep_artifact", default=data_prep_artifact)
    parsed_args = parser.parse_args()
    model_training = ModelTraining(
        data_prep_artifact=parsed_args.data_prep_artifact)
    model_training.initiate()
