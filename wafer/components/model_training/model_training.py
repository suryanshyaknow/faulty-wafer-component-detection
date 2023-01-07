import os
import argparse
import numpy as np
from wafer.logger import lg
from wafer.entities.artifact import DataPreparationArtifact, ModelTrainingArtifact
from wafer.entities.config import ModelTrainingConfig
from wafer.components.model_training.model_selection import BestModelSelection
from wafer.utils.file_ops import BasicUtils
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
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
    repo = {}  # dict for Model's Perfromance Reports

    def initiate(self) -> ModelTrainingArtifact:
        try:
            lg.info(f"\n{'='*27} MODEL TRAINING {'='*40}")

            ############################### Load the "prepared" Training set ####################################
            lg.info('laoding the prepared training set..')
            wafers = BasicUtils.load_numpy_array(
                file_path=self.data_prep_artifact.prepared_training_set_path, desc="training")
            lg.info('prepared training set fetched successfully..')
            lg.info(f'Shape of the "training" set: {wafers.shape}')

            ########################### Select and Train models based on clusters ##############################

            # Traverse through each cluster and find a best model for it
            for i in range(self.data_prep_artifact.n_clusters):
                lg.info(f"\n{'*'*27} CLUSTER {i} {'*'*40}")

                cluster_key = f"Cluster {i}"
                self.repo[cluster_key] = {}
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
                self.repo[cluster_key]["model"] = best_mod_name

                ################################## Overfitting Check ###########################################
                if len(np.unique(y_train)) == 0:  # then can't use `roc_auc_score`, Will go ahead with `accuracy`
                    # Performance on Training set
                    train_acc = best_mod.score(X_train, y_train)
                    lg.info(
                        f'Best model\'s "{best_mod}" Accuracy on training set: {train_acc}')
                    self.repo[cluster_key]["training Accuracy"] = round(
                        train_acc, 3)

                    # Performance on Test set
                    test_acc = best_mod.score(X_test, y_test)
                    lg.info(
                        f'Best model\'s "{best_mod}" Accuracy on test set: {test_acc}')
                    self.repo[cluster_key]["test Accuracy"] = round(
                        test_acc, 3)

                    # Overfitting Check
                    lg.info("Performing check for Overfitting..")
                    diff = abs(train_acc - test_acc)
                    lg.info(
                        f"Overfitting Threshold: {self.model_training_config.overfit_thresh}")
                    lg.info(f"the difference we got : {diff}")
                    if diff > self.model_training_config.overfit_thresh:
                        lg.warning(
                            f"Since the difference between Accuracies on training and test set is greater than the overfitting thresh i.e {self.model_training_config.overfit_thresh}, the model definitely Overfits! ")
                    else:
                        lg.info("Model ain't Overfitting. We're good to go!")

                else:  # gonna go ahead with `roc_auc_score`
                    # Performance on Training set
                    y_train_pred = best_mod.predict(X_train)
                    train_auc = roc_auc_score(y_train, y_train_pred)
                    lg.info(
                        f'Best model\'s "{best_mod}" AUC on training set: {train_auc}')
                    self.repo[cluster_key]["train AUC"] = round(train_auc, 3)

                    # Performance on Test set
                    y_test_pred = best_mod.predict(X_test)
                    test_auc = roc_auc_score(y_test, y_test_pred)
                    lg.info(
                        f'Best model\'s "{best_mod}" AUC on test set: {test_auc}')
                    self.repo[cluster_key]["test AUC"] = round(test_auc, 3)

                    # Overfitting Check
                    lg.info("Performing check for Overfitting..")
                    diff = abs(train_auc - test_auc)
                    lg.info(
                        f"Overfitting Threshold: {self.model_training_config.overfit_thresh}")
                    lg.info(f"the difference we got : {diff}")
                    if diff > self.model_training_config.overfit_thresh:
                        lg.warning(
                            f"Since the difference between AUCs on training and test set is greater than the overfitting thresh i.e {self.model_training_config.overfit_thresh}, the model definitely Overfits! ")
                    else:
                        lg.info("Model ain't Overfitting. We're good to go!")

                ########################## Save best Model for given Cluster ###################################
                lg.info(
                    f'Saving the best model "{best_mod_name}" built for "Cluster {i}"..')
                BasicUtils.save_cluster_based_model(
                    model=best_mod, model_name=best_mod_name, model_dir=self.model_training_config.cluster_based_models_dir, cluster=i)
                lg.info(f'..best model "{best_mod_name}" saved successfully!')

                ############################# Save Models Performance Report ###################################
                lg.info("Readying the Models Performance Report..")
                BasicUtils.write_json_file(
                    file_path=self.model_training_config.performance_report_path, data=self.repo, file_desc="Models Performance Report")
                lg.info("Models Performance Report prepared successfully!")

            ########################### Prepare the Model Training Artifact ####################################
            model_training_artifact = ModelTrainingArtifact(
                cluster_based_models_dir=self.model_training_config.cluster_based_models_dir,
                performance_report_path=self.model_training_config.performance_report_path)
            lg.info(f"Model Training Artifact: {model_training_artifact}")
            lg.info("Model Training completed!")

            ...
        except Exception as e:
            lg.exception(e)
            raise e


if __name__ == "__main__":
    data_prep_artifact = DataPreparationArtifact(
        preprocessor_path=r'artifacts\01072023__012253\data_preparation\preprocessor\preprocessor.pkl',
        clusterer_path=r'artifacts\01072023__012253\data_preparation\clusterer\clusterer.pkl',
        prepared_training_set_path=r'artifacts\01072023__012253\data_preparation\preprocessed\train.npz',
        n_clusters=3)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_prep_artifact", default=data_prep_artifact)
    parsed_args = parser.parse_args()
    model_training = ModelTraining(
        data_prep_artifact=parsed_args.data_prep_artifact)
    model_training.initiate()
