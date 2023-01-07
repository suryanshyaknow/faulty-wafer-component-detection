import os
import argparse
import numpy as np
from wafer.logger import lg
from wafer.CONFIG import ModelRegistryConfig
from wafer.entities import config, artifact
from wafer.utils.file_ops import BasicUtils
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass


@dataclass
class ModelEvaluation:
    """Shall be used to trigger Model Evaluation stage in which it's to be determined that whether the older 
    models are to be replaced in the production grade pipeline (to say, if the newer models are performing 
    better than the currently deployed ones).

    Args:
        data_ingestion_artifact (artifact.DataIngestionArtifact): Takes in a `DataIngestionArtifact` object as 
        a prerequisite for Model Evaluation stage.
        data_transformation_artifact (artifact.DataTransformationArtifact): Takes in a 
        `DataTransformationArtifact` object as a prerequisite for Model Evaluation stage.
        model_training_artifact (artifact.ModelTrainingArtifact): Takes in a `ModelTrainingArtifact` object as 
        a prerequisite for Model Evaluation stage.
    """
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.ModelEvaluation" class')

    data_ingestion_artifact: artifact.DataIngestionArtifact
    data_prep_artifact: artifact.DataPreparationArtifact
    model_training_artifact: artifact.ModelTrainingArtifact

    model_eval_config = config.ModelEvaluationConfig()
    model_registry_config = ModelRegistryConfig()
    target = config.BaseConfig().target

    def initiate(self) -> artifact.ModelEvaluationArtifact:
        """Initiates the Model Evaluation stage of the training pipline in which it's determined that whether 
        the current delpoyed models are to be replaced in the production grade pipline by the latest models 
        and in turn returns the artifact config containing the decision `are_models_replaced`, along with 
        their `improved_accuracy`, if yes.

        Raises:
            e: Raises relevant exception should any sort of error pops up during the execution of Model 
            Evaluation component.
        Returns:
            artifact.ModelEvaluationArtifact: Configuration object containing the decision `are_models_replaced` 
            along with their `improved_scores`.
        """
        try:
            lg.info(f"\n{'='*27} MODEL EVALUATION {'='*40}")
            latest_dir = self.model_registry_config.get_latest_dir_path()

            ################### Compare the latest models to the old ones if there are ones ###################
            # If there're no old models, then the configure the current ones
            if latest_dir is None:
                lg.info(
                    "There are no old models present to get compared to the latest ones!")
                model_eval_artifact = artifact.ModelEvaluationArtifact(
                    are_models_replaced=True, improved_scores=None)
                lg.info(f"Model Evaluation Artifact: {model_eval_artifact}")
            else:
                lg.info(
                    'QUEST FOR KEEPING THE BETTER MODELS BETWEEN "JUST BUILT" MODELS AND "CURRENTLY DEPLOYED" ONES BEGINS NOW...')

                ############################# Load Latest Artifacts (just built) #############################
                lg.info('loading the latest built verisons of artifacts...')
                # load latest Preprocessor
                latest_preprocessor = BasicUtils.load_object(
                    file_path=self.data_prep_artifact.preprocessor_path, obj_desc="latest Preprocessor")
                # load latest Clusterer
                latest_clusterer = BasicUtils.load_object(
                    file_path=self.data_prep_artifact.preprocessor_path, obj_desc="latest Clusterer")
                # grab latest Models dir
                latest_models_dir = self.model_training_artifact.cluster_based_models_dir
                lg.info("..latest built version of artifacts loaded successfully!")

                ###################### Load deployed Artifacts from the Model Registry ######################
                lg.info(
                    "loading the deployed version of artifacts from Model Registry...")
                # fetch older Preprocessor
                older_preprocessor = BasicUtils.load_object(
                    file_path=self.model_registry_config.get_latest_preprocessor_path(), obj_desc="older preprocessor")
                # fetch older Clusterer
                older_clusterer = BasicUtils.load_object(
                    file_path=self.model_registry_config.get_latest_clusterer_path(), obj_desc="older Clusterer")
                # grab older Models dir
                older_models_dir = self.model_registry_config.get_latest_models_dir()
                lg.info('..deployed version of artifacts loaded successfully!')

                ################################## Load the Test set ########################################
                lg.info("loading the test set to gauge models true performance...")
                wafers_test = BasicUtils.load_csv_as_dataframe(
                    file_path=self.data_ingestion_artifact.test_set_path, desc="Test")
                lg.info("..test set loaded succesfully!")
                lg.info(f"Shape of the Test set: {wafers_test.shape}")
                # Separate Test features and labels out
                X_test, y_test = BasicUtils.get_features_and_labels(
                    df=wafers_test, target=[self.target], desc="Test")

                ########################## Evaluate the latest Models' performance ##########################
                lg.info("EVALUATING THE LATEST BUILT VERSION OF MODELS...")
                # Drop redundant features (fetch only those that were used in Training)
                lg.info(
                    "keeping only the features in the test set that were used in the training...")
                input_feats = list(latest_preprocessor.feature_names_in_)
                lg.info("..said features fetched succesfully!")
                # Preprocess Test Instances
                X_test_trans = latest_preprocessor.transform(
                    X_test[input_feats])
                # Cluster Test Instances and Configure the prepared Test Array
                y_test_kmeans = latest_clusterer.predict(X_test_trans)
                wafers_test_prep = np.c_[X_test_trans, y_test_kmeans, y_test]
                # Traverse through each cluster and Gauge each Model's performance on test set
                latest_scores = []
                for i in range(self.data_prep_artifact.n_clusters):
                    lg.info(
                        f'Gauging the true performance of model on "Cluster {i}" of test set...')

                    # Filter Cluster Instances
                    lg.info(f'filtering "Cluster {i}" instances..')
                    wafers_clus = wafers_test_prep[wafers_test_prep[:, -2] == i]
                    lg.info(
                        f'Shape of "Cluster {i}" instances: {wafers_clus.shape}')

                    # Separate Features and Labels out
                    lg.info("Separating clustered features and labels out..")
                    X_test, y_test = np.delete(
                        wafers_clus, [-2, -1], axis=1), wafers_clus[:, -1]
                    lg.info(
                        "..separated clustered features and labels successfully!")

                    # Load Cluster based Model from Model Training component's artifact
                    _, mod = BasicUtils.load_cluster_based_model(
                        model_dir=latest_models_dir, cluster=i)

                    # Gauge Model's performance
                    if len(np.unique(y_test)) == 0:  # then can't use `roc_auc_score`, Will go ahead with `accuracy`
                        test_acc = mod.score(X_test, y_test)
                        lg.info(
                            f'Accuracy on "Cluster {i}" of test set: {test_acc}')
                        latest_scores.append(test_acc)
                    else:  # gonna go ahead with the `roc_auc_score`
                        y_test_pred = mod.predict(X_test)
                        test_auc = roc_auc_score(y_test, y_test_pred)
                        lg.info(
                            f'AUC score on "Cluster {i}" of test set: {test_auc}')
                        latest_scores.append(test_auc)

                lg.info(
                    f"latest Models performances on test set: {latest_scores}")

                ########################## Evaluate the older Models' performance ###########################
                # ************************ (Latest Models form Model Registry) ******************************
                lg.info("EVALUATING THE MODELS FROM THE MODEL REGISTRY...")
                # Drop redundant features (fetch only those that were used in Training)
                lg.info(
                    "keeping only the features in the test set that were used in the training...")
                input_feats = list(older_preprocessor.feature_names_in_)
                lg.info("..said features fetched succesfully!")
                # Preprocess Test Instances
                X_test_trans = older_preprocessor.transform(
                    X_test[input_feats])
                # Cluster Test Instances and Configure the prepared Test Array
                y_test_kmeans = older_clusterer.predict(X_test_trans)
                wafers_test_prep = np.c_[X_test_trans, y_test_kmeans, y_test]
                # Traverse through each cluster and Gauge each Model's performance on test set
                n_clusters = np.unique(y_test_kmeans)  # configured unique clusters
                model_reg_scores = []
                for i in n_clusters:
                    lg.info(
                        f'Gauging the true performance of model on "Cluster {i}" of test set...')

                    # Filter Cluster Instances
                    lg.info(f'filtering "Cluster {i}" instances..')
                    wafers_clus = wafers_test_prep[wafers_test_prep[:, -2] == i]
                    lg.info(
                        f'Shape of "Cluster {i}" instances: {wafers_clus.shape}')

                    # Separate Features and Labels out
                    lg.info("Separating clustered features and labels out..")
                    X_test, y_test = np.delete(
                        wafers_clus, [-2, -1], axis=1), wafers_clus[:, -1]
                    lg.info(
                        "..separated clustered features and labels successfully!")

                    # Load Cluster based Model from Model Registry
                    _, mod = BasicUtils.load_cluster_based_model(
                        model_dir=older_models_dir, cluster=i)

                    # Gauge Model's performance
                    if len(np.unique(y_test)) == 0:  # then can't use `roc_auc_score`, Will go ahead with `accuracy`
                        test_acc = mod.score(X_test, y_test)
                        lg.info(
                            f'Accuracy on "Cluster {i}" of test set: {test_acc}')
                        model_reg_scores.append(test_acc)
                    else:  # gonna go ahead with the `roc_auc_score`
                        y_test_pred = mod.predict(X_test)
                        test_auc = roc_auc_score(y_test, y_test_pred)
                        lg.info(
                            f'AUC score on "Cluster {i}" of test set: {test_auc}')
                        model_reg_scores.append(test_auc)

                lg.info(
                    f"Older (Model Registry's) Models performances on test set: {model_reg_scores}")

                ################ Comparison between the two versions to keep the better ones ################
                if (sum(latest_scores) - sum(model_reg_scores)) <= self.model_eval_config.replace_models_thresh:
                    lg.exception('The `Models from Model Registry` altogether performed better than the \
`Latest Built Models` and quite evidently they shan\'t be replaced!')
                    raise Exception('The `Models from Model Registry` altogether performed better than the \
`Latest Built Models` and quite evidently they shan\'t be replaced!')

                lg.info('the `Latest Built Models` performed better than the ones from the Model Registry \
and quite evidently they shall be replaced!')

                ###################### Prepare the Model Evaluation Artifact ################################
                improved_scores = [
                    abs(latest_scores[i] - model_reg_scores[i]) for i in range(n_clusters)]
                model_eval_artifact = artifact.ModelEvaluationArtifact(
                    are_models_replaced=True, improved_scores=improved_scores)
                lg.info(f"Model Evaluation Artifact: {model_eval_artifact}")

            lg.info("Model Evaluation completed!")
            ...
        except Exception as e:
            lg.exception(e)
            raise e


if __name__ == "__main__":

    data_ingestion_artifact = artifact.DataIngestionArtifact(
        feature_store_file_path=r'artifacts\01072023__012253\data_ingestion\feature_store\wafers.csv',
        training_set_path=r'artifacts\01072023__012253\data_ingestion\datasets\train.csv',
        test_set_path=r'artifacts\01072023__012253\data_ingestion\datasets\test.csv')

    data_prep_artifact = artifact.DataPreparationArtifact(
        preprocessor_path=r'artifacts\01072023__012253\data_preparation\preprocessor\preprocessor.pkl',
        clusterer_path=r'artifacts\01072023__012253\data_preparation\clusterer\clusterer.pkl',
        prepared_training_set_path=r'artifacts\01072023__012253\data_preparation\preprocessed\train.npz',
        n_clusters=3)

    model_training_artifact = artifact.ModelTrainingArtifact(
        cluster_based_models_dir=r'artifacts\01072023__012253\model_training\cluster_based_models',
        performance_report_path=r'artifacts\01072023__012253\model_training\models_performace_report\report.json')

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_ingestion_artifact", default=data_ingestion_artifact)
    parser.add_argument("--data_prep_artifact", default=data_prep_artifact)
    parser.add_argument("--model_training_artifact", default=model_training_artifact)
    parsed_args = parser.parse_args()
    model_training = ModelEvaluation(
        data_ingestion_artifact=parsed_args.data_ingestion_artifact,
        data_prep_artifact=parsed_args.data_prep_artifact,
        model_training_artifact=parsed_args.model_training_artifact)
    model_training.initiate()
