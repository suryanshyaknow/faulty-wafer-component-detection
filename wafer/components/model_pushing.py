import os
from wafer.logger import lg
from wafer.entities.config import ModelPushingConfig
from wafer.entities.artifact import ModelPushingArtifact, ModelTrainingArtifact, DataPreparationArtifact
from wafer.CONFIG import ModelRegistryConfig
from wafer.utils.file_ops import BasicUtils
from dataclasses import dataclass


@dataclass
class ModelPushing:
    """Shall be used to trigger Model Pushing stage in which latestly built models and corrosponding artifacts 
    are to be pushed into the Model Registry and saved as Model Pushing's artifacts as well.

    Args:
        data_transformation_artifact (DataTransformationArtifact): Takes in a `DataPreparationArtifact` object 
        for accessing the config of artifacts that were built during the Data Preparation stage.
        model_training_artifact (ModelTrainingArtifact): Takes in a `ModelTrainingArtifact` object for accessing 
        the config of the models that were built during the Model Training stage.
    """
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.ModelPushing" class')

    data_prep_artifact: DataPreparationArtifact
    model_training_artifact: ModelTrainingArtifact

    model_pushing_config = ModelPushingConfig()
    model_registry_config = ModelRegistryConfig()

    def initiate(self) -> ModelPushingArtifact:
        """Initiates the Model Pushing stage in which latestly built artifacts and models are gonna be pushed 
        into the model registry.

        Raises:
            e: Raises relevant exception ahould any sort of error pops up during the Model Pushing stage.

        Returns:
            ModelPushingArtifact: Configuraton object that contains configs of the latestly built models
            objects.
        """
        try:
            lg.info(f"\n{'='*27} MODEL PUSHING {'='*40}")

            ############################# Load Objects which are to be saved ###################################
            lg.info("LOADING THE MODELS AND ARTIFACTS THAT ARE TO BE PUSHED...")
            # Load Preprocessor from Data Preparation's artifacts
            preprocessor = BasicUtils.load_object(
                file_path=self.data_prep_artifact.preprocessor_path, obj_desc="preprocessor")

            # Load Clusterer from Data Preparation's artifacts
            clusterer = BasicUtils.load_object(
                file_path=self.data_prep_artifact.clusterer_path, obj_desc="clusterter")

            # load the Models built during the Model Training stage
            lg.info(
                "loading the cluster based models that were built and trained during the Model Training stage...")
            loaded_mods = []
            for i in range(self.data_prep_artifact.n_clusters):
                loaded_mods.append(BasicUtils.load_cluster_based_model(
                    model_dir=self.model_training_artifact.cluster_based_models_dir, cluster=i))
            lg.info(
                "...loaded all models built during Model Training stage successfully!")

            ############################## Save these Objects to Model Registry ################################
            lg.info(
                "DUMPING ALL THE LOADED MODELS AND ARTIFACTS INTO MODEL REGISTRY...")
            # Dump Preprocessor
            preprocessor_path = self.model_registry_config.save_latest_preprocessor_at()
            BasicUtils.save_object(
                file_path=preprocessor_path, obj=preprocessor, obj_desc="latest Preprocessor")
            # Dump Clusterer
            clusterer_path = self.model_registry_config.save_latest_clusterer_at()
            BasicUtils.save_object(
                file_path=clusterer_path, obj=clusterer, obj_desc="latest Clusterer")
            # Dump all Cluster based Models
            for cluster, i in enumerate(loaded_mods):
                BasicUtils.save_cluster_based_model(
                    model=i[1], model_name=i[0], model_dir=self.model_registry_config.get_latest_models_dir, cluster=cluster)
            lg.info(
                "...dumped all models and corrospondng artifacts into Model Registry successfully!")

            ######################### Save these Objects as Model Pushing's Artifacts ##########################
            lg.info(
                "SAVING ALL THE LOADED MODELS AND ARTIFACTS AS MODEL PUSHING ARTIFACTS...")
            # Save Preprocessor as Model Pushing Artifact

            # Save Clusterer as Model Pushing Arifact

            # Save all Cluster based Models as Model Pushing Artifacts

            ############################### Prepare the Model Pushing Artifact #################################

        except Exception as e:
            lg.exception(e)
            raise e
