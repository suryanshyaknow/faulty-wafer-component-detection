from wafer.logger import lg
from dataclasses import dataclass
import os
import argparse
from wafer.components.data_validation import DataValidation
from wafer.components.data_ingestion import DataIngestion
from wafer.components.data_preparation.data_preparation import DataPreparation
from wafer.components.model_training.model_training import ModelTraining
from wafer.components.model_evaluation import ModelEvaluation
from wafer.components.model_pushing import ModelPushing


@dataclass
class TrainingPipeline:
    """Shall be used for triggering the Training pipeline."""
    lg.info("Training Pipeline begins now..")
    lg.info(f"Entered the {os.path.basename(__file__)[:-3]}.TrainingPipeline")

    new_data: bool = False

    def begin(self):
        """Commences the training pipeline starting from Data Validation component followed by 
        Data Ingestion, Data Preparation, Model Training, Model Evaluation and at last, 
        Model Pushing.

        Raises:
            e: Raises exception should any sort of error pops up during the training pipeline 
            flow execution.
        """
        try:
            ######################### DATA VALIDATION ######################################
            data_validation = DataValidation()
            data_validation_artifact = data_validation.initiate()

            ######################### DATA INGESTION #######################################
            data_ingestion = DataIngestion(
                data_validation_artifact=data_validation_artifact, new_data=self.new_data)
            data_ingestion_artifact = data_ingestion.initiate()

            ######################### DATA PREPARATION #####################################
            data_prep = DataPreparation(
                data_ingestion_artifact=data_ingestion_artifact)
            data_prep_artifact = data_prep.initiate()

            ######################### MODEL TRAINING #######################################
            model_training = ModelTraining(
                data_prep_artifact=data_prep_artifact)
            model_training_artifact = model_training.initiate()

            ######################### MODEL EVALUATION #####################################
            model_eval = ModelEvaluation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_prep_artifact=data_prep_artifact,
                model_training_artifact=model_training_artifact)
            model_eval_artifact = model_eval.initiate()

            ######################### MODEL PUSHING ########################################
            model_pushing = ModelPushing(
                data_prep_artifact=data_prep_artifact,
                model_training_artifact=model_training_artifact)
            model_pushing_artifact = model_pushing.initiate()
            ...
        except Exception as e:
            lg.exception(e)
            raise e
        else:
            lg.info("Training Pipeline ran with success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_data", default=False)
    parsed_args = parser.parse_args()
    training_pipeline = TrainingPipeline(new_data=parsed_args.new_data)
    training_pipeline.begin()
