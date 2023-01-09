import os
from wafer.logger import lg
from dotenv import load_dotenv
from typing import Optional
from wafer.entities.config import PREPROCESSOR, CLUSTERER
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Configuration class that shall be used to access any sort of config pertaining to the relational database 
    used in this project.
    """
    load_dotenv()
    mongodb_url = os.getenv("MONGO_DB_URL")
    database_name = "wafers"
    training_collection = "training-batches"
    prediction_collection = "prediction-batches"


@dataclass
class ModelRegistryConfig:
    """Shall be used to access any dir of the Model Registry.

    Note: Model Registry is a centralized dir for older models and relevant artifacts (built in past) and will 
    contain newer if it's determined that they perform better than the older ones, with their dirs sorted by 
    integers in an increasing manner.

    Args:
        model_registry (str, optional): Name of the Model Registry dir. Defaults to "saved_models".
        preprocessor (str, optional): Preprocessor's dir name inside the model registry. Defaults to 
        "preprocessor".
        model_dir (str, optional): Model dir's name inside the model registry. Defaults to "model".
    """

    model_registry: str = "saved_models"
    preprocessor_dir: str = "preprocessor"
    clusterer_dir: str = "clusterer"
    model_dir: str = "models"

    # Make the Model Registry dir
    os.makedirs(model_registry, exist_ok=True)

    def get_latest_dir_path(self) -> Optional[str]:
        """Returns path of the latest dir of Model Registry.

        Raises:
            e: Raises relevant exception should any sort of error pops up while returning the latest dir path.

        Returns:
            Optional[str]: Path of the latest dir of Model Registry.
        """
        try:
            dirs = os.listdir(self.model_registry)
            if len(dirs) == 0:
                lg.warning(
                    "As of now there are no such directories in the Model Registry!")
                return None

            # Typecasting dir names from str to int
            dirs = list(map(int, dirs))
            latest_dir = max(dirs)

            return os.path.join(self.model_registry, str(latest_dir))
            ...
        except Exception as e:
            lg.exception(e)
            raise e
    
    def get_latest_models_dir(self) -> str:
        """Returns the path of the latest models dir of the Model Registry.

        Raises:
            e: Raises relevant exception should any sort of error pops up while returning the latest model path.

        Returns:
            str: Latest Model dir of the Model Registry.
        """
        try:
            latest_dir = self.get_latest_dir_path()
            lg.info('Getting the "latest models dir" from the Model Registry..')
            if latest_dir is None:
                lg.exception(
                    "Even the dir doesn't exist and you are expecting a model, shame!")
                raise Exception(
                    "Even the dir doesn't exist and you are expecting a model, shame!")

            return os.path.join(latest_dir, self.model_dir)
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    def get_latest_preprocessor_path(self) -> str:
        """Returns the path of the `latest preprocessor` from the Model Registry.

        Raises:
            e: Raises relevant exception should any sort of error pops up while returning the latest preprocessor 
            path.

        Returns:
            str: Latest Preprocessor path of the Model Registry.
        """
        try:
            latest_dir = self.get_latest_dir_path()
            lg.info('Getting the "latest preprocessor path" from the Model Registry..')
            if latest_dir is None:
                lg.exception(
                    "Even the dir doesn't exist and you are expecting a preprocessor, shame!")

            return os.path.join(latest_dir, self.preprocessor_dir, PREPROCESSOR)
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    def get_latest_clusterer_path(self) -> str:
        """Returns the path of the `latest Clusterer` from the Model Registry.

        Raises:
            e: Raises relevant exception should any sort of error pops up while returning the latest clusterer 
            path.

        Returns:
            str: Latest Clusterer's path from the Model Registry.
        """
        try:
            latest_dir = self.get_latest_dir_path()
            lg.info('Getting the "latest Target Encoder path" from the Model Registry..')
            if latest_dir is None:
                lg.exception(
                    "Even the dir doesn't exist and you are expecting a clusterer, shame!")

            return os.path.join(latest_dir, self.clusterer_dir, CLUSTERER)
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    def get_latest_dir_path_to_save(self) -> str:
        """Returns the latest dir where the latest models and relevant artifacts shall be stored.

        Raises:
            e: Raises relevant exception should any sort of error pops up while returning the latest dir 
            (for saving newer artifacts) path.

        Returns:
            str: Latest dir path to save the latest models and relevant artifacts.
        """
        try:
            lg.info('Configuring the dir path where the "latest artifacts" are to be saved..')
            latest_dir = self.get_latest_dir_path()

            if latest_dir is None:
                return os.path.join(self.model_registry, str(0))
            latest_dir_num = int(os.path.basename(latest_dir))
            return os.path.join(self.model_registry, str(latest_dir_num+1))
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    def save_latest_preprocessor_at(self) -> str:
        """Path in the Model Registry to save the latest Preproessor at.

        Raises:
            e: Raises relevant exception should any sort of error pops up while returning the latest preprocessor 
            path (for saving newer preprocessor).

        Returns:
            str: Dir path where the latest Preprocessor is to be stored.
        """
        try:
            latest_dir_to_save = self.get_latest_dir_path_to_save()
            lg.info('Configuring the path where the "latestly built Preprocessor" is to be stored..')
            return os.path.join(latest_dir_to_save, self.preprocessor_dir, PREPROCESSOR)
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    def save_latest_clusterer_at(self) -> str:
        """Path in the Model Registry to save the latest Clusterer at.

        Raises:
            e: Raises relevant exception should any sort of error pops up while returning the latest clusterer 
            path (for saving newer clusterer).

        Returns:
            str: Path where the latest Clusterer is to be stored at.
        """
        try:
            latest_dir_to_save = self.get_latest_dir_path_to_save()
            lg.info('Configuring the path where the "latestly built Clusterer" is to be stored..')
            return os.path.join(latest_dir_to_save, self.clusterer_dir, CLUSTERER)
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    def save_latest_models_at(self) -> str:
        """Dir path in the Model Registry to save the latest Models at.
        
        Raises:
            e: Raises relevant exception should any sort of error pops up while returning the latest model dir 
            (for saving newer models) path.

        Returns:
            str: Dir path where the latest Models are to be stored.
        """
        try:
            latest_dir = self.get_latest_dir_path()
            lg.info('Configuring the path where the "latestly trained Models" are to be stored..')
            return os.path.join(latest_dir, self.model_dir)
            ...
        except Exception as e:
            lg.exception(e)
            raise e
