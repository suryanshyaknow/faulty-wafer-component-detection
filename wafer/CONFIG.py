import os
from wafer.logger import lg
from dotenv import load_dotenv
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Configuration class that shall be used to access any sort of config pertaining to the relational
    database used in this project.
    """
    load_dotenv()
    mongodb_url: str = os.getenv("MONGO_DB_URL")
    database_name: str = "wafers"
    collection_name: str = "training-batch"