import pymongo
import pandas as pd
from wafer.logger import lg
import os
from typing import List
from dataclasses import dataclass


@dataclass
class MongoDBOperations:
    """This class is exclusively for performing all MongoDB pertaining operations.
    
    Args:
        connection_string (str): Takes in the `client url` to establish connection to MongoDB.
        database_name (str): Database to which connection is to be established.
        collection_name (str): Desired collection name of the said database. 
    """
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.dBOperations" class')

    connection_string: str
    database_name: str
    collection_name: str
    client = None
    database = None
    collection = None

    def establishConnectionToMongoDB(self):
        """This method establishes the connection to the desired MongoDB Cluster.

        Raises:
            e: Throws exception if any error pops up while establishing connection to MongoDB.
        """
        try:
            lg.info("establishing the connection to MongoDB..")
            self.client = pymongo.MongoClient(self.connection_string)
        except Exception as e:
            lg.exception(e)
            raise e
        else:
            lg.info("connection established successfully!")

    def selectDB(self):
        """This method chooses the desired dB from the MongoDB Cluster.

        Raises:
            e: Throws exception if any error pops up while selecting desired database from MongoDB.
        """
        try:
            self.establishConnectionToMongoDB()
            lg.info("searching for the database..")
            self.database = self.client[self.database_name]
        except Exception as e:
            lg.exception(e)
            raise e
        else:
            lg.info(
                f'"{self.database_name}" database chosen succesfully!')

    def createOrselectCollection(self):
        """This method shall create the desired collection in the selected database of the MongoDB Cluster.

        Raises:
            e: Throws exception if any error pops up while creating or selecting any desired collection in selected
            database of MongoDB.
        """
        try:
            self.selectDB()

            collections = self.database.list_collection_names()
            lg.info(
                f"looking for the collection \"{self.collection_name}\" in the database..")
            if self.collection_name in collections:
                lg.info(
                    f'collection found! selecting the collection" {self.collection_name}"..')
                self.collection = self.database[self.collection_name]
                lg.info("..said collection selected successfully!")
            else:
                lg.warning(
                    f'collection "{self.collection_name}" not found in the database, gotta create it..')
                lg.info(
                    f'creating the collection "{self.collection_name}"..')
                self.collection = self.database[self.collection_name]
                lg.info("..said collection created successfully!")
        except Exception as e:
            lg.exception(e)
            raise e

    def dumpData(self, records: List, data_desc: str):
        """Dumps the desired bulk data (to be parameterized in a form of list) to the

        Raises:
            e: Throws exception if any error pops up while dumping data into selected database of MongoDB.

        Args:
            records (List): The bulk data that's to be dumped into the collection (in a form of List).
            data_desc (str): Description of the data that's to be dumped.
        """
        try:
            self.createOrselectCollection()

            lg.info(
                f'dumping the "{data_desc} data" to the collection "{self.collection_name}"..')
            self.collection.insert_many(records)
            ...
        except Exception as e:
            lg.exception(e)
            raise e
        else:
            lg.info(f'dumped "{data_desc} data" with success!')

    def getDataAsDataFrame(self) -> pd.DataFrame:
        """This method prepares a feature-store-file out of all the data from the selected database.

        Raises:
            e: Throws exception if any error pops up while loading data as dataframe from MongoDB's database.

        Returns:
            pandas.DataFrame: Data from the given collection of the MongoDB database in form of pandas dataframe.
        """
        try:
            self.createOrselectCollection()
            lg.info(
                f'fetching all the data from collection "{self.collection_name}" of database "{self.database_name}"..')
            df = pd.DataFrame(list(self.collection.find()))
            lg.info("data readied as the dataframe!")
            df.drop(columns=["_id"], inplace=True)
        except Exception as e:
            lg.exception(e)
            raise e
        else:
            lg.info("returning the extracted data as dataframe..")
            return df

    def emptyCollection(self) -> None:
        """Strips off the selected collection of selected MongoDB database of all records.

        Raises:
            e: Raises exception should any kinda error pops up while execution of this method.
        """
        try:
            self.createOrselectCollection()
            lg.info(f'Emptying the collection "{self.collection_name}" of database "{self.database_name}"..')
            self.collection.delete_many({})
            lg.info(f'..successfully emptied the collection "{self.collection_name}" of database "{self.database_name}"!')
            ...
        except Exception as e:
            lg.exception(e)
            raise e


if __name__ == "__main__":
    MongoDBOperations()