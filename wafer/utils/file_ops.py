import json
from wafer.logger import lg
from typing import Dict


class BasicUtils:
    """Shall be used for accessing basic utilities methods."""

    @classmethod
    def read_json_file(cls, file_path: str, file_desc: str) -> Dict:
        """Loads and returns the json file's content located at `file_path`, if there's one and throws exception 
        if there's none.

        Args:
            file_path (str): Location of the json file that's to be loaded.
            file_desc (str): Description of said json file.

        Raises:
            e: Throws relevant exception if any error pops while loading or returning the json file's content.

        Returns:
            Dict: Json file's content.
        """
        try:
            lg.info(
                f"fetching the data from the \"{file_desc}\" lying at \"{file_path}\"..")
            with open(file_path, 'r') as f:
                data = json.load(f)
                lg.info("data fetched successfully!")
            f.close()

            return data
            ...
        except Exception as e:
            lg.exception(e)
            raise e
