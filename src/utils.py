import os
import re
import pandas as pd
import hashlib

from typing import List


def clean_string(
    string: str,
):
    string = re.sub(
        r"\d+",
        "",
        string,
    )
    string = re.sub(
        r"\W+",
        "",
        string,
    )  # remove non-alphanumeric characters

    return string


def read_folder(
    data_folder: str,
) -> List[str]:
    try:
        files = os.listdir(data_folder)
        return [
            os.path.join(
                data_folder,
                file,
            )
            for file in files
        ]
    except FileNotFoundError:
        raise FileNotFoundError(f"Data folder '{data_folder}' not found.")
    except Exception as e:
        raise Exception(f"An error occurred while reading data folder '{data_folder}': {str(e)}")


def hash_string(
    string: str,
):
    return hashlib.sha256(string.encode()).hexdigest()
