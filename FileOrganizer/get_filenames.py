import os
from loguru import logger
from pprint import pprint


def get_files_in_directory(dir_path=r"C:\Users\User\Downloads"):
    filenames = os.listdir(dir_path)
    # logger.info(f"filenames are {filenames}")
    pprint(filenames)
    return filenames
