import re
from os.path import isfile, join, exists, dirname
import os
from pathlib import Path
from typing import List, Optional


class FileSystem:
    """!
    A helper class providing useful functions to interact with the filesystem.
    """

    @staticmethod
    def files_in(directory: str, regex: Optional[str] = None) -> List[str]:
        """!
        Retrieve the name of the files present within the directory passed as parameters.
        @param directory: the directory whose files must be returned
        @param regex: a regex that filters the file name to retrieve (None for no filter)
        @return the files
        """

        # Compile the regex, if needed.
        if regex is not None:
            regex = re.compile(regex)

        # Iterate over all directory entries.
        files = []
        for entry in os.listdir(directory):

            # Add the current entry, if it is a file matching the regex.
            if not isfile(join(directory, entry)):
                continue
            if regex is None:
                files.append(entry)
            elif regex.match(entry):
                files.append(entry)

        return files

    @staticmethod
    def create_directory_and_file(path: str) -> None:
        """!
        Create the directory and file corresponding to the path passed as parameter.
        @param path: the file's path
        """
        checkpoint_dir = dirname(path)
        if not exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not exists(path):
            Path(path).touch(exist_ok=True)
