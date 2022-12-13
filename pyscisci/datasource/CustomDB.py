import os
import sys
import json
import gzip
import zipfile

import pandas as pd
import numpy as np
from nameparser import HumanName

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from pyscisci.datasource.readwrite import load_preprocessed_data, load_int, load_float, load_html_str
from pyscisci.database import BibDataBase
from pyscisci.utils import download_file_from_google_drive

# hide this annoying performance warnings
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class CustomDB(BibDataBase):
    """
    Base class for creating a CustomDB.

    """

    def __init__(self, path2database = '', database_extension='csv.gz', keep_in_memory = False, global_filter=None, 
        enable_dask=False, show_progress=True):

        self._default_init(path2database, database_extension, keep_in_memory, global_filter, enable_dask, show_progress)

        self.PublicationIdType = int
        self.AffiliationIdType = int
        self.AuthorIdType = int
        self.JournalIdType = int

    def set_new_data_paths(new_path_dict={}):
        """
        Override path to the dataframe collections based on a new custom hierarchy.

        Parameters
        --------
        new_path_dict : dict
            A dictionary where each key is a dataframe name to override.  E.g. 'author', 'pub', 'paa', 'pub2field', etc.
            and each item is the new dataframe path.

        """
        for dfname, new_path in new_path_dict.items():
            self.set_new_data_path(dfname, new_path)
