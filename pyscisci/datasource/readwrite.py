import os
import sys
import pandas as pd
import numpy as np

from unidecode import unidecode
from html.parser import HTMLParser

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


from pyscisci.utils import isin_sorted

def load_int(v):
    try:
        return int(v)
    except ValueError:
        return None

def load_float(v):
    try:
        return float(v)
    except ValueError:
        return None

def load_html_str(s):
    if s is None:
        return ''
    else:
        return unidecode(HTMLParser().unescape(s))

def load_xml_text(root_element, default=''):
    if len(root_element) == 0:
        return default
    else:
        return root_element[0].text


def load_preprocessed_data(dataname, path2database, columns = None, isindict=None, duplicate_subset=None,
    duplicate_keep='last', dropna=None, keep_source_file=False, prefunc2apply=None, postfunc2apply=None, show_progress=False):
    """
        Load the preprocessed DataFrame from a preprocessed directory.

        Parameters
        ----------
        :param dataname : str
            The type of preprocessed data to load.

        :param path2database : str
            The path to the database directory.

        :param columns : list, default None
            Load only this subset of columns

        :param isindict : dict, default None
            Dictionary of format {"ColumnName":"ListofValues"} where "ColumnName" is a data column
            and "ListofValues" is a sorted list of valid values.  A DataFrame only containing rows that appear in
            "ListofValues" will be returned.

        :param duplicate_subset : list, default None
            Drop any duplicate entries as specified by this subset of columns

        :param duplicate_keep : str, default 'last', Optional
            If duplicates are being dropped, keep the 'first' or 'last'
            (see `pandas.DataFram.drop_duplicates <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html>`_)

        :param dropna : list, default None, Optional
            Drop any NaN entries as specified by this subset of columns

        :param keep_source_file : bool, default False
            Keep track of the source file the data was loaded from.

        :param prefunc2apply : callable, default None
            A function to apply to each of the sub-DataFrames as they are loaded before filtering.

        :param postfunc2apply : callable, default None
            A function to apply to each of the sub-DataFrames as they are loaded after filtering.

        Returns
        -------
        DataFrame
            dataname DataFrame.

    """

    path2files = os.path.join(path2database, dataname)
    if not os.path.exists(path2files):
        # TODO: make a real warning
        raise NotImplementedError("First preprocess the raw data.")
        return []

    if isinstance(columns, str):
        columns = [columns]

    if isinstance(dropna, str):
        dropna = [dropna]

    if isinstance(duplicate_subset, str):
        duplicate_subset = [duplicate_subset]

    if isinstance(isindict, dict):
        isindict = {isinkey:np.sort(isinlist) for isinkey, isinlist in isindict.items()}


    FileNumbers = sorted([int(fname.replace(dataname, '').split('.')[0]) for fname in os.listdir(path2files) if dataname in fname])

    desc=''
    if isinstance(show_progress, str):
        desc = show_progress

    data_df = []
    for ifile in tqdm(FileNumbers, desc=desc, leave=True, disable=not show_progress):
        fname = os.path.join(path2files, dataname+"{}.hdf".format(ifile))
        subdf = pd.read_hdf(fname, mode = 'r')

        if callable(prefunc2apply):
            subdf = prefunc2apply(subdf)

        if isinstance(columns, list):
            subdf = subdf[columns]

        if isinstance(dropna, list):
            subdf.dropna(subset = dropna, inplace = True, how = 'any')

        if isinstance(isindict, dict):
            for isinkey, isinlist in isindict.items():
                subdf = subdf[isin_sorted(subdf[isinkey], isinlist)]

        if isinstance(duplicate_subset, list):
            subdf.drop_duplicates(subset = duplicate_subset, keep = duplicate_keep, inplace = True)

        if keep_source_file:
            subdf['filetag'] = ifile

        if callable(postfunc2apply):
            postfunc2apply(subdf)

        data_df.append(subdf)

    data_df = pd.concat(data_df)

    if isinstance(duplicate_subset, list):
        data_df.drop_duplicates(subset = duplicate_subset, keep = duplicate_keep, inplace = True)

    return data_df

def append_to_preprocessed_df(newdf, path2database, preprocessname, startcount=0):
    """
        Append the newdf to the preprocessed DataFrames from a preprocessed directory.

        Parameters
        ----------
        :param newdf : DataFrame
            The new DataFrame to save on the processed data.  It must have at least one column in common.

        :param path2database : str
            The path to the database directory.

        :param preprocessname : str
            The type of preprocessed data to which we append the new data.

    """

    path2files = os.path.join(path2database, preprocessname)

    Nfiles = sum(preprocessname in fname for fname in os.listdir(path2files))

    for ifile in range(Nfiles):
        datadf = pd.read_hdf(os.path.join(path2files, preprocessname + '{}.hdf'.format(ifile+startcount)))
        datadf = datadf.merge(newdf, how = 'left')
        datadf.to_hdf(os.path.join(path2files, preprocessname + '{}.hdf'.format(ifile+startcount)), key = preprocessname, mode = 'w')
