import os
import pandas as pd
import numpy as np

from pyscisci.utils import isin_sorted

def load_preprocessed_data(dataname, path2database, columns = None, isindict = None, duplicate_subset = None,
    duplicate_keep = 'last', dropna = None, keep_source_file = False, func2apply = None):

    #TODO: progress bar

    path2files = os.path.join(path2database, dataname)
    if not os.path.exists(path2files):
        # TODO: make a real warning
        print("First preprocess the raw data.")
        return []

    if isinstance(columns, str):
        columns = [columns]

    if isinstance(dropna, str):
        dropna = [dropna]

    if isinstance(duplicate_subset, str):
        duplicate_subset = [duplicate_subset]

    if isinstance(isindict, dict):
        isindict = {isinkey:np.sort(isinlist) for isinkey, isinlist in isindict.items()}


    Nfiles = sum(dataname in fname for fname in os.listdir(path2files))

    data_df = []
    for ifile in range(Nfiles):
        fname = os.path.join(path2files, dataname+"{}.hdf".format(ifile))
        subdf = pd.read_hdf(fname, mode = 'r')

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

        if callable(func2apply):
            func2apply(subdf)

        data_df.append(subdf)

    data_df = pd.concat(data_df)

    if isinstance(duplicate_subset, list):
        data_df.drop_duplicates(subset = duplicate_subset, keep = duplicate_keep, inplace = True)

    return data_df

def append_to_preprocessed_df(newdf, path2database, preprocessname):

    path2files = os.path.join(path2data, preprocessname)

    Nfiles = sum(preprocessname in fname for fname in os.listdir(path2files))

    for ifile in range(Nfiles):
        datadf = pd.read_hdf(os.path.join(path2files, preprocessname + '{}.hdf'.format(ifile)))
        datadf = datadf.merge(newdf, how = 'left')
        datadf.to_hdf(os.path.join(path2files, preprocessname + '{}.hdf'.format(ifile)), key = preprocessname, mode = 'w')

