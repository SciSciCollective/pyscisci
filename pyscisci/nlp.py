# -*- coding: utf-8 -*-
"""
.. module:: scinlp
    :synopsis: The Natual Langauge Processing for SciSci

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>

 """
import os
import sys
import pandas as pd
import numpy as np

# For now, I dont think we need to make the full pySciSci package dependent on these packages
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
except ImportError:
    raise ImportError('Please install sklearn to take full advantage of the NLP tools. \n pip install sklearn')

try:
    from Levenshtein import ratio
except ImportError:
    raise ImportError('Please install python-Levenshtein to take full advantage of the NLP tools. \n pip install python-Levenshtein')

try:
    from sparse_dot_topn import awesome_cossim_topn
except ImportError:
    raise ImportError("Please install sparse_dot_topn to take full advantage of the NLP tools. \n pip install sparse_dot_topn")

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def levenshtein_best_match(s1, options, lower_bound=0.75):
    """
    Uses Levenshtein distance to pick the best match from a list of options.

    Returns tuple: (index in the options list, Levenshtein dist)
    """

    return max(zip(range(len(options)), [ratio(s1, s) for s in options]), key = lambda x: x[1])

def align_publications(df1, df2=None, columns2match_exact=['Year'], column2match_approx='Title', ntop=1, cosine_lower_bound=0.75,
    use_threads=False, n_jobs=2, lev_lower_bound=0.9, show_progress=False):
    """
    Fast way to match publications between two datasets.  We first match subsets of exact values
    between the two DataFrames, as specified by `columns2match_exact`.
    We then use a fast approximate string matching to match values in `columns2match_approx` to within a threshold.

    Parameters
    ----------
    :param df1 : DataFrame
        A DataFrame with the publication information.

    :param df2 : DataFrame, Optional
        Another DataFrame with the publication information.  If None, then df1 is used again.

    :param columns2match_exact : list, Default: ['Year']
        The columns to match exactly between DataFrames.

    :param column2match_approx : list, Default: 'Title'
        The column to match approximately between DataFrames.

    :param ntop : int, Default 1
        The number of best matches from df2 to return for each row of df1.

    :param lower_bound : float, Default 0.75
        The lowerbound for cosine similarity when doing a fuzzy string match.

    :param use_threads : bool, Default False
        Use multithreading when calculating cosine similarity for fuzzy string matching.

    :param n_jobs : int, Optional, Default 2
        If use_threads is True, the number of threads to use in the parall calculation.

    :param show_progress : bool, Default False
        If True, show a progress bar tracking the calculation.

    """
    # we can do an exact match from merge
    if (columns2match_exact is None or len(columns2match_exact) > 0) and (column2match_approx is None or len(column2match_approx) == 0):
        # get the index name and reset the index to force it as a column
        indexcol = df2.index.name
        df2 = df2.reset_index(drop=False)
        # now merge the dataframes and drop duplicates giving an exact match
        mdf = df1.merge(df2[columns2match_exact + [indexcol]], how='left', on=columns2match_exact)
        mdf.drop_duplicates(subset=columns2match_exact, keep='first', inplace=True)
        return mdf[indexcol]

    # otherwise, if there is a column to match approximately then we need to prep for fuzzy matching
    elif len(column2match_approx) > 0:

        # we take a two-step approach to fuzzy matching
        # 1) first we employ a super fast but not very accurate cosine-similarity
        #    matching to narrow down the possible pair-wise matches
        #    for each string, we create feature vectors from 3-char counts
        tfidf = TfidfVectorizer(min_df=1, ngram_range = (3,3), analyzer='char', lowercase=False)
        tfidf1 = tfidf.fit_transform(df1[column2match_approx])
        tfidf2 = tfidf.transform(df2[column2match_approx])

        matches = np.empty(tfidf1.shape[0])
        matches[:] = np.NaN

        # if there are no columns to match exactly
        if (columns2match_exact is None or len(columns2match_exact) == 0):

            # 1) first do the all-to-all cosine similarity and extract up to the ntop best possible matches
            co= awesome_cossim_topn(tfidf1, tfidf2.T, ntop=ntop, lower_bound=cosine_lower_bound, use_threads=use_threads, n_jobs=n_jobs).tocoo()

            # 2) now use the Levenshtein
            for row in tqdm(set(co.row), desc="Align Publications", disable=not show_progress):
                rowcol = co.col[co.row==row]
                argmatch, lev_dist = levenshtein_best_match(df1.loc[row, column2match_approx], df2.iloc[rowcol][column2match_approx])
                if lev_dist >= lev_lower_bound:
                    matches[row] = rowcol[argmatch]


        else:

            df2groups = df2.groupby(columns2match_exact)

            def subgroup_match(subdf):
                if not df2groups.indices.get(subdf.name, None) is None:
                    sub_tfidf1 = tfidf1[subdf.index.values]
                    sub_tfidf2 = tfidf2[df2groups.indices[subdf.name]]
                    co = awesome_cossim_topn(sub_tfidf1, sub_tfidf2.transpose(), ntop=ntop, lower_bound=cosine_lower_bound, use_threads=use_threads, n_jobs=n_jobs).tocoo()

                    # 2) now use the Levenshtein distance to find the best match
                    for row in set(co.row):
                        rowcol = co.col[co.row==row]
                        argmatch, lev_dist = levenshtein_best_match(subdf.iloc[row][column2match_approx],
                            df2.iloc[df2groups.indices[subdf.name][rowcol]][column2match_approx])
                        if lev_dist >= lev_lower_bound:
                            matches[subdf.index.values[row]] = df2groups.indices[subdf.name][rowcol[argmatch]]

            # register our pandas apply with tqdm for a progress bar
            tqdm.pandas(desc='Publication Matches', disable= not show_progress)

            df1.groupby(columns2match_exact, group_keys=True).progress_apply(subgroup_match)

        return matches



