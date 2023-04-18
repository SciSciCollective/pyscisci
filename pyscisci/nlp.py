# -*- coding: utf-8 -*-
"""
.. module:: scinlp
    :synopsis: The Natual Langauge Processing for SciSci

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>

 """
import os
import sys
import re
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from pyscisci.sparsenetworkutils import project_bipartite_mat, threshold_network
from pyscisci.utils import *

import unicodedata
from unidecode import  unidecode
from nameparser import HumanName

# For now, I dont think we need to make the full pySciSci package dependent on these packages
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

def abstractindex2text(abstract_index):
    """
    The abstracts in OpenAlex and MAG are saved in the inverted index format.  This function coverts back to the original text.
    """

    # unpack the index list
    word_index = [ (word, idx) for word, idxlist in abstract_index.items() for idx in idxlist] 
    # sort the words
    word_index = sorted(word_index, key = lambda x : x[1])
    # insert the spaces back in and return
    return " ".join(list(zip(*word_index))[0])


def standardize_doi(doistr):
    """
    
    """

    

    
def strip_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

def clean_names(name, remove_parentheses=True):
    name = strip_accents(name)

    if remove_parentheses:
        name = re.sub("[\(\[].*?[\)\]]", "", name) # remove text between () and []
        if '(' in name:
            name=name[:name.index('(')]
        for c in [')', ']','.']:
            name=name.replace(c, '')
    for c in [',', ' - ', '- ', ' -']:
        name=name.replace(c, '-')
    name = name.replace(' & ', '&')
    return name.strip()

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

        matches = np.empty(tfidf1.shape[0])
        matches[:] = np.NaN

        # if there are no columns to match exactly
        if (columns2match_exact is None or len(columns2match_exact) == 0):

            tfidf2 = tfidf.transform(df2[column2match_approx])

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

                    sub_tfidf2 = tfidf.transform(df2.loc[df2groups.indices[subdf.name]][column2match_approx])

                    #sub_tfidf2 = tfidf2[df2groups.indices[subdf.name]]
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

def coword_network(df, text_column='Title', stop_words= 'english', strip_accents='ascii', lowercase=True, threshold=1, vocabulary=None, show_progress=False):
    """
    Create the co-word network.

    Parameters
    ----------
    df : DataFrame
        A DataFrame with the text snippits (typically titles or abstracts).

    text_column : str, default "Title"
        The column of text snippits

    stop_words : str or list of str, default 'english'
        The stopword dictionary to employ.
            - 'english' : sklearn english stopwords
            - list of str : a list specifying the exact stopwords to remove
            - None : no stopword dictionary (use all stopwords)

    strip_accents : str, default 'ascii'
        Remove accents and perform other character normalization during the preprocessing step.  None does nothing.

    lowercase : bool, default True
        Convert all characters to lowercase before tokenizing.

    threshold : int, default 1
        The minimum number of times two words should appear together.

    vocabulary : list, default None
        List of focus terms to limit. If not given, a vocabulary is determined from the input text. 

    show_progress : bool, default False
        If True, show a progress bar tracking the calculation.

 

    Returns
    -------
    coo_matrix or dict of coo_matrix
        The adjacency matrix for the co-authorship network

    word2int, dict
        A mapping of words to the row/column of the adjacency matrix.


    """
    required_columns = ['PublicationId', text_column]
    check4columns(df, required_columns)
    df = df[required_columns].dropna()

    df.drop_duplicates(subset=required_columns, inplace=True)

    cv = CountVectorizer(stop_words=stop_words, analyzer='word', min_df=threshold, lowercase=lowercase, strip_accents=strip_accents,
        vocabulary=vocabulary)
    bipartite_adj = cv.fit_transform(df[text_column].values)

    word2int = cv.vocabulary_

    adj_mat = project_bipartite_mat(bipartite_adj, project_to = 'col')
    adj_mat = threshold_network(adj_mat)

    return adj_mat, word2int

