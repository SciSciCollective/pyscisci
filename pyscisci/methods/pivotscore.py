# -*- coding: utf-8 -*-
"""
.. module:: pivotscore
    :synopsis: Set of functions for typical bibliometric citation analysis

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """
import sys
import pandas as pd
import numpy as np

from pyscisci.utils import groupby_count, changepoint, pandas_cosine_similarity

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm




### Pivot Score
def pivot_score(pub2author, pub2ref, previous_k=None, year_window=None, show_progress=False):
    """
    Calculate the pivot score index as proposed in :cite:`Hill2022pivotpenalty`.

    None is returned for the first publication in an author's career (because there is no previous history), or when
    no other publications have been published within the year_window.

    Parameters
    ----------
    pub2author : dataframe
        The publication author linkages with Year.

    pub2ref : dataframe
        The citing-cited publication linkages which contains the CitedJournalId for the cited articles.

    previous_k : int, default None
        Only compare against the previous k publications.

    year_window : int, default None
        Only compare against publications from the last year_window years.

    show_progress : bool, default False
        Show calculation progress. 

    Returns
    ----------
    pivotscore : DataFrame
        The pivotscore for each Author Publication.

    """
    if show_progress:
        print("Starting computation of pivot score.")

    pub2refjournalcounts = pyscisci.groupby_count(pub2ref, ['CitingPublicationId', 'CitedJournalId'], 
                                              'CitedPublicationId', count_unique=True)
    pub2refjournalcounts.rename(columns={'CitedPublicationIdCount':'CitedJournalCount'}, inplace=True)
    
    pa_refs = paa.merge(pub2refjournalcounts, how='left', left_on = 'PublicationId', right_on='CitingPublicationId')
    del pa_refs['CitingPublicationId']
    
    pa_refs.dropna(inplace=True)
    pa_refs['CitedJournalId'] = pa_refs['CitedJournalId'].astype(int)
    pa_refs.sort_values(by=['AuthorId', 'Year', 'PublicationId', 'CitedJournalId'], inplace=True)
    pa_refs.reset_index(drop=True, inplace=True)

    pscore = pa_refs.groupby('AuthorId').apply(author_pivot, 
                                               previous_k=previous_k, year_window=year_window).reset_index()
    del pscore['level_1']

    return pscore


def author_pivot(authordf):
        
        pubgroups = authordf.groupby('PublicationId', sort=False)
        
        allpubidx = None
        if not previous_k is None:
            allpubidx = pyscisci.changepoint(authordf['PublicationId'].values)
        
        
        pivotresults = []
        
        def publication_pivot(pubgroup):
            pubidx = pubgroup.index[0]
            pid = pubgroup.name
            if pubidx==0: pivotresults.append([pid, None])
            else:
                i=len(pivotresults)
                if not previous_k is None and i > previous_k:
                    history = authordf.iloc[allpubidx[i-previous_k]:pubidx]
                else:
                    history = authordf.iloc[:pubidx]

                if not year_window is None:
                    history = history[history['Year'] >= pubgroup['Year'].values[0] - year_window]

                if history.shape[0] > 0:
                    history = history.groupby('CitedJournalId', sort=False, as_index=False)['CitedJournalCount'].sum()
                    
                    cosine = pandas_cosine_similarity(history, pubgroup, col_key='CitedJournalId', col_values='CitedJournalCount')
                    
                    pivotresults.append([pid, cosine])
                else:
                    pivotresults.append([pid, None])  
        
        pubgroups.apply(publication_pivot)
                
        return pd.DataFrame(pivotresults, columns=['PublicationId', 'PivotScore'])
