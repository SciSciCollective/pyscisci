# -*- coding: utf-8 -*-
"""
.. module:: distruption index
    :synopsis: Set of functions for finding the disruption index.

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """
import sys
import pandas as pd
import numpy as np

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

### Disruption
def disruption_index(pub2ref, focus_pub_ids = None, cite_window = None, ref_window = None, show_progress=False):
    """
    Calculate the disruption index as first proposed in :cite:`Funk2017disrupt` and used in :cite:`Wu2019teamsdisrupt`.
    We also include the windowed disruption index used in :cite:`Park2023timedisrupt`.

    Parameters
    ----------

    :param pub2ref : DataFrame
        A DataFrame with the citation information for each Publication.

    :param focus_pub_ids : numpy array
        A subset of publication ids to focus on for the disruption index.

    :param cite_window : list of two ints, default None
        If None, no citation window is applied.
        If [l, u] where, l,u are ints, then only citations whose year difference is greater than or equal to the lower bound l and 
            less than or equal to the upper bound u are used.  e.g. [0,5] uses citations within 5 years of publication (and not before).

    :param ref_window : list of two ints, default None
        If None, no reference window is applied.
        If [l, u] where, l,u are ints, then only references whose year difference is greater than or equal to the lower bound l and 
            less than or equal to the upper bound u are used.  e.g. [0,5] uses references within 5 years of publication (and not after).

    show_progress : bool, default False
        Show calculation progress. 

    Returns
    -------
    disruption : DataFrame
        A DataFrame with the disruption index for all (cited) publications or publications from the focus_pub_ids list.


    """
    if show_progress:
        print("Starting computation of disruption index.")

    if ref_window is None:
        reference_groups = pub2ref.groupby('CitingPublicationId', sort = False)['CitedPublicationId']
    else:
        ref_sub = [ ((y1-y2) >= ref_window[0] and (y1-y2) <=ref_window[1]) for y1,y2 in pub2ref[['CitingYear', 'CitedYear']].values]
        reference_groups = pub2ref.loc[ref_sub].groupby('CitingPublicationId', sort = False)['CitedPublicationId']

    if cite_window is None:
        citation_groups = pub2ref.groupby('CitedPublicationId', sort = False)['CitingPublicationId']
    else:
        cite_sub = [ ((y1-y2) >= cite_window[0] and (y1-y2) <=cite_window[1]) for y1,y2 in pub2ref[['CitingYear', 'CitedYear']].values]
        citation_groups = pub2ref.loc[cite_sub].groupby('CitingPublicationId', sort = False)['CitedPublicationId']

    if focus_pub_ids is None:
        if cite_window is None:
            focus_pub_ids = pub2ref['CitedPublicationId'].unique()
        else:
            focus_pub_ids = pub2ref.loc[cite_sub]['CitedPublicationId'].unique()

    def get_citation_groups(pid):
        try:
            return citation_groups.get_group(pid).values
        except KeyError:
            return np.array([])

    def _disruption_index(focusid):

        # if the focus publication has no references or citations, then it has a disruption of None
        try:
            focusref = reference_groups.get_group(focusid)
        except KeyError:
            return None

        try:
            citing_focus = citation_groups.get_group(focusid)
        except KeyError:
            return None


        # implementation 1: keep it numpy
        #cite2ref = reduce(np.union1d, [get_citation_groups(refid) for refid in focusref])
        #nj = np.intersect1d(cite2ref, citing_focus.values).shape[0]
        #nk = cite2ref.shape[0] - nj

        # implementation 2: but dicts are faster...
        cite2ref = {citeid:1 for refid in focusref for citeid in get_citation_groups(refid)}
        nj = sum(cite2ref.get(pid, 0) for pid in citing_focus.values )
        nk = len(cite2ref) - nj

        ni = citing_focus.shape[0] - nj

        return float(ni - nj)/(ni + nj + nk)

    disrupt = [[focusciting, _disruption_index(focusciting)] for focusciting
        in tqdm(focus_pub_ids, leave=True, desc='Disruption Index', disable= not show_progress) if get_citation_groups(focusciting).shape[0] > 0]

    return pd.DataFrame(disrupt, columns = ['PublicationId', 'DisruptionIndex'])

