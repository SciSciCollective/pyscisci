# -*- coding: utf-8 -*-
"""
.. module:: network normalized citation index
    :synopsis: Set of functions for finding the network normalized citation index.

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

from pyscisci.utils import isin_sorted

### Network Normalized Citation
def netnormcite_index(pub2ref, pub2year=None, focus_pub_ids = None, T=5, show_progress=False):
    """
    Calculate the network normalized citation index as first proposed in :cite:`Ke2023netnorm`.

    Parameters
    ----------

    :param pub2ref : DataFrame
        A DataFrame with the citation information for each Publication.

    :param pub2year : DataFrame, optional
        A DataFrame with the year of publication for each Publication if this is not already included in pub2ref.

    :param focus_pub_ids : numpy array
        A subset of publication ids to focus on for the citation index.  
        Note, the full pub2ref is still required because we need to find the co-citation neighborhoods.

    :param T : int, default 5
        Number of years for citation window, must be 1 or greater.

    show_progress : bool, default False
        Show calculation progress. 

    Returns
    -------
    disruption : DataFrame
        A DataFrame with the network normalized citation index for all (cited) publications or publications from the focus_pub_ids list.


    """
    if show_progress:
        print("Starting computation of network normalized index.")

    if not ('CitingYear' in list(pub2ref) or 'CitedYear' in list(pub2ref)):
        pub2ref = pub2ref.merge(pub2year, how='left', left_on='CitedPublicationId', right_on ='PublicationId').rename(columns={'Year':'CitedYear'})
        del pub2ref['PublicationId']
        pub2ref = pub2ref.merge(pub2year, how='left', left_on='CitingPublicationId', right_on ='PublicationId').rename(columns={'Year':'CitingYear'})
        del pub2ref['PublicationId']

    if focus_pub_ids is None:
        yfocus_pubs = pub2ref[['CitedPublicationId', 'CitedYear']].drop_duplicates(keep='first')
    else:
        yfocus_pubs = pub2ref[isin_sorted(pub2ref['CitedPublicationId'].values, np.sort(focus_pub_ids))][['CitedPublicationId', 'CitedYear']].drop_duplicates(keep='first')


    reference_groups = pub2ref.groupby(['CitingPublicationId'], sort = False)['CitedPublicationId']
    def get_reference_groups(pid):
        try:
            return reference_groups.get_group(pid).values
        except KeyError:
            return np.array([])

    citation_groups = pub2ref.groupby(['CitingYear', 'CitedPublicationId'], sort = False)['CitingPublicationId']
    def get_citation_groups(pid, y):
        try:
            return citation_groups.get_group((y, pid)).values
        except KeyError:
            return np.array([])

    yearly_citation_counts = citation_groups.nunique()
    def get_yearly_ncites(y, pid):
        try:
            return yearly_citation_counts[(y, pid)]
        except KeyError:
            return 0


    def _netnorm_index(focusid, y):

        cnormt = 0
        for t in range(0, T+1):

            paper2y_cocite = {refid:get_yearly_ncites(y+t,refid) for citeid in get_citation_groups(focusid, y+t) for refid in get_reference_groups(citeid) }
            
            # the co-citation neighborhood doesnt include the focus publication
            cnorm_denom = sum(ncites for refid, ncites in paper2y_cocite.items() if refid != focusid)
            
            if cnorm_denom > 0 and len(paper2y_cocite)  > 1:
                cnorm_denom = cnorm_denom / (len(paper2y_cocite) - 1)
                cnormt += get_yearly_ncites(y+t,focusid) / cnorm_denom

        return cnormt

    netnorm = [[focus_pub, yfocus, _netnorm_index(focus_pub, yfocus)] for focus_pub, yfocus
        in tqdm(yfocus_pubs.values, leave=True, desc='Network-normalized Citation', disable= not show_progress)]

    return pd.DataFrame(netnorm, columns = ['PublicationId', 'CitedYear', 'Cnorm{}'.format(T)])

