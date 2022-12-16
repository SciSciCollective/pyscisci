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
def disruption_index(pub2ref, focus_pub_ids = None, show_progress=False):
    """
    Calculate the disruption index as first proposed in :cite:`Funk2017disrupt` and used in :cite:`Wu2019teamsdisrupt`.

    

    """
    if show_progress:
        print("Starting computation of disruption index.")

    if focus_pub_ids is None:
        focus_pub_ids = pub2ref['CitedPublicationId'].unique()

    reference_groups = pub2ref.groupby('CitingPublicationId', sort = False)['CitedPublicationId']
    citation_groups = pub2ref.groupby('CitedPublicationId', sort = False)['CitingPublicationId']

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

