# -*- coding: utf-8 -*-
"""
.. module:: novelty
    :synopsis: The novelty and conventionality.

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """
import numpy as np
import scipy.sparse as spsparse
import tempfile

default_scratch_path = tempfile.TemporaryDirectory()

def compute_novelty(database, scratch_path = '', n_samples = 10):
    """
    This function calculates the yearly cnorm for a publication.

    The cnorm is the number of citations a publication recieved in a year, divided by the average number of citations
    all publications in the co-citation neighborhood recieved that year (Ke [h]_).

    :param BibDataBase database:
        The bibliometric database object.

    :param list citations:
        The list of citing publications.

    :returns:
        The yearly cnorm

    See the Publication() object for usage

    References
    ----------
    .. [h] Ke, Q., Gates, A.J., Uzzi, B. (2013): "title",
           *in submission*.
           DOI: xxx
    """

    database_table, int2journal = create_database_table(database)

    for isample in range(n_samples):
        database_table = database_table.groupby(['CitingYear', 'CitedYear'], sort=False)['CitedJournal'].transform(np.random.permutation)


def get_scratch_path():
    return default_scratch_path.name

def create_database_table(database):
    infotable = []
    for pub in database.publication_generator():
        for ref in pub.references:
            infotable.append([pub.year, pub.journal, ref.year, ref.journal])

    infotable = pd.DataFrame(infotable, columns = ['CitingYear', 'CitingJournal', 'CitedYear', 'CitingJournal'])

    journals = sorted([journal.id for journal in database.journal_generator()])
    journal2int = {j:i for i,j in enumerate(journals)}

    infotable['CitingJournal'] = [journal2int[j] for j in infotable['CitingJournal']]
    infotable['CitedJournal'] = [journal2int[j] for j in infotable['CitedJournal']]

    return infotable, {i:j for j,i in journal2int.items()}

def database_table_to_journalmatrix(database_table):



