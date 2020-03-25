# -*- coding: utf-8 -*-
"""
.. module:: disruption
    :synopsis: The disruption index.

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

def compute_disruption(database, year, references, citations):
    """ This function calculates the distruption index for a publication.

    See :cite:`Origdisrupt` [w]_.

    :param BibDataBase database:
        The bibliometric database object.

    :param int year:
        The publication's year.

    :param list references:
        The list of publication objects in the references.

    :param list citations:
        The list of publication objects in the citations.

    :returns:
        The disruption index

    Notes
    ----------
    See the Publication() object for usage

    References
    ----------
        [w] Wu, J. E., Wang, D., Evans, J. (2019): "Small teams disrput in science and technology",
           *Nature* 102(46).
           DOI: 10.1073/pnas.0507655102
    """

    cites2ref_id = set([])
    for pub in references:
        cites2ref_id.update([database.get_pub(cite2ref).id for ref in references for cite2ref in database.get_pub(ref).citations if database.get_pub(cite2ref).year >= year])

    cites2me_id = set(citations)

    ni = len(cites2me_id - cites2ref_id)
    nj = len(cites2me_id & cites2ref_id)
    nk = len(cites2ref_id - cites2me_id)


    return (ni - nj) / (ni + nj + nk)