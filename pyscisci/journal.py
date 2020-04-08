# -*- coding: utf-8 -*-
"""
.. module:: journal
    :synopsis: The main Journal class

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

from .publicationcollection import PublicationCollection

class Journal(PublicationCollection):
    """
    Base class Journals

    """

    def __init__(self, database, journaldict = {}):

        self.database = database

        self.from_dict(journaldict)


    def reset_properties(self):

        self.reset_collection_properties()

        return self

    def from_dict(self, journal_dict):

        self.id = journal_dict.get('id', 0)
        self.fullname = journal_dict.get('fullname', None)
        self.issn = journal_dict.get('lastname', None)
        self.publisher = journal_dict.get('publisher', None)

        self.collection_from_dict(journal_dict)


    def citation_index(self, year, k=2):
        numerator = sum([self.yearly_citations.get(year - ik, 0) for ik in range(1, k+1)])
        denominator = sum([self.yearly_productivity.get(year - ik, 0) for ik in range(1, k+1)])

        if denominator == 0:
            return 0.0
        else:
            return numerator / float(denominator)

