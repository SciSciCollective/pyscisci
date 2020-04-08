# -*- coding: utf-8 -*-
"""
.. module:: publicationcollection
    :synopsis: The main PublicationCollection class

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

from collections import Counter


class PublicationCollection(object):
    """
    Base class for PublicationCollection.

    :param dict elm2clu_dict: optional
        Initialize based on an elm2clu_dict: { elementid: [clu1, clu2, ... ] }.
        The value is a list of clusters to which the element belongs.

    """

    def __init__(self, database):

        self.database = database

        self.publications = []

        self.reset_collection_properties()


    def reset_collection_properties(self, property_dict = {}):

        self._start_year = property_dict.get('_start_year', None)
        self._end_year = property_dict.get('_end_year', None)
        self._publication_years = property_dict.get('_publication_years', None)

        self._career_length = property_dict.get('_career_length', None)
        self._total_productivity = property_dict.get('_total_productivity', None)
        self._yearly_productivity = property_dict.get('_yearly_productivity', None)
        self._annual_productivity = property_dict.get('_annual_productivity', None)

        self._total_citations = property_dict.get('_total_citations', None)
        self._yearly_citations = property_dict.get('_yearly_citations', None)
        self._cummulative_citations = property_dict.get('_cummulative_citations', None)


    def collection_from_dict(self, property_dict):

        self.publications = property_dict.get('publications', [])
        self.reset_collection_properties(property_dict)

        return self

    @property
    def publication_years(self):
        if self._publication_years is None:
            self._publication_years = sorted(set(self.database.get_pub(pub).year for pub in self.publications))

        return self._publication_years

    @property
    def start_year(self):
        if self._start_year is None:
            self._start_year = min(self.publication_years)
        return self._start_year

    @property
    def end_year(self):
        if self._end_year is None:
            self._end_year = max(self.publication_years)
        return self._end_year

    @property
    def career_length(self):
        if self._career_length is None:
            self._career_length = self.end_year - self.start_year + 1
        return self._career_length

    @property
    def total_productivity(self):
        if self._total_productivity is None:
            self._total_productivity = len(self.publications)
        return self._total_productivity

    @property
    def yearly_productivity(self):
        if self._yearly_productivity is None:
            self._yearly_productivity = Counter(self.database.get_pub(pub).year for pub in self.publications)
        return self._yearly_productivity

    @property
    def annual_productivity(self):
        if self._annual_productivity is None:
            self._annual_productivity = self.total_productivity / self.career_length
        return self._annual_productivity

    @property
    def total_citations(self):
        if self._total_citations is None:
            self._total_citations = sum(self.database.get_pub(pub).total_citations for pub in self.publications)
        return self._total_citations

    @property
    def yearly_citations(self):
        if self._yearly_citations is None:
            self._yearly_citations = sum((self.database.get_pub(pub).yearly_citations for pub in self.publications), Counter())
        return self._yearly_citations

    @property
    def cummulative_yearly_citations(self):
        if self._cummulative_citations is None:
            self._cummulative_citations = {}
            for y in range(self.start_year, max(self.yearly_citations.keys()) + 1):
                self._cummulative_citations[y] = self.yearly_citations.get(y, 0) + self._cummulative_citations.get(y-1, 0.0)
        return self._cummulative_citations

