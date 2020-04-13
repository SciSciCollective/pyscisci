# -*- coding: utf-8 -*-
"""
.. module:: affiliation
    :synopsis: The main Affiliation class

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

from .publicationcollection import PublicationCollection

class Affiliation(PublicationCollection):
    """
    Base class Affiliations

    """

    def __init__(self, database, affiliation_dict = {}):

        self.database = database

        self.from_dict(affiliation_dict)

    def reset_properties(self):

        self.reset_collection_properties()


    def from_dict(self, affiliation_dict):

        self.id = affiliation_dict.get('id', 0)
        self.fullname = affiliation_dict.get('fullname', '')
        self.longitude = affiliation_dict.get('longitude', None)
        self.latitude = affiliation_dict.get('latitude', None)
        self.gridid = affiliation_dict.get('gridid', None)
        self.webpage = affiliation_dict.get('webpage', None)
        self.wikipage = affiliation_dict.get('wikipage', None)
        self.country = affiliation_dict.get('country', None)
        self.address = affiliation_dict.get('address', None)
        self.state = affiliation_dict.get('state', None)

        self.metadata = affiliation_dict.get('metadata', {})

        self.collection_from_dict(affiliation_dict)

        self.authors = set(affiliation_dict.get('authors', []))

