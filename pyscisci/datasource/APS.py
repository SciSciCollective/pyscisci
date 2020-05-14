import os
import json
import gzip

import pandas as pd
import numpy as np
from nameparser import HumanName
import requests
from lxml import etree

from pyscisci.datasource.readwrite import load_preprocessed_data, load_int, load_float, load_html_str
from pyscisci.database import BibDataBase

class APS(BibDataBase):
    """
    Base class for APS interface.

    The APS comes as a single xml file.

    You must request usage through their website: https://journals.aps.org/datasets

    """

    def __init__(self, path2database = '', keep_in_memory = False):

        self.path2database = path2database
        self.keep_in_memory = keep_in_memory

        self._affiliation_df = None
        self._pub_df = None
        self._author_df = None
        self._pub2year = None
        self._pub2ref_df = None
        self._pub2refnoself_df = None
        self._author2pub_df = None
        self._paa_df = None

        self.PublicationIdType = int
        self.AffiliationIdType = int
        self.AuthorIdType = str


    def preprocess(self, file_name = 'dblp.xml.gz', process_name = True, num_file_lines=10**7):
        """
        Bulk preprocess the DBLP raw data.

        """
        raise NotImplementedError



    def download_from_source(self, citation_link=None, metadata_link = None):

        pass

    def parse_affiliations(self, preprocess = False):
        raise NotImplementedError("DBLP is stored as a single xml file.  Run preprocess to parse the file.")

    def parse_authors(self, preprocess = False, process_name = True, num_file_lines = 5*10**6):
        raise NotImplementedError("DBLP is stored as a single xml file.  Run preprocess to parse the file.")

    def parse_publications(self, preprocess = False, num_file_lines=10**7):
        raise NotImplementedError("DBLP is stored as a single xml file.  Run preprocess to parse the file.")

    def parse_references(self, preprocess = False, num_file_lines=10**7):
        raise NotImplementedError("DBLP does not contain reference or citation information.")

    def parse_publicationauthoraffiliation(self, preprocess = False, num_file_lines=10**7):
        raise NotImplementedError("DBLP is stored as a single xml file.  Run preprocess to parse the file.")

    def parse_fields(self, preprocess = False, num_file_lines=10**7):
        raise NotImplementedError("DBLP does not contain field information.")

