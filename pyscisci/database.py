# -*- coding: utf-8 -*-
"""
.. module:: datasource
    :synopsis: Set of classes to work with different bibliometric data sources

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """
import os
import json
import gzip
from collections import defaultdict

import pandas as pd
import numpy as np
from nameparser import HumanName

from pyscisci.utils import isin_sorted, zip2dict, load_int, load_float, groupby_count
from pyscisci.metrics import *
from pyscisci.datasource.readwrite import load_preprocessed_data, append_to_preprocessed_df


class BibDataBase(object):

    """
    Base class for all bibliometric database interfaces.

    The BibDataBase provides a parasomonious structure for each of the specific data sources (MAG, WOS, etc.).

    There are four primary types of functions:
        1. *Parseing* Functions (data source specific) that parse the raw data files
        2. *Loading* Functions that load DataFrames
        3. *Processing* Functions that calculate advanced data types from the raw files
        4. *Analysis* Functions that calculate Science of Science metrics.

    Parameters
    -------

    :param path2database: str
        The path to the database files

    :param keep_in_memory: bool, default False
        Flag to keep database files in memory once loaded
    """

    def __init__(self, path2database = '', keep_in_memory = False):

        self.path2database = path2database
        self.keep_in_memory = keep_in_memory

        self._affiliation_df = None
        self._pub_df = None
        self._author_df = None
        self._pub2year = None
        self._pub2ref_df = None
        self._author2pub_df = None
        self._paa_df = None
        self._pub2refnoself_df = None

    @property
    def affiliation_df(self):
        """
        The DataFrame keeping affiliation information. Columns may depend on the specific datasource.

        Columns
        -------
        'AffiliationId', 'NumberPublications', 'NumberCitations', 'FullName', 'GridId', 'OfficialPage', 'WikiPage', 'Latitude', 'Longitude'

        """
        if self._affiliation_df is None:
            if self.keep_in_memory:
                self._affiliation_df = self.load_affiliations()
            else:
                return self.load_affiliations()

        return self._affiliation_df


    @property
    def author_df(self):
        """
        The DataFrame keeping author information. Columns may depend on the specific datasource.

        Columns
        -------
        'AuthorId', 'LastKnownAffiliationId', 'NumberPublications', 'NumberCitations', 'FullName', 'LastName', 'FirstName', 'MiddleName'

        """
        if self._author_df is None:
            if self.keep_in_memory:
                self._author_df = self.load_authors()
            else:
                return self.load_authors()

        return self._author_df

    @property
    def pub_df(self):
        """
        The DataFrame keeping publication information. Columns may depend on the specific datasource.

        Columns
        -------
        'PublicationId', 'Year', 'JournalId', 'FamilyId', 'Doi', 'Title', 'Date', 'Volume', 'Issue', 'DocType'

        """
        if self._pub_df is None:
            if self.keep_in_memory:
                self._pub_df = self.load_publications()
            else:
                return self.load_publications()

        return self._pub_df

    @property
    def pub2year(self):
        """
        A dictionary mapping PublicationId to Year.

        Columns
        -------
        'Journal': 'j', 'Book':'b', '':'', 'BookChapter':'bc', 'Conference':'c', 'Dataset':'d', 'Patent':'p', 'Repository':'r'

        """
        if self._pub2year is None:
            if self.keep_in_memory:
                self._pub2year = self.load_pub2year()
            else:
                return self.load_pub2year()

        return self._pub2year

    @property
    def journal_df(self):
        """
        The DataFrame keeping journal information.  Columns may depend on the specific datasource.

        Columns
        -------
        'JournalId', 'FullName', 'Issn', 'Publisher', 'Webpage'

        """
        if self._journal_df is None:
            if self.keep_in_memory:
                self._journal_df = self.load_journals()
            else:
                return self.load_journals()

        return self._journal_df

    @property
    def pub2ref_df(self):
        """
        The DataFrame keeping citing and cited PublicationId.

        Columns
        -------
        CitingPublicationId, CitedPublicationId

        """
        if self._pub2ref_df is None:
            if self.keep_in_memory:
                self._pub2ref_df = self.load_references()
            else:
                return self.load_references()

        return self._pub2ref_df

    @property
    def pub2refnoself_df(self):
        """
        The DataFrame keeping citing and cited PublicationId after filtering out the self-citations.

        Columns
        -------
        CitingPublicationId, CitedPublicationId

        """
        if self._pub2refnoself_df is None:
            if self.keep_in_memory:
                self._pub2refnoself_df = self.load_references(noselfcite=True)
            else:
                return self.load_references(noselfcite=True)

        return self._pub2refnoself_df

    @property
    def paa_df(self):
        """
        The DataFrame keeping all publication, author, affiliation relationships.  Columns may depend on the specific datasource.

        Columns
        -------
        'PublicationId', 'AuthorId', 'AffiliationId', 'AuthorSequence',  'OrigAuthorName', 'OrigAffiliationName'

        """
        if self._paa_df is None:
            if self.keep_in_memory:
                self._paa_df = self.load_publicationauthoraffiliation()
            else:
                return self.load_publicationauthoraffiliation()

        return self._paa_df

    @property
    def author2pub_df(self):
        """
        The DataFrame keeping all publication, author relationships.  Columns may depend on the specific datasource.

        Columns
        -------
        'PublicationId', 'AuthorId'

        """
        if self._paa_df is None:
            if self.keep_in_memory:
                self._paa_df = self.load_publicationauthoraffiliation(columns = ['AuthorId', 'PublicationId'],
                    duplicate_subset = ['AuthorId', 'PublicationId'], dropna = ['AuthorId', 'PublicationId'])
            else:
                return self.load_publicationauthoraffiliation(columns = ['AuthorId', 'PublicationId'],
                    duplicate_subset = ['AuthorId', 'PublicationId'], dropna = ['AuthorId', 'PublicationId'])

        return self._paa_df

    @property
    def pub2field_df(self):
        """
        The DataFrame keeping all publication field relationships.  Columns may depend on the specific datasource.

        Columns
        -------
        'PublicationId', 'FieldId'

        """

        if self._pub2field_df is None:
            if self.keep_in_memory:
                self._pub2field_df = self.load_pub2field()
            else:
                return self.load_pub2field()

        return self._pub2field_df

    @property
    def fieldinfo_df(self):
        """
        The DataFrame keeping all publication field relationships.  Columns may depend on the specific datasource.

        Columns
        -------
        'FieldId', 'FieldLevel', 'NumberPublications', 'FieldName'

        """
        if self._fieldinfo_df is None:
            if self.keep_in_memory:
                self._fieldinfo_df = self.load_fieldinfo()
            else:
                return self.load_fieldinfo()

        return self._fieldinfo_df

    ## Basic Functions for loading data from either preprocessed sources or the raw database files

    def load_affiliations(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None, show_progress=False):
        """
        Load the Affiliation DataFrame from a preprocessed directory, or parse from the raw files.

        Parameters
        ----------
        preprocess : bool, default True, Optional
            Attempt to load from the preprocessed directory.

        columns : list, default None, Optional
            Load only this subset of columns

        isindict : dict, default None, Optional
            Dictionary of format {"ColumnName":"ListofValues"} where "ColumnName" is a data column
            and "ListofValues" is a sorted list of valid values.  A DataFrame only containing rows that appear in
            "ListofValues" will be returned.

        duplicate_subset : list, default None, Optional
            Drop any duplicate entries as specified by this subset of columns

        duplicate_keep : str, default 'last', Optional
            If duplicates are being dropped, keep the 'first' or 'last'
            (see `pandas.DataFram.drop_duplicates <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html>`_)

        dropna : list, default None, Optional
            Drop any NaN entries as specified by this subset of columns

        Returns
        -------
        DataFrame
            Affililation DataFrame.

        """
        if show_progress:
            show_progress='Loading Affiliations'

        if preprocess and os.path.exists(os.path.join(self.path2database, 'affiliation')):
            return load_preprocessed_data('affiliation', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna, show_progress)
        else:
            return self.parse_affiliations()

    def load_authors(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None, process_name = True):
        """
        Load the Author DataFrame from a preprocessed directory, or parse from the raw files.

        Parameters
        ----------
        preprocess : bool, default True, Optional
            Attempt to load from the preprocessed directory.

        columns : list, default None, Optional
            Load only this subset of columns

        isindict : dict, default None, Optional
            Dictionary of format {"ColumnName":"ListofValues"} where "ColumnName" is a data column
            and "ListofValues" is a sorted list of valid values.  A DataFrame only containing rows that appear in
            "ListofValues" will be returned.

        duplicate_subset : list, default None, Optional
            Drop any duplicate entries as specified by this subset of columns

        duplicate_keep : str, default 'last', Optional
            If duplicates are being dropped, keep the 'first' or 'last'
            (see `pandas.DataFram.drop_duplicates <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html>`_)

        dropna : list, default None, Optional
            Drop any NaN entries as specified by this subset of columns

        process_name : bool, default True, Optional
            If True, then when processing the raw file, the package `NameParser <https://nameparser.readthedocs.io/en/latest/>`_
            will be used to split author FullNames.

        Returns
        -------
        DataFrame
            Author DataFrame.

        """
        if show_progress:
            show_progress='Loading Authors'

        if preprocess and os.path.exists(os.path.join(self.path2database, 'author')):
            return load_preprocessed_data('author', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna, show_progress)
        else:
            return self.parse_authors(process_name=process_name)

    def load_publications(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None, show_progress=False):
        """
        Load the Publication DataFrame from a preprocessed directory, or parse from the raw files.

        Parameters
        ----------
        :param preprocess: bool, default True, Optional
            Attempt to load from the preprocessed directory.

        columns : list, default None, Optional
            Load only this subset of columns

        isindict : dict, default None, Optional
            Dictionary of format {"ColumnName":"ListofValues"} where "ColumnName" is a data column
            and "ListofValues" is a sorted list of valid values.  A DataFrame only containing rows that appear in
            "ListofValues" will be returned.

        duplicate_subset : list, default None, Optional
            Drop any duplicate entries as specified by this subset of columns

        duplicate_keep : str, default 'last', Optional
            If duplicates are being dropped, keep the 'first' or 'last'
            (see `pandas.DataFram.drop_duplicates <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html>`_)

        dropna : list, default None, Optional
            Drop any NaN entries as specified by this subset of columns

        Returns
        -------
        DataFrame
            Publication DataFrame.

        """
        if show_progress:
            show_progress='Loading Publications'

        if preprocess and os.path.exists(os.path.join(self.path2database, 'publication')):
            return load_preprocessed_data('publication', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna, show_progress)
        else:
            return self.parse_publications()

    def load_pub2year(self):

        if os.path.exists(os.path.join(self.path2database, 'pub2year.json.gz')):
            with gzip.open(os.path.join(self.path2database, 'pub2year.json.gz'), 'r') as infile:
                pub2year = json.loads(infile.read().decode('utf8'))
            return {self.PublicationIdType(k):int(y) for k,y in pub2year.items() if not y is None}

    def load_journals(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None, show_progress=False):
        """
        Load the Journal DataFrame from a preprocessed directory, or parse from the raw files.

        Parameters
        ----------
        preprocess : bool, default True, Optional
            Attempt to load from the preprocessed directory.

        columns : list, default None, Optional
            Load only this subset of columns

        isindict : dict, default None, Optional
            Dictionary of format {"ColumnName":"ListofValues"} where "ColumnName" is a data column
            and "ListofValues" is a sorted list of valid values.  A DataFrame only containing rows that appear in
            "ListofValues" will be returned.

        duplicate_subset : list, default None, Optional
            Drop any duplicate entries as specified by this subset of columns

        duplicate_keep : str, default 'last', Optional
            If duplicates are being dropped, keep the 'first' or 'last'
            (see `pandas.DataFram.drop_duplicates <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html>`_)

        dropna : list, default None, Optional
            Drop any NaN entries as specified by this subset of columns

        Returns
        -------
        DataFrame
            Journal DataFrame.

        """
        if show_progress:
            show_progress='Loading Journals'

        if preprocess and os.path.exists(os.path.join(self.path2database, 'journal')):
            return load_preprocessed_data('journal', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna, show_progress)
        else:
            return self.parse_publications()

    def load_references(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', noselfcite = False, dropna = None, show_progress=False):
        """
        Load the Pub2Ref DataFrame from a preprocessed directory, or parse from the raw files.

        Parameters
        ----------
        preprocess : bool, default True, Optional
            Attempt to load from the preprocessed directory.

        columns : list, default None, Optional
            Load only this subset of columns

        isindict : dict, default None, Optional
            Dictionary of format {"ColumnName":"ListofValues"} where "ColumnName" is a data column
            and "ListofValues" is a sorted list of valid values.  A DataFrame only containing rows that appear in
            "ListofValues" will be returned.

        duplicate_subset : list, default None, Optional
            Drop any duplicate entries as specified by this subset of columns

        duplicate_keep : str, default 'last', Optional
            If duplicates are being dropped, keep the 'first' or 'last'
            (see `pandas.DataFram.drop_duplicates <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html>`_)

        dropna : list, default None, Optional
            Drop any NaN entries as specified by this subset of columns

        noselfcite : bool, default False, Optional
            If True, then the preprocessed pub2ref files with self-citations removed will be used.

        Returns
        -------
        DataFrame
            Pub2Ref DataFrame.

        """
        if noselfcite:
            fileprefix = 'pub2refnoself'
        else:
            fileprefix = 'pub2ref'

        if show_progress:
            show_progress='Loading {}'.format(fileprefix)

        if preprocess and os.path.exists(os.path.join(self.path2database, fileprefix)):
            return load_preprocessed_data(fileprefix, self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna, show_progress)
        else:
            return self.parse_references()

    def load_publicationauthoraffiliation(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None, show_progress=False):
        """
        Load the PublicationAuthorAffilation DataFrame from a preprocessed directory, or parse from the raw files.

        Parameters
        ----------
        preprocess : bool, default True, Optional
            Attempt to load from the preprocessed directory.

        columns : list, default None, Optional
            Load only this subset of columns

        isindict : dict, default None, Optional
            Dictionary of format {"ColumnName":"ListofValues"} where "ColumnName" is a data column
            and "ListofValues" is a sorted list of valid values.  A DataFrame only containing rows that appear in
            "ListofValues" will be returned.

        duplicate_subset : list, default None, Optional
            Drop any duplicate entries as specified by this subset of columns

        duplicate_keep : str, default 'last', Optional
            If duplicates are being dropped, keep the 'first' or 'last'
            (see `pandas.DataFram.drop_duplicates <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html>`_)

        dropna : list, default None, Optional
            Drop any NaN entries as specified by this subset of columns

        Returns
        -------
        DataFrame
            PublicationAuthorAffilation DataFrame.

        """
        if show_progress:
            show_progress='Loading Publication Author Affiliation'

        if preprocess and os.path.exists(os.path.join(self.path2database, 'publicationauthoraffiliation')):
            return load_preprocessed_data('publicationauthoraffiliation', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna, show_progress)
        else:
            return self.parse_publicationauthoraffiliation()

    def load_pub2field(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None, show_progress=False):
        """
        Load the Pub2Field DataFrame from a preprocessed directory, or parse from the raw files.

        Parameters
        ----------
        :param preprocess : bool, default True, Optional
            Attempt to load from the preprocessed directory.

        :param columns : list, default None, Optional
            Load only this subset of columns

        :param isindict : dict, default None, Optional
            Dictionary of format {"ColumnName":"ListofValues"} where "ColumnName" is a data column
            and "ListofValues" is a sorted list of valid values.  A DataFrame only containing rows that appear in
            "ListofValues" will be returned.

        :param duplicate_subset : list, default None, Optional
            Drop any duplicate entries as specified by this subset of columns

        :param duplicate_keep : str, default 'last', Optional
            If duplicates are being dropped, keep the 'first' or 'last'
            (see `pandas.DataFram.drop_duplicates <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html>`_)

        :param dropna : list, default None, Optional
            Drop any NaN entries as specified by this subset of columns

        Returns
        -------
        DataFrame
            Pub2Field DataFrame.

        """
        if show_progress:
            show_progress='Loading Fields'

        if preprocess and os.path.exists(os.path.join(self.path2database, 'pub2field')):
            return load_preprocessed_data('pub2field', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna, show_progress)
        else:
            return self.parse_fields()

    def load_fields(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None, show_progress=False):
        """
        Load the Field Information DataFrame from a preprocessed directory, or parse from the raw files.

        Parameters
        ----------
        :param preprocess : bool, default True, Optional
            Attempt to load from the preprocessed directory.

        :param columns : list, default None, Optional
            Load only this subset of columns

        :param isindict : dict, default None, Optional
            Dictionary of format {"ColumnName":"ListofValues"} where "ColumnName" is a data column
            and "ListofValues" is a sorted list of valid values.  A DataFrame only containing rows that appear in
            "ListofValues" will be returned.

        :param duplicate_subset : list, default None, Optional
            Drop any duplicate entries as specified by this subset of columns

        :param duplicate_keep : str, default 'last', Optional
            If duplicates are being dropped, keep the 'first' or 'last'
            (see `pandas.DataFram.drop_duplicates <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html>`_)

        :param dropna : list, default None, Optional
            Drop any NaN entries as specified by this subset of columns

        Returns
        -------
        DataFrame
            FieldInformation DataFrame.

        """
        if show_progress:
            show_progress='Loading Field Info'
        if preprocess and os.path.exists(os.path.join(self.path2database, 'fieldinfo')):
            return load_preprocessed_data('fieldinfo', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna, show_progress)
        else:
            return self.parse_fields()

    def load_impact(self, preprocess = True, include_yearnormed = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None, show_progress=False):
        """
        Load the precomputed impact DataFrame from a preprocessed directory.

        Parameters
        ----------
        :param preprocess : bool, default True
            Attempt to load from the preprocessed directory.

        :param include_yearnormed: bool, default True
            Normalize all columns by yearly average.

        :param columns : list, default None
            Load only this subset of columns

        :param isindict : dict, default None, Optional
            Dictionary of format {"ColumnName":"ListofValues"} where "ColumnName" is a data column
            and "ListofValues" is a sorted list of valid values.  A DataFrame only containing rows that appear in
            "ListofValues" will be returned.

        :param duplicate_subset : list, default None, Optional
            Drop any duplicate entries as specified by this subset of columns

        :param duplicate_keep : str, default 'last', Optional
            If duplicates are being dropped, keep the 'first' or 'last'
            (see `pandas.DataFram.drop_duplicates <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html>`_)

        :param dropna : list, default None, Optional
            Drop any NaN entries as specified by this subset of columns

        Returns
        -------
        DataFrame
            FieldInformation DataFrame.

        """
        if show_progress:
            show_progress='Loading Impact'

        if include_yearnormed:
            def normfunc(impactdf):
                impactcolumns = [c for c in list(impactdf) if not c in ['PublicationId', 'Year']]
                for c in impactcolumns:
                    impactdf[c+'_norm'] = impactdf[c]/impactdf[c].mean()
                return impactdf
        else:
            def normfunc(impactdf):
                return impactdf
        if preprocess and os.path.exists(os.path.join(self.path2database, 'impact')):
            return load_preprocessed_data('impact', self.path2database, columns, isindict, duplicate_subset,
                duplicate_keep, dropna, prefunc2apply=normfunc, show_progress=show_progress)
        else:
            raise self.compute_impact()

    """
    To be rewritten for each specific data source (MAG, WOS, etc.)
    """

    def download_from_source(self):
        raise NotImplementedError

    def parse_affiliations(self, preprocess = False):
        raise NotImplementedError

    def parse_authors(self, preprocess = False, process_name = True, num_file_lines = 5*10**6):
        raise NotImplementedError

    def parse_publications(self, preprocess = False, num_file_lines=10**7):
        raise NotImplementedError

    def parse_references(self, preprocess = False, num_file_lines=10**7):
        raise NotImplementedError

    def parse_publicationauthoraffiliation(self, preprocess = False, num_file_lines=10**7):
        raise NotImplementedError

    def parse_fields(self, preprocess = False, num_file_lines=10**7):
        raise NotImplementedError


    # Analysis

    def author_productivity(self, df=None, colgroupby = 'AuthorId', colcountby = 'PublicationId', show_progress=False):
        """
        Calculate the total number of publications for each author.

        Parameters
        ----------
        :param df : DataFrame, default None, Optional
            A DataFrame with the author2publication information.  If None then the database 'author2pub_df' is used.

        :param colgroupby : str, default 'AuthorId', Optional
            The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

        :param colcountby : str, default 'PublicationId', Optional
            The DataFrame column with Publication Ids.  If None then the database 'PublicationId' is used.


        Returns
        -------
        DataFrame
            Productivity DataFrame with 2 columns: 'AuthorId', 'Productivity'

        """
        if df is None:
            df = self.author2pub_df

        # we can use show_progress to pass a label for the progress bar
        if show_progress:
            show_progress='Author Productivity'

        newname_dict = zip2dict([str(colcountby)+'Count', '0'], ['Productivity']*2)
        return groupby_count(df, colgroupby, colcountby, count_unique=True, show_progress=show_progress).rename(columns=newname_dict)

    def author_yearly_productivity(self, df=None, colgroupby = 'AuthorId', datecol = 'Year', colcountby = 'PublicationId', show_progress=False):
        """
        Calculate the number of publications for each author in each year.

        Parameters
        ----------
        :param df : DataFrame, default None, Optional
            A DataFrame with the author2publication information.  If None then the database 'author2pub_df' is used.

        :param colgroupby : str, default 'AuthorId', Optional
            The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

        :param datecol : str, default 'Year', Optional
            The DataFrame column with Year information.  If None then the database 'Year' is used.

        :param colcountby : str, default 'PublicationId', Optional
            The DataFrame column with Publication Ids.  If None then the database 'PublicationId' is used.

        Returns
        -------
        DataFrame
            Productivity DataFrame with 3 columns: 'AuthorId', 'Year', 'YearlyProductivity'

        """
        if df is None:
            df = self.author2pub_df

        # we can use show_progress to pass a label for the progress bar
        if show_progress:
            show_progress='Yearly Productivity'

        newname_dict = zip2dict([str(colcountby)+'Count', '0'], ['YearlyProductivity']*2)
        return groupby_count(df, [colgroupby, datecol], colcountby, count_unique=True, show_progress=show_progress).rename(columns=newname_dict)

    def author_career_length(self, df = None, colgroupby = 'AuthorId', colrange = 'Year', show_progress=False):
        """
        Calculate the career length for each author.  The career length is the length of time from the first
        publication to the last publication.

        Parameters
        ----------
        :param df : DataFrame, default None, Optional
            A DataFrame with the author2publication information.  If None then the database 'author2pub_df' is used.

        :param colgroupby : str, default 'AuthorId', Optional
            The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

        :param colrange : str, default 'Year', Optional
            The DataFrame column with Date information.  If None then the database 'Year' is used.

        Returns
        -------
        DataFrame
            Productivity DataFrame with 2 columns: 'AuthorId', 'CareerLength'

        """
        if df is None:
            df = self.author2pub_df

        # we can use show_progress to pass a label for the progress bar
        if show_progress:
            show_progress='Career Length'

        newname_dict = zip2dict([str(colrange)+'Range', '0'], ['CareerLength']*2)
        return groupby_range(df, colgroupby, colrange, show_progress=show_progress).rename(columns=newname_dict)

    def author_productivity_trajectory(self, df =None, colgroupby = 'AuthorId', datecol = 'Year', colcountby = 'PublicationId', show_progress=False):
        """
        Calculate the author yearly productivity trajectory.  See :cite:`way2017misleading`

        The algorithmic implementation can be found in :py:func:`citationanalysis.compute_yearly_productivity_traj`.

        Parameters
        ----------
        :param df : DataFrame, default None
            A DataFrame with the author2publication information.  If None then the database 'author2pub_df' is used.

        :param colgroupby : str, default 'AuthorId'
            The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

        :param datecol : str, default 'Year'
            The DataFrame column with Date information.  If None then the database 'Year' is used.

        :param colcountby : str, default 'PublicationId'
            The DataFrame column with Publication Ids.  If None then the database 'PublicationId' is used.

        Returns
        -------
        DataFrame
            Trajectory DataFrame with 5 columns: 'AuthorId', 't_break', 'b', 'm1', 'm2'

        """
        if df is None:
            df = self.author2pub_df

        return compute_yearly_productivity_traj(df, colgroupby = colgroupby)

    def author_hindex(self, df = None, colgroupby = 'AuthorId', colcountby = 'Ctotal', show_progress=False):
        """
        Calculate the author yearly productivity trajectory.  See :cite:`hirsch2005index` for the derivation.

        The algorithmic implementation can be found in :py:func:`citationanalysis.compute_hindex`.

        Parameters
        ----------
        :param df : DataFrame, default None, Optional
            A DataFrame with the author2publication information.  If None then the database 'author2pub_df' is used.

        :param colgroupby : str, default 'AuthorId', Optional
            The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

        :param colcountby : str, default 'Ctotal', Optional
            The DataFrame column with Citation counts for each publication.  If None then the database 'Ctotal' is used.

        Returns
        -------
        DataFrame
            Trajectory DataFrame with 2 columns: 'AuthorId', 'Hindex'

        """
        if show_progress: print("Starting Author H-index. \nLoading Data.")

        if df is None:
            df = self.author2pub_df.merge(self.impact_df[['AuthorId', colcountby]], on='PublicationId', how='left')

        if show_progress: print("Computing H-index.")
        return compute_hindex(df, colgroupby = colgroupby, colcountby = colcountby, show_progress=show_progress)


    def remove_selfcitations(self, preprocess=True, show_progress=False):
        """
        Prcoess the pub2ref DataFrame and remove all citation relationships that share an Author.

        Parameters
        ----------
        :param preprocess : bool, default True, Optional
            If True then the new preprocessed DataFrames are saved in pub2refnoself

        Returns
        -------
        DataFrame
            Pub2Ref DataFrame with 2 columns: 'CitingPublicationId', 'CitedPublicationId'

        """
        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'pub2refnoself')):
                os.mkdir(os.path.join(self.path2database, 'pub2refnoself'))

        pub2authors = defaultdict(set)
        for pid, aid in self.author2pub_df[['PublicationId', 'AuthorId']].values:
            pub2authors[pid].add(aid)

        fullrefdf = []
        # loop through all pub2ref files
        Nreffiles = sum('pub2ref' in fname for fname in os.listdir(os.path.join(self.path2database, 'pub2ref')))
        for ifile in tqdm(range(Nreffiles), desc='Removing Self-citations', disable= not show_progress):
            refdf = pd.read_hdf(os.path.join(self.path2database, 'pub2ref', 'pub2ref{}.hdf'.format(ifile)))

            # get citing cited pairs with no common authors
            noselfcite = np.array([len(pub2authors[citingpid] & pub2authors[citedpid]) == 0 for citingpid, citedpid in refdf.values])

            # keep only citing-cited pairs without a common author
            refdf = refdf.loc[noselfcite]

            if preprocess:
                refdf.to_hdf(os.path.join(self.path2database, 'pub2refnoself', 'pub2refnoself{}.hdf'.format(ifile)), key = 'pub2ref')
            else:
                fullrefdf.append(refdf)

        if not preprocess:
            return pd.concat(fullrefdf)


    def compute_impact(self, preprocess=True, citation_horizons = [5,10], noselfcite = True):
        """
        Calculate several of the common citation indices.
            * 'Ctotal' : The total number of citations.
            * 'Ck' : The total number of citations within the first k years of publcation, for each k value specified by `citation_horizons`.
            * 'Ctotal_noself' : The total number of citations with self-citations removed.
            * 'Ck' : The total number of citations within the first k years of publcation with self-citations removed, for each k value specified by `citation_horizons`.

        Parameters
        ----------
        :param preprocess : bool, default True, Optional
            If True then the impact measures are saved in preprocessed files.

        :param citation_horizons : list, default [5,10], Optional
            The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

        :param noselfcite : Bool, default 'True', Optional
            If True then the noselfcitation pub2ref files are also processed.

        Returns
        -------
        DataFrame
            The impact DataFrame with at least two columns: 'PublicationId', 'Year', + citation columns

        """

        # first load the publication year information
        pub2year = self.load_pub2year()

        # now get the reference list and merge with year info
        pub2ref = self.pub2ref_df

        # drop all citations that happend before the publication year
        pub2ref = pub2ref.loc[[pub2year.get(citingpid, 0) >= pub2year.get(citedpid, 0) for citingpid, citedpid in pub2ref[['CitingPublicationId', 'CitedPublicationId']].values]]

        # calcuate the total citations
        citation_df = groupby_count(pub2ref, colgroupby='CitedPublicationId', colcountby='CitingPublicationId', unique=True )
        citation_df.rename(columns={'CitingPublicationIdCount':'Ctotal', 'CitedPublicationId':'PublicationId'}, inplace=True)

        # go from the larest k down
        for k in np.sort(citation_horizons)[::-1]:

            # drop all citations that happend after the k
            #pub2ref = pub2ref.loc[pub2ref['CitingPublicationYear'] <= pub2ref['CitedPublicationYear'] + k]
            pub2ref = pub2ref.loc[[pub2year.get(citingpid, 0) <= pub2year.get(citedpid, 0) + k for citingpid, citedpid in pub2ref[['CitingPublicationId', 'CitedPublicationId']].values]]

            # recalculate the impact
            k_citation_df = groupby_count(pub2ref, colgroupby='CitedPublicationId', colcountby='CitingPublicationId', unique=True )
            k_citation_df.rename(columns={'CitingPublicationIdCount':'C{}'.format(k), 'CitedPublicationId':'PublicationId'}, inplace=True)

            citation_df = citation_df.merge(k_citation_df, how='left', on='PublicationId')

        # get the Cited Year
        citation_df['Year'] = [pub2year.get(pid, 0) for pid in citation_df['PublicationId'].values]


        if noselfcite:
            del pub2ref
            pub2ref = self.pub2refnoself_df

            # drop all citations that happend before the publication year
            pub2ref = pub2ref.loc[[pub2year.get(citingpid, 0) >= pub2year.get(citedpid, 0) for citingpid, citedpid in pub2ref[['CitingPublicationId', 'CitedPublicationId']].values]]

            # calcuate the total citations
            citation_noself_df = groupby_count(pub2ref, colgroupby='CitedPublicationId', colcountby='CitingPublicationId', unique=True )
            citation_noself_df.rename(columns={'CitingPublicationIdCount':'Ctotal_noself', 'CitedPublicationId':'PublicationId'}, inplace=True)

            # go from the larest k down
            for k in np.sort(citation_horizons)[::-1]:

                # drop all citations that happend after the k
                #pub2ref = pub2ref.loc[pub2ref['CitingPublicationYear'] <= pub2ref['CitedPublicationYear'] + k]
                pub2ref = pub2ref.loc[[pub2year.get(citingpid, 0) <= pub2year.get(citedpid, 0) + k for citingpid, citedpid in pub2ref[['CitingPublicationId', 'CitedPublicationId']].values]]

                # recalculate the impact
                k_citation_df = groupby_count(pub2ref, colgroupby='CitedPublicationId', colcountby='CitingPublicationId', unique=True )
                k_citation_df.rename(columns={'CitingPublicationIdCount':'C{}_noself'.format(k), 'CitedPublicationId':'PublicationId'}, inplace=True)

                citation_noself_df = citation_noself_df.merge(k_citation_df, how='left', on='PublicationId')

        citation_df = citation_df.merge(citation_noself_df, how='left', on='PublicationId')

        # set all nan to 0
        citation_df.fillna(0, inplace=True)

        if preprocess:

            if not os.path.exists(os.path.join(self.path2database, 'impact')):
                os.mkdir(os.path.join(self.path2database, 'impact'))

            for y, cdf in citation_df.groupby('Year', sort=True):
                cdf.to_hdf(os.path.join(self.path2database, 'impact', 'impact{}.hdf'.format(y)), mode='w', key ='impact')

        else:
            return citation_df




    def compute_teamsize(self, save2pubdf = True, show_progress=False):
        """
        Calculate the teamsize of publications, defined as the total number of Authors on the publication.

        Parameters
        ----------
        :param save2pubdf : bool, default True, Optional
            If True the results are appended to the preprocessed publication DataFrames.

        :param show_progress: bool, default False
            If True, display a progress bar for the count.

        Returns
        -------
        DataFrame
            TeamSize DataFrame with 2 columns: 'PublicationId', 'TeamSize'

        """
        if show_progress: print("Starting TeamSize computation. \nLoading Data.")
        pub2authorseq_df = self.load_publicationauthoraffiliation(columns = ['PublicationId', 'AuthorId', 'AuthorSequence'],
            duplicate_subset = ['PublicationId', 'AuthorId'], dropna = ['PublicationId', 'AuthorId'])

        # registar our pandas apply with tqdm for a progress bar
        tqdm.pandas(desc='TeamSize', disable= not show_progress)

        pub2teamsize = pub2authorseq_df.groupby('PublicationId', sort=False)['AuthorSequence'].progress_apply(lambda x: x.max()).astype(int).to_frame().reset_index().rename(columns={'AuthorSequence':'TeamSize'})

        if save2pubdf:
            if show_progress: print("Saving Teamsize.")
            append_to_preprocessed_df(pub2teamsize, self.path2database, 'publication')

        return pub2teamsize


    def compute_yearly_citations(self, preprocess = True, show_progress=False):

        if show_progress:
            print("Starting Computation of Yearly Citations")

        # first load the publication year information
        pub2year = self.pub2year

        # now get the reference list and merge with year info
        pub2ref = self.pub2ref_df

        pub2ref['CitingYear'] = [pub2year.get(citingpid, 0) for citingpid in pub2ref['CitingPublicationId'].values]

        # drop all citations that happend before the publication year
        pub2ref = pub2ref.loc[[citingyear >= pub2year.get(citedpid, 0) for citingyear, citedpid in pub2ref[['CitingYear', 'CitedPublicationId']].values]]

        if show_progress:
            print("Yearly Citation Data Prepared")

        # calcuate the total citations
        citation_df = groupby_count(pub2ref, colgroupby=['CitedPublicationId', 'CitingYear'], colcountby='CitingPublicationId', unique=True )
        citation_df.rename(columns={'CitingPublicationIdCount':'YearlyCitations', 'CitedPublicationId':'PublicationId'}, inplace=True)

        # get the Cited Year
        citation_df['CitedYear'] = [pub2year.get(pid, 0) for pid in citation_df['PublicationId'].values]

        citation_df.sort_values(by=['CitedYear', 'CitedPublicationId', 'CitingYear'], inplace=True)

        if show_progress:
            print("Yearly Citations Found")

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'temporalimpact')):
                os.mkdir(os.path.join(self.path2database, 'temporalimpact'))

            for y, cdf in citation_df.groupby('CitedYear', sort=True):
                cdf.to_hdf(os.path.join(self.path2database, 'temporalimpact', 'temporalimpact{}.hdf'.format(y)), mode='w', key ='temporalimpact')

            if show_progress:
                print("Yearly Citations Saved")

        else:
            return citation_df

    def filter_doctypes(self, doctypes = ['j', 'b', 'bc', 'c'], show_progress=False):

        """
        Filter all of the publication files keeping only the publications of specified doctype.

        :param list doctypes: optional
            the list of doctypes

        :return None:

        """
        doctypes = np.sort(doctypes)

        if show_progress: print("Starting DocType filter. \nFiltering Publications.")


        valid_pubids = []
        pub2year = {}
        pub2doctype = {}
        Nfiles = sum('publication' in fname for fname in os.listdir(os.path.join(self.path2database, 'publication')))
        for ifile in range(Nfiles):
            pubdf = pd.read_hdf(os.path.join(self.path2database, 'publication', 'publication{}.hdf'.format(ifile)))
            pubdf.loc[isin_sorted(pubdf['DocType'].values, doctypes)]
            pubdf.dropna(subset=['Year'], inplace=True)
            pubdf['Year'] = pubdf['Year'].astype(int)
            pubdf.to_hdf(os.path.join(self.path2database, 'publication', 'publication{}.hdf'.format(ifile)), key='pub', mode='w')
            valid_pubids.extend(pubdf['PublicationId'].values)
            for pid, y, dt in pubdf[['PublicationId', 'Year', 'DocType']].values:
                pub2year[pid] = y
                pub2doctype[pid] = dt

        with gzip.open(os.path.join(self.path2database, 'pub2year.json.gz'), 'w') as outfile:
            outfile.write(json.dumps(pub2year).encode('utf8'))

        with gzip.open(os.path.join(self.path2database, 'pub2doctype.json.gz'), 'w') as outfile:
            outfile.write(json.dumps(pub2doctype).encode('utf8'))

        del pubdf

        valid_pubids = np.sort(valid_pubids)

        if show_progress: print("Filtering References.")

        Nfiles = sum('pub2ref' in fname for fname in os.listdir(os.path.join(self.path2database, 'pub2ref')))
        for ifile in range(Nfiles):
            pub2refdf = pd.read_hdf(os.path.join(self.path2database, 'pub2ref', 'pub2ref{}.hdf'.format(ifile)))
            pub2refdf = pub2refdf.loc[isin_sorted(pub2refdf['CitedPublicationId'].values, valid_pubids)]
            pub2refdf = pub2refdf.loc[isin_sorted(pub2refdf['CitingPublicationId'].values, valid_pubids)]
            pub2refdf.to_hdf(os.path.join(self.path2database, 'pub2ref', 'pub2ref{}.hdf'.format(ifile)),
                key='pub2ref', mode='w')

        if show_progress: print("Filtering Publication and Author.")

        Nfiles = sum('publicationauthoraffiliation' in fname for fname in os.listdir(os.path.join(self.path2database, 'publicationauthoraffiliation')))
        for ifile in range(Nfiles):
            paa_df = pd.read_hdf(os.path.join(self.path2database, 'publicationauthoraffiliation', 'publicationauthoraffiliation{}.hdf'.format(ifile)))
            paa_df = paa_df.loc[isin_sorted(paa_df['PublicationId'].values, valid_pubids)]
            paa_df.to_hdf(os.path.join(self.path2database, 'publicationauthoraffiliation', 'publicationauthoraffiliation{}.hdf'.format(ifile)),
                key='paa', mode='w')

        if show_progress: print("Finished filtering DocType.")



