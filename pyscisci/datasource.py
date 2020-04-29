# -*- coding: utf-8 -*-
"""
.. module:: datasource
    :synopsis: Set of classes to work with different bibliometric data sources

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """
import json
import gzip
import numpy as np
from nameparser import HumanName

from .utils import isin_sorted, zip2dict, load_int, load_float
from .analysis import groupby_count

def load_preprocessed_data(dataname, path2database, columns = None, isindict = None, duplicate_subset = None,
    duplicate_keep = 'last', dropna = None, keep_file = False):

    #TODO: progress bar

    path2files = os.path.join(path2data, dataname)
    if not os.path.exists(path2file):
        # TODO: make a real warning
        print("First preprocess the raw data.")
        return []

    if isinstance(columns, str):
        columns = [columns]

    if isinstance(dropna, str):
        dropna = [dropna]

    if isinstance(duplicate_subset, str):
        duplicate_subset = [duplicate_subset]

    if isinstance(isindict, dict):
        isindict = {isinkey:np.sort(isinlist) for isinkey, isinlist in isindict.items()}


    Nfiles = sum(dataname in fname for fname in os.listdir(path2files))

    data_df = []
    for ifile in range(Nfiles):
        fname = os.path.join(path2files, dataname+"{}.hdf".format(ifile))
        subdf = pd.read_hdf(fname, mode = 'r')

        if isinstance(columns, list):
            subdf = subdf[columns]

        if isinstance(dropna, list):
            subdf.dropna(subset = dropna, inplace = True, how = 'any')

        if isinstance(isindict, dict):
            for isinkey, isinlist in isindict.items():
                subdf = subdf[isin_sorted(subdf[isinkey], isinlist)]

        if isinstance(duplicate_subset, list):
            subdf.drop_duplicates(subset = duplicate_subset, keep = duplicate_keep, inplace = True)

        if keep_file:
            subdf['filetag'] = ifile

        data_df.append(subdf)

    data_df = pd.concat(data_df)

    if isinstance(duplicate_subset, list):
        data_df.drop_duplicates(subset = duplicate_subset, keep = duplicate_keep, inplace = True)

    return data_df

class MAG(object):
    """
    Base class for Microsoft Academic Graph interface.

    The MAG comes structured into three folders: mag, advanced, nlp.
    Explain downloading etc.

    """

    def __init__(self, path2database = ''):

        self.path2database = path2database

        self.affiliation_df = None
        self.pub_df = None
        self.author_df = None
        self.pub2year = None
        self.pub2ref_df = None

    def preprocess(self, dflist = None):
        if dflist is None:
            dflist = ['affiliation', 'author', 'publication', 'reference']

        if 'affiliation' in dflist:
            self.parse_affiliations(preprocess = True)

        if 'author' in dflist:
            self.parse_authors(preprocess = True)

        if 'publication' in dflist:
            self.parse_publications(preprocess = True)

        if 'reference' in dflist:
            self.parse_references(preprocess = True)


    def load_affiliations(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None):

        if preprocess:
            self.affiliation_df = load_preprocessed_data('affiliation', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna)
        else:
            self.parse_affiliations()

    def parse_affiliations(self, preprocess = False):

        affil_int_columns = [0, 7, 8]
        affil_str_columns = [3, 4, 5, 6]
        affil_float_columns = [9, 10]

        affil_column_names = ['AffiliationId', 'NumberPublications', 'NumberCitations', 'FullName', 'GridId', 'OfficialPage', 'WikiPage', 'Latitude', 'Longitude']

        affiliation_info = []
        with open(os.path.join(self.path2database, 'mag', 'Affiliations.txt'), 'r') as infile:
            for line in infile:
                sline = line.replace('\n', '').split('\t')
                affline = [load_int(sline[i]) for i in affil_int_columns]
                affline += [sline[i] for i in affil_str_columns]
                affline += [load_float(sline[i]) for i in affil_float_columns]
                affiliation_info.append(affline)

        self.affiliation_df = pd.DataFrame(affiliation_info, columns = affil_column_names)

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'affiliation')):
                os.mkdir(os.path.join(self.path2database, 'affiliation'))
            self.affiliation_df.to_hdf(os.path.join(self.path2database, 'affiliation', 'affiliation0.hdf'), key = 'affiliation', mode = 'w')
            self.affiliation_df = None

    def load_authors(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None, process_name = True):

        if preprocess:
            self.author_df = load_preprocessed_data('author', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna)
        else:
            self.parse_authors(process_name=process_name)

    def parse_authors(self, preprocess = False, process_name = True):

        author_int_columns = [0, 4, 5, 6]

        author_column_names = ['AuthorId', 'LastKnownAffiliationId', 'NumberPublications', 'NumberCitations', 'FullName']
        if process_name:
            author_column_names += ['LastName', 'FirstName', 'MiddleName']


        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'author')):
                os.mkdir(os.path.join(self.path2database, 'author'))

        iauthor = 0
        ifile = 0
        authorinfo = []
        with open(os.path.join(self.path2database, 'mag', 'Authors.txt'), 'r') as infile:

            for line in infile:
                sline = line.split('\t')
                adata = [load_int(sline[ip]) for ip in author_int_columns] + [sline[2]]
                if process_name:
                    hname = HumanName(unicodedata.normalize('NFD', sline[2]))
                    adata += [hname.last, hname.first, hname.middle]
                authorinfo.append(adata)
                iauthor += 1

                if preprocess and iauthor % (5*10**6) == 0:
                    pd.DataFrame(authorinfo, columns = author_column_names).to_hdf(
                        os.path.join(self.path2database, 'author', 'author{}.hdf'.format(ifile)),
                                                                                key = 'author', mode = 'w')

                    ifile += 1
                    authorinfo = []

            self.author_df = pd.DataFrame(authorinfo, columns = author_column_names)
            if preprocess:
                self.author_df.to_hdf(os.path.join(self.path2database, 'author', 'author{}.hdf'.format(ifile)),
                                                                            key = 'author', mode = 'w')

                self.author_df = None

    def load_publications(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None):

        if preprocess:
            self.pub_df = load_preprocessed_data('publication', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna)
        else:
            self.parse_publications()

    def load_pub2year(self):

        with gzip.open(os.path.join(self.path2database, 'pub2year.json.gz'), 'r') as infile:
            for line in infile:
                pub2year = json.loads(line.decode('utf8'))
        self.pub2year = {int(k):int(v) for k,v in pub2year.items() if not v is None}

    def parse_publications(self, preprocess = False):
        doctype = {'Journal': 'j', 'Book':'b', '':'', 'BookChapter':'bc', 'Conference':'c', 'Dataset':'d', 'Patent':'p', 'Repository':'r'}

        pub_int_columns = [0, 7, 10, 21]
        pub_str_columns = [2, 4, 8, 13, 14]
        pub_column_names = ['PublicationId', 'Year', 'JournalId', 'FamilyId',  'Doi', 'Title', 'Date', 'Volume', 'Issue', 'DocType']

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'publication')):
                os.mkdir(os.path.join(self.path2database, 'publication'))

        ipub = 0
        ifile = 0
        pubinfo = []

        self.pub2year = {}

        with open(os.path.join(self.path2database, 'mag', 'Papers.txt'), 'r') as infile:
            for line in infile:
                sline = line.replace('\n', '').split('\t')
                pline = [load_int(sline[ip]) for ip in pub_int_columns] + [sline[ip] for ip in pub_str_columns] + [doctype[sline[3]]]
                self.pub2year[pline[0]] = pline[1]
                pubinfo.append(pline)
                ipub += 1

                if preprocess and ipub % 10**7 == 0:
                        pd.DataFrame(pubinfo, columns = pub_column_names).to_hdf(
                            os.path.join(self.path2database, 'publication', 'publication{}.hdf'.format(ifile)),
                                                                                    key = 'publication', mode = 'w')

                        ifile += 1
                        pubinfo = []

            self.pub_df = pd.DataFrame(pubinfo, columns = pub_column_names)
            if preprocess:
                self.pub_df.to_hdf(os.path.join(self.path2database, 'publication', 'publications{}.hdf'.format(ifile)),
                                                                                key = 'publication', mode = 'w')

                with gzip.open(os.path.join(self.path2database, 'pub2year.json.gz'), 'w') as outfile:
                    outfile.write(json.dumps(self.pub2year).encode('utf8'))

                self.pub_df = None
                self.pub2year = None


    def parse_references(self, preprocess = False):

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'pub2ref')):
                os.mkdir(os.path.join(self.path2database, 'pub2ref'))

        iref = 0
        ifile = 0
        pub2ref_info = []
        with open(os.path.join(self.path2database, 'mag', 'PaperReferences.txt'), 'r') as infile:
            for line in infile:
                sline = line.replace('\n', '').split('\t')
                pub2ref_info.append([load_int(sline[ip]) for ip in range(2)])
                iref += 1

                if preprocess and iref % 10**8 == 0:
                    pd.DataFrame(pub2ref_info, columns = ['CitingPublicationId', 'CitedPublicationId']).to_hdf(
                        os.path.join(self.path2database, 'pub2ref', 'pub2ref{}.hdf'.format(ifile)),
                                                                                key = 'pub2ref', mode = 'w')

                    ifile += 1
                    pub2ref_info = []

            self.pub2ref_df = pd.DataFrame(pub2ref_info, columns = ['CitingPublicationId', 'CitedPublicationId'])
            if preprocess:
                self.pub2ref_df.to_hdf(os.path.join(self.path2database, 'pub2ref', 'pub2ref{}.hdf'.format(ifile)),
                                                                                key = 'pub2ref', mode = 'w')
                self.pub2ref_df = None

    def load_refdict(self, years=None):
        if years is None:
            years = list(range(1800, 2021))

        elif isinstance(years, int):
            years = [years]

        self.pub2refdict = {}
        for y in years:
            with gzip.open(os.path.join(self.path2database, 'pub2refdict', 'pub2refdict{}.json.gz'.format(y)), 'r') as infile:
                for line in infile:
                    self.pub2refdict[y] = {int(k):citelist for k, reflist in json.loads(line.decode('utf8')).items()}

    def process_refdict(self, preprocess = True):
        year_breakpts = [1800, 1990, 2011, 2016, 2021]

        if self.pub2year is None:
            self.load_pub2year()

        for ibreak in range(len(year_breakpts) - 1):
            start_year = year_breakpts[ibreak]
            end_year = year_breakpts[ibreak - 1]

            pub2ref = {y:{} for y in range(start_year, end_year)}

            Nreffiles = sum('pub2ref' in fname for fname in os.listdir(os.path.join(self.path2database, 'pub2ref')))

            for ifile in range(Nreffiles):

                refdf = pd.read_hdf(os.path.join(self.path2database, 'pub2ref', 'pub2ref{}.hdf'.format(ifile)))

                for citing, cited in refdf.values:

                    citingyear = self.pub2year.get(citing, 0)
                    citedyear = self.pub2year.get(cited, 0)

                    if citingyear >= start_year and citingyear < end_year and citedyear > 0:
                        if pub2ref[citingyear].get(citing, None) is None:
                            pub2ref[citingyear][int(citing)] = [int(cited)]
                        else:
                            pub2ref[citingyear][int(citing)].append(int(cited))

            for y in sorted(pub2ref.keys()):
                with gzip.open(os.path.join(self.path2database, 'pub2refdict', 'pub2refdict{}.json.gz'.format(y)), 'w') as outfile:
                    outfile.write(json.dumps(pub2ref[y]).encode('utf8'))

    def load_citedict(self, years=None):
        if years is None:
            years = list(range(1800, 2021))

        elif isinstance(years, int):
            years = [years]

        self.pub2citedict = {}
        for y in years:
            with gzip.open(os.path.join(self.path2database, 'pub2citedict', 'pub2citedict{}.json.gz'.format(y)), 'r') as infile:
                for line in infile:
                    self.pub2citedict[y] = {int(k):citelist for k, citelist in json.loads(line.decode('utf8')).items()}



    def process_citedict(self, preprocess = True):
        year_breakpts = [1800, 1990, 2011, 2016, 2021]

        if self.pub2year is None:
            self.load_pub2year()

        for ibreak in range(len(year_breakpts) - 1):
            start_year = year_breakpts[ibreak]
            end_year = year_breakpts[ibreak - 1]

            pub2cite = {y:{} for y in range(start_year, end_year)}

            Nreffiles = sum('pub2ref' in fname for fname in os.listdir(os.path.join(self.path2database, 'pub2ref')))

            for ifile in range(Nreffiles):

                refdf = pd.read_hdf(os.path.join(self.path2database, 'pub2ref', 'pub2ref{}.hdf'.format(ifile)))

                for citing, cited in refdf.values:

                    citingyear = self.pub2year.get(citing, 0)
                    citedyear = self.pub2year.get(cited, 0)

                    if citedyear >= start_year and citedyear < end_year and citingyear >= citedyear:
                        if pub2cite[citedyear].get(cited, None) is None:
                            pub2cite[citedyear][int(cited)] = [int(citing)]
                        else:
                            pub2cite[citedyear][int(cited)].append(int(citing))

            for y in sorted(pub2cite.keys()):
                with gzip.open(os.path.join(self.path2database, 'pub2citedict', 'pub2citedict{}.json.gz'.format(y)), 'w') as outfile:
                    outfile.write(json.dumps(pub2cite[y]).encode('utf8'))


    def process_author2affil():

        Nafffiles = sum('publicationauthoraffiliation' in fname for fname in os.listdir(os.path.join(self.path2database, 'publicationauthoraffiliation')) )

        author2affildict = {}
        for ifile in range(Nafffiles):
            affdf = pd.read_hdf(os.path.join(self.path2database, 'publicationauthoraffiliation', 'publicationauthoraffiliation{}.hdf'.format(ifile)))
            affdf.dropna(subset=['AffiliationId'], inplace=True)
            for authid, affid in affdf[['AuthorId', 'AffiliationId']].values:
                affid = str(affid)
                if author2affildict.get(authid, None) is None:
                    author2affildict[authid] = {affid:1}
                elif author2affildict[authid].get(affid, None) is None:
                    author2affildict[authid][affid] = 1
                else:
                    author2affildict[authid][affid] += 1

        Nauthorfiles = sum('author' in fname for fname in os.listdir(os.path.join(self.path2database, 'author')) )
        for ifile in range(Nauthorfiles):
            authordf = pd.read_hdf(os.path.join(self.path2database, 'author', 'author{}.hdf'.format(ifile)))

            with gzip.open(os.path.join(self.path2database, 'authoraffiliation', 'authoraffiliation{}.tsv.gz'.format(ifile)), 'w') as outfile:
                for aid in authordf['AuthorID'].values:
                    outfile.write("{}\t{}\n".format(aid, json.dumps(author2affildict.get(aid, {}))).encode('utf8'))






class WOS(object):
    """
    Base class for Web of Science interface.

    TODO: everything

    """

    def __init__(self, path2database = ''):

        self.path2database = path2database

class Dimensions(object):
    """
    Base class for Dimensions interface.

    TODO: everything

    """

    def __init__(self, path2database = ''):

        self.path2database = path2databases

