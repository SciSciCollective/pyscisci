import os
import sys
import json
import gzip
import zipfile

import pandas as pd
import numpy as np
from nameparser import HumanName
import requests
from lxml import etree

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

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
        self._journal_df = None
        self._author_df = None
        self._pub2year = None
        self._pub2ref_df = None
        self._pub2refnoself_df = None
        self._author2pub_df = None
        self._paa_df = None
        self._pub2field_df=None
        self._fieldinfo_df=None

        self.PublicationIdType = int
        self.AffiliationIdType = int
        self.AuthorIdType = str


    def preprocess(self, archive_year=2019, pubid2int=False, metadata_archive=None, citation_archive=None, show_progress=True):
        """
        Bulk preprocess the APS raw data.

        """
        if metadata_archive is None:
            metadata_archive = 'aps-dataset-metadata-{}.zip'.format(archive_year)

        self.parse_publications(preprocess=True, preprocess_dicts=True, pubid2int=pubid2int,
            archive_name = metadata_archive, show_progress=show_progress)

        if citation_archive is None:
            citation_archive = 'aps-dataset-citations-{}.zip'.format(archive_year)

        self.parse_references(preprocess=True, pubid2int=pubid2int, archive_name=citation_archive, show_progress=show_progress)



    def download_from_source(self):

        import webbrowser
        webbrowser.open("https://journals.aps.org/datasets")

        raise NotImplementedError("APS is shared by request from the American Physical Society.")


    def parse_affiliations(self, preprocess = False):
        raise NotImplementedError("APS is stored as a json archive.  Run preprocess to parse the archive.")

    def parse_authors(self, preprocess = False, process_name = True, num_file_lines = 5*10**6):
        raise NotImplementedError("APS does not contain disambiguated author information.")

    def parse_publications(self, preprocess=False, preprocess_dicts=True, pubid2int=False,
        archive_name = 'aps-dataset-metadata-2019.zip', show_progress=False):

        archive = zipfile.ZipFile(os.path.join(self.path2database, archive_name), 'r')
        metadata_files = [fname for fname in archive.namelist() if 'aps-dataset-metadata' in fname and '.json' in fname]

        # check that the archive concatins the expected directory
        if len(metadata_files) > 0:

            if preprocess:
                if not os.path.exists(os.path.join(self.path2database, 'publication')):
                    os.mkdir(os.path.join(self.path2database, 'publication'))

                if not os.path.exists(os.path.join(self.path2database, 'journal')):
                    os.mkdir(os.path.join(self.path2database, 'journal'))

                if not os.path.exists(os.path.join(self.path2database, 'affiliation')):
                    os.mkdir(os.path.join(self.path2database, 'affiliation'))

                if not os.path.exists(os.path.join(self.path2database, 'publicationauthoraffiliation')):
                    os.mkdir(os.path.join(self.path2database, 'publicationauthoraffiliation'))

                if not os.path.exists(os.path.join(self.path2database, 'pub2field')):
                    os.mkdir(os.path.join(self.path2database, 'pub2field'))

                if not os.path.exists(os.path.join(self.path2database, 'fieldinfo')):
                    os.mkdir(os.path.join(self.path2database, 'fieldinfo'))


            journal_dict = {}
            journal_column_names = ['JournalId', 'FullName', 'AbbreviatedName', 'Publisher']

            pub_column_names = ['PublicationId', 'Title', 'Date', 'Year', 'Doi', 'JournalId', 'Volume', 'Issue', 'PageStart', 'PageEnd', 'DocType', 'TeamSize']

            pub_df = []
            pub2year = {}
            pub2doctype = {}
            pub2int = {}
            ipub = 0
            if pubid2int:
                pubintcol = ['PublicationId']
            else:
                pubintcol = []

            iaff = 0
            affil_dict = {}
            paa_df = []

            field_dict = {}
            pub2field_df = []

            for fname in tqdm(metadata_files, desc='aps-metadata', leave=True, disable=not show_progress):
                # load pub json
                pubjson = json.loads(archive.read(fname).decode('utf-8'))
                ipub += 1

                # start parsing publication information
                if pubid2int:
                    pubid = ipub
                    pub2int[pubjson.get('id', '')] = pubid
                else:
                    pubid = pubjson.get('id', '')
                pubinfo = [pubid]
                pubinfo.append(pubjson.get('title', {}).get('value', ''))
                pubinfo.append(pubjson.get('date', ''))
                pubinfo.append(load_int(pubjson.get('date', '').split('-')[0]))
                pub2year[pubid] = pubinfo[-1]
                pubinfo.append(pubjson.get('id', ''))

                # journal of publication
                journalid = pubjson.get('journal', {}).get('id', '')
                pubinfo.append(journalid)
                pubinfo.append(load_int(pubjson.get('volume', {}).get('number', '')))
                pubinfo.append(load_int(pubjson.get('issue', {}).get('number', '')))

                # add pagenumber info
                pubinfo.append(load_int(pubjson.get('pageStart', '')))
                if not pubjson.get('pageEnd', None) is None:
                    pubinfo.append(load_int(pubjson.get('pageEnd', '')))
                elif not (pubjson.get('numPages', None) is None or pubjson.get('pageStart', None) is None):
                    pubinfo.append(pubinfo[-1] + load_int(pubjson.get('numPages', '')))
                else:
                    pubinfo.append(None)

                # add the doctype
                pubinfo.append(pubjson.get('articleType', ''))
                pub2doctype[pubid] = pubinfo[-1]

                # calculate TeamSize
                pubinfo.append(len(pubjson.get('authors', [])))

                # finish publication infor
                pub_df.append(pubinfo)

                # check if we need to save journal information
                if journal_dict.get(journalid, None) is None:
                    journal_dict[journalid] = pubjson.get('journal', {})
                    journal_dict[journalid]['Publisher'] = pubjson.get('rights', {}).get('copyrightHolders', [{'name':''}])[0].get('name', '')

                # start parsing affiliation information
                pub_affid_map = {}
                for pubaffdict in pubjson.get('affiliations', []):
                    # check if the affiliation has been used before (only using string match)
                    # ToDo: add disambigation
                    if affil_dict.get(pubaffdict.get('name', ''), None) is None:
                        affil_dict[pubaffdict.get('name', '')] = iaff
                        iaff += 1

                    # map the affiliation to the AffiliationId
                    pub_affid_map[pubaffdict.get('id', '')] = affil_dict[pubaffdict.get('name', '')]

                authorseq = 1
                # now start parsing author information
                for authordict in pubjson.get('authors', []):
                    for affid in authordict.get('affiliationIds', [None]):
                        paa_df.append([pubid, authordict.get('name', ''), pub_affid_map.get(affid, None), authorseq])

                    authorseq += 1


                # now do the subject classifications
                for subjectdict in pubjson.get('classificationSchemes', {}).get('subjectAreas', []):
                    pub2field_df.append([pubid, subjectdict.get('id', None)])

                    if field_dict.get(subjectdict.get('id', None), None) is None:
                        field_dict[subjectdict.get('id', None)] = subjectdict.get('label', None)

                # ToDo: parse concepts

            if show_progress:
                print("Parsing Complete\nSaving Publication DataFrames")

            pub_df = pd.DataFrame(pub_df, columns = pub_column_names)
            for intcol in pubintcol + ['Year']:
                pub_df[intcol] = pub_df[intcol].astype(int)

            journal_rename_dict = {'name':'FullName', 'id':'JournalId', 'abbreviatedName':'AbbreviatedName'}
            journal_df = pd.DataFrame(journal_dict.values()).rename(columns=journal_rename_dict)

            affiliation_df = pd.DataFrame([[affid, name] for name, affid in affil_dict.items()], columns = ['AffiliationId', 'Address'])

            paa_df = pd.DataFrame(paa_df, columns = ['PublicationId', 'OrigAuthorName', 'AffiliationId', 'AuthorSequence'])
            for intcol in pubintcol+['AuthorSequence']:
                paa_df[intcol] = paa_df[intcol].astype(int)

            pub2field_df = pd.DataFrame(pub2field_df, columns = ['PublicationId', 'FieldId'])
            for intcol in pubintcol:
                pub2field_df[intcol] = pub2field_df[intcol].astype(int)

            field_df = pd.DataFrame([[fieldid, fieldname] for fieldid, fieldname in field_dict.items()], columns = ['FieldId', 'FullName'])

            if preprocess:
                pub_df.to_hdf(os.path.join(self.path2database, 'publication', 'publication0.hdf'), mode='w', key='publication')

                if pubid2int:
                    with gzip.open(os.path.join(self.path2database, 'pub2int.json.gz'), 'w') as outfile:
                        outfile.write(json.dumps(pub2int).encode('utf8'))

                if preprocess_dicts:
                    with gzip.open(os.path.join(self.path2database, 'pub2year.json.gz'), 'w') as outfile:
                        outfile.write(json.dumps(pub2year).encode('utf8'))

                    with gzip.open(os.path.join(self.path2database, 'pub2doctype.json.gz'), 'w') as outfile:
                        outfile.write(json.dumps(pub2doctype).encode('utf8'))


                journal_df.to_hdf(os.path.join(self.path2database, 'journal', 'journal0.hdf'), mode='w', key='journal')

                affiliation_df.to_hdf(os.path.join(self.path2database, 'affiliation', 'affiliation0.hdf'), mode='w', key='affiliation')

                paa_df.to_hdf(os.path.join(self.path2database, 'publicationauthoraffiliation', 'publicationauthoraffiliation0.hdf'), mode='w', key='publicationauthoraffiliation')

                pub2field_df.to_hdf( os.path.join(self.path2database, 'pub2field', 'pub2field0.hdf'), mode='w', key='pub2field')

                field_df.to_hdf( os.path.join(self.path2database, 'fieldinfo', 'fieldinfo0.hdf'), mode='w', key='pub2field')

        else:
            raise FileNotFoundError('The archive {0} does not contain a metadata directory: {1}.'.format(archive_name, 'aps-dataset-metadata'))

    def parse_references(self, preprocess=False, pubid2int=False, archive_name='aps-dataset-citations-2019.zip', show_progress=False):

        if preprocess and not os.path.exists(os.path.join(self.path2database, 'pub2ref')):
            os.mkdir(os.path.join(self.path2database, 'pub2ref'))

        if pubid2int:
            with gzip.open(os.path.join(self.path2database, 'pub2int.json.gz'), 'r') as infile:
                pub2int = json.loads(infile.read().decode('utf8'))

            def pub2int_map(doi):
                if pub2int.get(doi, None) is None:
                    pub2int[doi] = len(pub2int) + 1

                return pub2int[doi]


        rename_dict = {'citing_doi':'CitingPublicationId', 'cited_doi':'CitedPublicationId'}

        archive = zipfile.ZipFile(os.path.join(self.path2database, archive_name), 'r')

        citation_file = [fname for fname in archive.namelist() if 'aps-dataset-citations' in fname]
        if len(citation_file) > 0:
            citation_file = citation_file[0]
            csvlines = archive.read(citation_file).decode('utf-8').split('\n')

            pub2ref = [line.split(',') for line in tqdm(csvlines, desc='aps-citations', leave=True, disable=not show_progress)]

            pub2ref = pd.DataFrame(pub2ref[1:], columns = pub2ref[0]).rename(columns=rename_dict)
            if pubid2int:
                pub2ref['CitingPublicationId'] = [pub2int_map(pid) for pid in pub2ref['CitingPublicationId'].values]
                pub2ref['CitedPublicationId'] = [pub2int_map(pid) for pid in pub2ref['CitedPublicationId'].values]

                with gzip.open(os.path.join(self.path2database, 'pub2int.json.gz'), 'w') as outfile:
                        outfile.write(json.dumps(pub2int).encode('utf8'))

            if preprocess:
                pub2ref.to_hdf(os.path.join(self.path2database, 'pub2ref', 'pub2ref0.hdf'), mode='w', key = 'pub2ref')

            return pub2ref

        else:
            raise FileNotFoundError('The archive {0} does not contain a citation file: {1}.'.format(archive_name, 'aps-dataset-citations'))

    def parse_publicationauthoraffiliation(self, preprocess = False, num_file_lines=10**7):
        raise NotImplementedError("APS is stored as a json archive.  Run preprocess to parse the archive.")

    def parse_fields(self, preprocess = False, num_file_lines=10**7):
        raise NotImplementedError("APS is stored as a json archive.  Run preprocess to parse the archive.")

