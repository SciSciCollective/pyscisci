
import os
import sys
import json
import gzip

import pandas as pd
import numpy as np
from nameparser import HumanName
import unicodedata

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from pyscisci.datasource.readwrite import load_preprocessed_data, load_int, load_float, load_bool
from pyscisci.database import BibDataBase
from pyscisci.nlp import abstractindex2text

openalex_works_dfset = {'publications', 'references', 'publicationauthoraffiliation', 'concepts', 'fields', 'topics', 'abstracts', 'grants', 'text'}
openalex_dataframe_set = {'publications', 'works', 'sources', 'authors', 'institutions', 'affiliations', 'concepts', 'fields', 'topics', 'funders'}

class OpenAlex(BibDataBase):
    """
    Base class for the OpenAlex interface.

    This is an extension of 'BibDataBase' with processing functions specific to the MAG.
    See 'BibDataBase' in database.py for details of non-MAG specific functions.

    The MAG comes structured into three folders: mag, advanced, nlp.
    Explain downloading etc.

    """

    def __init__(self, path2database = '', database_extension='csv.gz', keep_in_memory = False, global_filter=None, 
        enable_dask=False, show_progress=True):

        self._default_init(path2database, database_extension, keep_in_memory, global_filter, enable_dask, show_progress)

        self.PublicationIdType = int
        self.AffiliationIdType = int
        self.AuthorIdType = int
        self.JournalIdType = int

        self.path2pub2grant = 'grant'
        self.path2funders = 'funder'

    @property
    def fieldhierarchy(self, expand_domains=False):
        """
        The DataFrame keeping all field2field hierarhcial relationships

        Notes
        -------
        columns: 'ParentFieldId', 'ChildFieldId'
        

        """
        fieldhier = pd.read_csv(os.path.join(self.path2database, self.path2fieldinfo, 'fieldhierarchy0.csv.gz'))

        if expand_domains:
            fieldhier2 = fieldhier.rename(columns = {'ParentFieldId': 'SubFieldId', 'ChildFieldId': 'TopicId'})
            fieldhier2= fieldhier2.merge(fieldhier, how='inner', left_on = 'SubFieldId', right_on = 'ChildFieldId')
            del fieldhier2['ChildFieldId']
            fieldhier2 = fieldhier2.rename(columns = {'ParentFieldId': 'FieldId'})
            fieldhier2= fieldhier2.merge(fieldhier, how='inner', left_on = 'FieldId', right_on = 'ChildFieldId')
            del fieldhier2['ChildFieldId']
            fieldhier2 = fieldhier2.rename(columns = {'ParentFieldId': 'DomainId'})
            fieldhier = fieldhier2[['TopicId', 'SubFieldId', 'FieldId', 'DomainId']]
        else:
            return fieldhier

    def preprocess(self, dataframe_list = None, show_progress=True):
        """
        Bulk preprocess the MAG raw data.

        Parameters
        ----------
        dataframe_list: list, default None
            The list of DataFrames to preprocess.  If None, all MAG DataFrames are preprocessed.
            
        show_progress: bool, default True
            Show progress with processing of the data.

        """
        if dataframe_list is None or 'all' in dataframe_list:
            dataframe_list = openalex_works_dfset.union(openalex_dataframe_set)

        pubdataframe_list = set(dataframe_list).intersection(openalex_works_dfset)

        if 'affiliations' in dataframe_list or 'institutions' in dataframe_list:
            self.parse_affiliations(preprocess = True, show_progress=show_progress)

        if 'authors' in dataframe_list:
            self.parse_authors(preprocess = True, show_progress=show_progress)

        if 'publications' in dataframe_list or 'works' in dataframe_list:
            self.parse_publications(preprocess = True, dataframe_list=pubdataframe_list, show_progress=show_progress)

        if 'topics' in dataframe_list or 'fields' in dataframe_list:
            self.parse_fields(preprocess=True, field_type = 'topics', show_progress=show_progress)
        elif 'concepts' in dataframe_list:
            self.parse_fields(preprocess=True, field_type = 'concepts', show_progress=show_progress)

        if 'sources' in dataframe_list or 'journals' in dataframe_list:
            self.parse_sources(preprocess=True, show_progress=show_progress)

        if 'funders' in dataframe_list:
            self.parse_funders(preprocess=True, show_progress=show_progress)

    def download_from_source(self, aws_access_key_id='', aws_secret_access_key='', specific_update='', aws_bucket = 'openalex',
        dataframe_list = 'all', rewrite_existing = False, edit_works = True, show_progress=True):

        """
        Download the OpenAlex files from Amazon Web Services.
        The files will be saved to the path specified by `path2database` into RawXML.

        Parameters
        ----------
        source_url: str, default 'ftp.ncbi.nlm.nih.gov'
            The base url for the ftp server from which to download.
        
        aws_access_key_id: str
            The acess key for AWS (not required)

        aws_secret_access_key: str
            The secret acess key for AWS (not required)

        specific_update: str
            Download only a specific update date, specified by the date in Y-M-D format, 2022-01-01.  
            If empty the full data is downloaded.

        dataframe_list: list
            The data types to download and save from OpenAlex.
                'all'
                'affiliations'
                'authors'
                'publications'
                'references'
                'publicationauthoraffiliation'
                'fields'
                'topics'
                'concepts'
                'abstracts'
                'funders'
                'sources'
                'grants'

        rewrite_existing: bool, default False
            If True, overwrite existing files or if False, only download any missing files.

        edit_works: bool, default True
            If True, edit the works to remove pieces of data.  If False, force keeping all entries from the works file.

        show_progress: bool, default True
            Show progress with processing of the data.
        
        # edited from https://stackoverflow.com/questions/31918960/boto3-to-download-all-files-from-a-s3-bucket
        """

        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config
        except ImportError:
            raise ImportError("To download the OpenAlex data, you must install the aws S3 client and python package boto3.")

        if len(aws_access_key_id) > 0 and len(aws_secret_access_key) > 0:
            s3_client = boto3.client(aws_access_key_id=aws_access_key_id,
              aws_secret_access_key=aws_secret_access_key)
        else:
            s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))


        dataframe_list = set(dataframe_list)

        if dataframe_list is None or 'all' in dataframe_list:
            dataframe_list = dataframe_list.union(openalex_works_dfset).union(openalex_dataframe_set)
        else:
            # some dataframes have different names from those in pyscisci
            if 'publications' in dataframe_list: dataframe_list.add('works')
            if 'journals' in dataframe_list: dataframe_list.add('sources')
            if 'fields' in dataframe_list: dataframe_list.add('topics')
            if 'affiliations' in dataframe_list: dataframe_list.add('institutions')

        # see if we have to do any editing
        if edit_works and any(not pubsub in dataframe_list for pubsub in ['works', 'references', 'publicationauthoraffiliation', 'topics', 'abstracts', 'text']):
            edit_works = True
        else:
            edit_works = False
        

        
        if not rewrite_existing:
            files_already_downloaded = [os.path.join(dirpath.replace(self.path2database, ''), file).lstrip('/') for (dirpath, dirnames, filenames) in os.walk(self.path2database) for file in filenames]
        else:
            files_already_downloaded = []

        keys = []
        dirs = []
        next_token = ''
        
        base_kwargs = {'Bucket':aws_bucket}
        while next_token is not None:
            kwargs = base_kwargs.copy()
            if next_token != '':
                kwargs.update({'ContinuationToken': next_token})
            results = s3_client.list_objects_v2(**kwargs)
            
            for cdict in results.get('Contents', {}):
                k = cdict.get('Key')

                if not k in files_already_downloaded:
                    if 'data' in k: 
                        if k.split('/')[1] in dataframe_list and (specific_update=='' or k.split('/')[2] == 'updated_date={}'.format(specific_update)):
                            if k[-1] != '/':
                                keys.append(k)
                            else:
                                dirs.append(k)
                    else:
                        keys.append(k)
            next_token = results.get('NextContinuationToken')

        # make all of the directories
        for d in dirs:
            dest_pathname = os.path.join(self.path2database, d)
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
        
        # now download all of the 
        for k in tqdm(keys, desc='Downloading OpenAlex File Number', leave=True, disable=not show_progress):
            dest_pathname = os.path.join(self.path2database, k)

            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            
            s3_client.download_file(aws_bucket, k, dest_pathname)

            # the works contains references, concepts, and astracts, check if we keep these
            if edit_works and (len(k.split('/')) > 1 and k.split('/')[1] == 'works' and k[-3:]=='.gz'):
                with gzip.open(dest_pathname, 'r') as infile:
                    with gzip.open(dest_pathname.split('.')[0] + '_temp.gz', 'w') as outfile:
                        for line in infile:
                            if line != '\n'.encode():
                                jline = json.loads(line)

                                if ( not ('fields' in dataframe_list or 'concepts' in dataframe_list) ) and 'concepts' in jline:
                                    del jline['concepts']

                                if ( not ('references' in dataframe_list) ) and 'referenced_works' in jline:
                                    del jline['referenced_works']

                                if ( not ('publicationauthoraffiliation' in dataframe_list) ) and 'authorships' in jline:
                                    del jline['authorships']

                                if ( not ('abstracts' in dataframe_list or 'text' in dataframe_list) ) and 'abstract_inverted_index' in jline:
                                    del jline['abstract_inverted_index']

                                newline = json.dumps(jline) + "\n"
                                outfile.write(newline.encode('utf-8'))

                if os.path.exists(dest_pathname):
                    os.remove(dest_pathname)
                os.rename(dest_pathname.split('.')[0] + '_temp.gz', dest_pathname)



    def clean_openalex_ids(self, oid):
        try:
            oid = oid.split('/')[-1]

            try:
                return int(oid)
            except:
                return int(oid[1:])
        except:
            return None

    def parse_affiliations(self, preprocess=True, specific_update='', show_progress=True):
        """
        Parse the OpenAlex Affilation raw data.

        Parameters
        ----------
        preprocess: bool, default True
            Save the processed data in new DataFrames.

        specific_update: str
            Parse only a specific update date, specified by the date in Y-M-D format, 2022-01-01.  
            If empty the full data is parsed.

        show_progress: bool, default True
            Show progress with processing of the data.


        Returns
        ----------
        DataFrame
            Affiliation DataFrame.
        """

        affil_column_names = ['AffiliationId', 'FullName', 'InstitutionType', 'GridId', 'WikiDataId', 'WikiPage', 'RORId', 'Country', 'City', 'Region', 'Latitude', 'Longitude', 'NumberPublications', 'NumberCitations']

        if specific_update == '' or specific_update is None:
            institution_dir = os.path.join(self.path2database, 'data', 'institutions')
        else:
            institution_dir = os.path.join(self.path2database, 'data/institutions', 'updated_date={}'.format(specific_update))

        if not os.path.exists(institution_dir):
            raise ValueError("The institutions data was not found in the DataBase path.  Please download the data before parsing.")
        else:
            files_to_parse = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in os.walk(institution_dir) for file in filenames if '.gz' in file]


        if not os.path.exists(os.path.join(self.path2database, self.path2affiliation)):
            os.mkdir(os.path.join(self.path2database, self.path2affiliation))

        ifile = 0
        affiliation_info = []
        for file_name in tqdm(files_to_parse, desc='Affiliations', leave=True, disable=not show_progress):
            with gzip.open(os.path.join(institution_dir, file_name), 'r') as infile:
                for line in infile:
                    if line != '\n'.encode():
                        aline = json.loads(line)
                        affline = [self.clean_openalex_ids(aline['id']), aline.get('display_name', None), aline.get('type', None)]
                        otherids = aline.get('ids', {})
                        affline += [otherids.get(idname, None) for idname in ['grid', 'wikidata', 'wikipedia', 'ror']]
                        geoinfo = aline.get('geo', {})
                        affline += [geoinfo.get(idname, None) for idname in ['country_code', 'city', 'region']]
                        affline += [load_float(geoinfo.get(idname, None)) for idname in ['latitude', 'longitude']]
                        affline += [load_int(aline.get(idname, None)) for idname in ['works_count', 'cited_by_count']]
                        affiliation_info.append(affline)
                    

            aff = pd.DataFrame(affiliation_info, columns = affil_column_names)
            if preprocess:
                fname = os.path.join(self.path2database, self.path2affiliation, '{}{}.{}'.format(self.path2affiliation, ifile, self.database_extension))
                self.save_data_file(aff, fname, key =self.path2affiliation)

                ifile += 1
                affiliation_info = []

        return aff

    def parse_authors(self, preprocess = True, specific_update='', process_name = True, show_progress=True):
        """
        Parse the OpenAlex Author raw data.

        Parameters
        ----------
        preprocess: bool, default True
            Save the processed data in new DataFrames.

        specific_update: str
            Parse only a specific update date, specified by the date in Y-M-D format, 2022-01-01.  
            If empty the full data is parsed.

        process_name: bool, default True
            If True, then when processing the raw file, the package `NameParser <https://nameparser.readthedocs.io/en/latest/>`_
            will be used to split author FullNames.

        show_progress: bool, default True
            Show progress with processing of the data.

        Returns
        ----------
        DataFrame
            Author DataFrame.
        """

        if specific_update == '' or specific_update is None:
            author_dir = os.path.join(self.path2database, 'data', 'authors')
        else:
            author_dir = os.path.join(self.path2database, 'data/authors', 'updated_date={}'.format(specific_update))

        if not os.path.exists(author_dir):
            raise ValueError("The authors data was not found in the DataBase path.  Please download the data before parsing.")
        else:
            files_to_parse = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in os.walk(author_dir) for file in filenames if '.gz' in file]


        author_column_names = ['AuthorId', 'FullName', 'LastKnownAffiliationId', 'ORCID', 'WikiPage', 'NumberPublications', 'NumberCitations']
        if process_name:
            author_column_names += ['LastName', 'FirstName', 'MiddleName']


        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, self.path2author)):
                os.mkdir(os.path.join(self.path2database, self.path2author))

        ifile = 0
        authorinfo = []
        author = pd.DataFrame(authorinfo, columns = author_column_names)
        for file_name in tqdm(files_to_parse, desc='Authors', leave=True, disable=not show_progress):
            with gzip.open(os.path.join(author_dir, file_name), 'r') as infile:
                for line in infile:
                    if line != '\n'.encode():
                        aline = json.loads(line)
                        authline = [self.clean_openalex_ids(aline['id']), aline.get('display_name', None), self.clean_openalex_ids(aline.get('last_known_institution', None))]
                        otherids = aline.get('ids', {})
                        authline += [otherids.get(idname, None) for idname in ['orcid', 'wikipedia']]
                        authline += [load_int(aline.get(idname, None)) for idname in ['works_count', 'cited_by_count']]

                        # process the first, middle, and last names for the author
                        if process_name:
                            name = aline.get('display_name', '')
                            if name is None:
                                name = ''
                            hname = HumanName(unicodedata.normalize('NFD', name))
                            authline += [hname.last, hname.first, hname.middle]

                        authorinfo.append(authline)

                author = pd.DataFrame(authorinfo, columns = author_column_names)

                # time to save
                if preprocess:
                    fname = os.path.join(self.path2database, self.path2author, '{}{}.{}'.format(self.path2author, ifile, self.database_extension))
                    self.save_data_file(author, fname, key =self.path2author)

                    ifile += 1
                    authorinfo = []

        return author

    def parse_sources(self, preprocess = True, specific_update='', show_progress=True):
        """
        Parse the OpenAlex Sources raw data.

        Parameters
        ----------
        preprocess: bool, default True
            Save the processed data in new DataFrames.

        specific_update: str
            Parse only a specific update date, specified by the date in Y-M-D format, 2022-01-01.  
            If empty the full data is parsed.

        show_progress: bool, default True
            Show progress with processing of the data.

        Returns
        ----------
        DataFrame
            Sources DataFrame.
        """
        if specific_update == '' or specific_update is None:
            source_dir = os.path.join(self.path2database, 'data', 'sources')
        else:
            source_dir = os.path.join(self.path2database, 'data/sources', 'updated_date={}'.format(specific_update))

        if not os.path.exists(source_dir):
            raise ValueError("The sources data was not found in the DataBase path.  Please download the data before parsing.")
        else:
            files_to_parse = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in os.walk(source_dir) for file in filenames if '.gz' in file]


        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, self.path2journal)):
                os.mkdir(os.path.join(self.path2database, self.path2journal))


        source_column_names = ['JournalId', 'FullName', 'Type', 'ISSN', 'ISSN_l', 'HomePage', 'Country', 'NumberPublications', 'NumberCitations', 'IsOpenAccess']
        source_info = []
        ifile = 0
        for file_name in tqdm(files_to_parse, desc='Sources', leave=True, disable=not show_progress):
            with gzip.open(os.path.join(source_dir, file_name), 'r') as infile:
                for line in infile:
                    if line != '\n'.encode():
                        vline = json.loads(line)
                        vdata = [self.clean_openalex_ids(vline['id'])]
                        vdata += [vline.get(idname, None) for idname in ['display_name', 'type', 'issn', 'issn_l', 'homepage_url', 'country_code']]
                        vdata += [load_int(vline.get(idname, None)) for idname in ['works_count', 'cited_by_count']]
                        vdata += [load_bool(vline.get(idname, None)) for idname in ['is_oa']]
                        source_info.append(vdata)

                source = pd.DataFrame(source_info, columns = source_column_names)
                if preprocess:
                    fname = os.path.join(self.path2database, self.path2journal, '{}{}.{}'.format(self.path2journal, ifile, self.database_extension))
                    self.save_data_file(source, fname, key =self.path2journal)

                    ifile += 1
                    source_info = []
        
        return source


    def parse_publications(self, preprocess = True, specific_update='', preprocess_dicts = True, start_from_file_num = 0,
        dataframe_list = ['publications', 'references', 'publicationauthoraffiliation', 'fields', 'grants'],
        show_progress=True):
        """
        Parse the OpenAlex Works raw data.

        Parameters
        ----------
        preprocess: bool, default True
            Save the processed data in new DataFrames.

        specific_update: str
            Parse only a specific update date, specified by the date in Y-M-D format, 2022-01-01.  
            If empty the full data is parsed.

        preprocess_dicts: bool, default True
            Save the processed Year and DocType data as dictionaries.

        dataframe_list: list
            The data types to download and save from OpenAlex.
                'all'
                'publications'
                'references'
                'publicationauthoraffiliation'
                'fields'
                'abstracts'
                'grants'

        show_progress: bool, default True
            Show progress with processing of the data.

        Returns
        ----------
        DataFrame
            Publication DataFrame.
        """

        if specific_update == '' or specific_update is None:
            work_dir = os.path.join(self.path2database, 'data', 'works')
        else:
            work_dir = os.path.join(self.path2database, 'data/works', 'updated_date={}'.format(specific_update))

        if not os.path.exists(work_dir):
            raise ValueError("The works data was not found in the DataBase path.  Please download the data before parsing.")
        else:
            files_to_parse = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in os.walk(work_dir) for file in filenames if '.gz' in file]


        if 'all' in dataframe_list or 'works' in dataframe_list:
            dataframe_list = openalex_works_dfset


        pub_column_names = ['PublicationId', 'JournalId', 'Year', 'NumberCitations', 'Title', 'Date', 'DocType', 'Doi', 'PMID', 'Volume', 'Issue', 'FirstPage', 'LastPage', 'Language', 'IsRetracted', 'IsParaText', 'IsOpenAccess', 'OpenAccessStatus']

        if preprocess and (('publications' in dataframe_list) or ('works' in dataframe_list)):
            if not os.path.exists(os.path.join(self.path2database, self.path2pub)):
                os.mkdir(os.path.join(self.path2database, self.path2pub))

            pubinfo = []
            pub2year = []
            pub2doctype = []

        if preprocess and ('references' in dataframe_list):
            if not os.path.exists(os.path.join(self.path2database, self.path2pub2ref)):
                os.mkdir(os.path.join(self.path2database, self.path2pub2ref))

            pub2ref = []

        paa_column_names = ['PublicationId', 'AuthorId', 'AffiliationId', 'AuthorSequence', 'AuthorPosition', 'IsCorresponding', 'OrigAuthorName', 'RawAffiliationString']

        if preprocess and ('publicationauthoraffiliation' in dataframe_list):
            if not os.path.exists(os.path.join(self.path2database, self.path2paa)):
                os.mkdir(os.path.join(self.path2database, self.path2paa))

            paa = []

        pub2field_column_names = ['PublicationId', 'FieldId', 'Score']

        if preprocess and (('fields' in dataframe_list) or ('topics' in dataframe_list) or ('concepts' in dataframe_list)):
            if not os.path.exists(os.path.join(self.path2database, self.path2pub2field)):
                os.mkdir(os.path.join(self.path2database, self.path2pub2field))

            pub2field = []

        pub2text_column_names = ['PublicationId', 'Language', 'Title', 'Abstract']

        if preprocess and ('abstracts' in dataframe_list or 'text' in dataframe_list):
            if not os.path.exists(os.path.join(self.path2database, self.path2pub2text)):
                os.mkdir(os.path.join(self.path2database, self.path2pub2text))

            pub2text = []

        pub2grants_column_names = ['PublicationId', 'FunderId', 'AwardId']

        if preprocess and ('grants' in dataframe_list):
            if not os.path.exists(os.path.join(self.path2database, self.path2pub2grant)):
                os.mkdir(os.path.join(self.path2database, self.path2pub2grant))

            pub2grant = []

        ifile = 0
        for file_name in tqdm(files_to_parse, desc='Works', leave=True, disable=not show_progress):

            if ifile < start_from_file_num:
                ifile+=1
            else:

                with gzip.open(os.path.join(work_dir, file_name), 'r') as infile:
                    
                    iline = 0
                    
                    for line in infile:
                        if line != '\n'.encode():
                            
                            wline = json.loads(line)
                            ownid = self.clean_openalex_ids(wline['id'])

                            if ('publications' in dataframe_list) or ('works' in dataframe_list):
                                wdata = [ownid]
                                location_info = wline.get('primary_location', {})
                                
                                if not location_info is None:
                                    sourceinfo = location_info.get('source', {})
                                    if not sourceinfo is None:
                                        sourceid = self.clean_openalex_ids(sourceinfo.get('id', ""))
                                    else:
                                        sourceid = None
                                else:
                                    sourceid = None
                                wdata += [sourceid]
                                wdata += [load_int(wline.get(idname, None)) for idname in ['publication_year', 'cited_by_count']]
                                wdata += [wline.get(idname, None) for idname in ['title', 'publication_date', 'type']]
                                doi = wline.get('doi', None)
                                if isinstance(doi, str):
                                    doi = doi.replace("https://doi.org/", "")
                                wdata += [doi]
                                pmid = wline.get("ids", {}).get('pmid', None)
                                if isinstance(pmid, str):
                                    pmid = pmid.replace("https://pubmed.ncbi.nlm.nih.gov/", "")
                                wdata += [pmid]
                                bibinfo = wline.get('biblio', {})
                                wdata += [bibinfo.get(idname, None) for idname in ['volume', 'issue', 'first_page', 'last_page', 'language']]
                                wdata += [load_bool(wline.get(extraid, None)) for extraid in ['is_retracted', 'is_paratext', 'is_oa']]
                                oainfo = wline.get('open_access', {})
                                wdata += [oainfo.get(oai, None) for oai in ['oa_status']]

                                if preprocess_dicts:
                                    pub2year.append([ownid, wdata[2]])
                                    pub2doctype.append([ownid, wdata[6]])

                                pubinfo.append(wdata)

                            if ('references' in dataframe_list):
                                pub2ref.extend([[ownid, self.clean_openalex_ids(citedid)] for citedid in wline.get('referenced_works', [])])


                            if ('publicationauthoraffiliation' in dataframe_list):
                                iauthor = 1
                                for authorinfo in wline.get('authorships', []):
                                    authorid = self.clean_openalex_ids(authorinfo.get('author', {}).get('id', None))
                                    
                                    authorname = authorinfo.get('raw_author_name', '')
                                    affilstr = authorinfo.get('raw_affiliation_string', '')

                                    authorpos = authorinfo.get('author_position', None)
                                    authorcorr = authorinfo.get('is_corresponding', None)

                                    if (iauthor == 1 and authorinfo.get('author_position', None) != 'first'):
                                        iauthor = None

                                    institution_list = authorinfo.get('institutions', [])
                                    if len(institution_list) == 0:
                                        institution_list = [None]
                                    else:
                                        institution_list = [self.clean_openalex_ids(affinfo.get('id', None)) for affinfo in institution_list]

                                    for inid in institution_list:
                                        paa.append([ownid, authorid, inid, iauthor, authorpos, authorcorr, authorname, affilstr])

                                    if not iauthor is None:
                                        iauthor += 1

                            if ('topics' in dataframe_list):
                                pub_topics = wline.get('topics', [])
                                for topic_dict in pub_topics:
                                    pub2field.append([ownid, self.clean_openalex_ids(topic_dict.get('id', None)), load_float(topic_dict.get('score', None))])

                            elif ('concepts' in dataframe_list):
                                pub_concepts = wline.get('concepts', [])
                                for con_dict in pub_concepts:
                                    pub2field.append([ownid, self.clean_openalex_ids(con_dict.get('id', None)), load_float(con_dict.get('score', None))])

                            if ('grants' in dataframe_list):
                                pub_grants = wline.get('grants', [])
                                for grant_dict in pub_grants:
                                    pub2grant.append([ownid, self.clean_openalex_ids(grant_dict.get('funder', None)), grant_dict.get('award_id', None)])

                            if ('abstracts' in dataframe_list or 'text' in dataframe_list) and 'abstract_inverted_index' in wline:
                                pub2text.append([ownid, bibinfo.get('language', None), wline.get('title', None), abstractindex2text(wline.get('abstract_inverted_index', {}))])

                    
                    if ('publications' in dataframe_list) or ('works' in dataframe_list):
                        pub = pd.DataFrame(pubinfo, columns = pub_column_names)
                        if preprocess:
                            fname = os.path.join(self.path2database, self.path2pub, '{}{}.{}'.format(self.path2pub, ifile, self.database_extension))
                            self.save_data_file(pub, fname, key =self.path2pub)
                            
                            pubinfo = []
                    
                    if ('references' in dataframe_list):
                        pub2refdf = pd.DataFrame(pub2ref, columns = ['CitingPublicationId', 'CitedPublicationId'])
                        if preprocess:
                            fname = os.path.join(self.path2database, self.path2pub2ref, '{}{}.{}'.format(self.path2pub2ref, ifile, self.database_extension))
                            self.save_data_file(pub2refdf, fname, key =self.path2pub2ref)

                            pub2ref = []

                    if ('publicationauthoraffiliation' in dataframe_list):
                        paadf = pd.DataFrame(paa, columns = paa_column_names)
                        if preprocess:
                            fname = os.path.join(self.path2database, self.path2paa, '{}{}.{}'.format(self.path2paa, ifile, self.database_extension))
                            self.save_data_file(paadf, fname, key =self.path2paa)

                            paa = []

                    if ('fields' in dataframe_list) or ('concepts' in dataframe_list) or ('topics' in dataframe_list):
                        pub2fielddf = pd.DataFrame(pub2field, columns = pub2field_column_names)
                        if preprocess:
                            fname = os.path.join(self.path2database, self.path2pub2field, '{}{}.{}'.format(self.path2pub2field, ifile, self.database_extension))
                            self.save_data_file(pub2fielddf, fname, key =self.path2pub2field)

                            pub2field = []

                    if ('grants' in dataframe_list):
                        pub2grantsdf = pd.DataFrame(pub2grant, columns = pub2grants_column_names)
                        if preprocess:
                            fname = os.path.join(self.path2database, self.path2pub2grant, '{}{}.{}'.format(self.path2pub2grant, ifile, self.database_extension))
                            self.save_data_file(pub2grantsdf, fname, key =self.path2pub2grant)

                            pub2grant = []

                    if ('abstracts' in dataframe_list or 'text' in dataframe_list):
                        pub2textdf = pd.DataFrame(pub2text, columns = pub2text_column_names)
                        if preprocess:
                            fname = os.path.join(self.path2database, self.path2pub2text, '{}{}.{}'.format(self.path2pub2text, ifile, self.database_extension))
                            self.save_data_file(pub2textdf, fname, key =self.path2pub2text)

                        pub2text = []

                    ifile += 1

        if preprocess_dicts and (('publications' in dataframe_list) or ('works' in dataframe_list)) and preprocess_dicts:

            fname = os.path.join(self.path2database, '{}.{}'.format('pub2year', self.database_extension))
            self.save_data_file(pd.DataFrame(pub2year, columns=['PublicationId', 'Year']), fname, key ='pub2year')

            fname = os.path.join(self.path2database, '{}.{}'.format('pub2doctype', self.database_extension))
            self.save_data_file(pd.DataFrame(pub2doctype, columns=['PublicationId', 'DocType']), fname, key ='pub2doctype')

        return True

    def parse_fields(self, preprocess = True, field_type = 'topics', specific_update='', show_progress=True):
        if field_type == 'topics':
            self.parse_topics(specific_update='', show_progress=True)
        elif field_type == 'concepts':
            self.parse_topics(specific_update='', show_progress=True)

    def parse_topics(self, preprocess = True, specific_update ='', show_progress=True):
        """
        Parse the OpenAlex Topics

        these are arranged into domains, fields, subfields, and topics
        Parameters
        ----------
        preprocess: bool, default True
            Save the processed data in new DataFrames.

        specific_update: str
            Parse only a specific update date, specified by the date in Y-M-D format, 2022-01-01.  
            If empty the full data is parsed.

        show_progress: bool, default True
            Show progress with processing of the data.

        Returns
        ----------
        DataFrame
            FieldInfo DataFrame.
        """


        field_column_names = ['FieldId', 'FieldName', 'FieldLevel', 'FieldlevelName', 'WikiData', 'NumberPublications', 'NumberCitations', 'FieldDescription']

        #field_topic_hier = ['domains', 'fields', 'subfields', 'topics']
        field_topic_hier = ['fields', 'topics']

        if not os.path.exists(os.path.join(self.path2database, self.path2fieldinfo)):
            os.mkdir(os.path.join(self.path2database, self.path2fieldinfo))

        fieldinfo = []
        fieldhierarchy = []
        
        for ilevel, ft_level in enumerate(field_topic_hier):

            files_to_parse = []

            if specific_update == '' or specific_update is None:
                topic_dir = os.path.join(self.path2database, 'data', ft_level)
            else:
                topic_dir = os.path.join(self.path2database, 'data/{}'.format(ft_level), 'updated_date={}'.format(specific_update))

            if not os.path.exists(topic_dir):
                raise ValueError("The topics data was not found in the DataBase path.  Please download the data before parsing.")
            else:
                files_to_parse += [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in os.walk(topic_dir) for file in filenames if '.gz' in file]

            for file_name in tqdm(files_to_parse, desc=ft_level, leave=True, disable=not show_progress):
                with gzip.open(os.path.join(topic_dir, file_name), 'r') as infile:
                    for line in infile:
                        if line != '\n'.encode():
                            cline = json.loads(line)
                            fid = self.clean_openalex_ids(cline.get('id', None))
                            fielddata = [fid, cline.get('display_name', None), ilevel, ft_level]
                            fielddata.append(cline.get('ids', {}).get('wikipedia', None))
                            fielddata += [load_int(cline.get(prop, None)) for prop in ['works_count', 'cited_by_count']]
                            fielddata.append(cline.get('description', None))

                            fieldinfo.append(fielddata)

                            if ilevel < len(field_topic_hier) - 1:
                                for child in cline.get(field_topic_hier[ilevel+1], []):
                                    fieldhierarchy.append([fid, self.clean_openalex_ids(child.get('id', None))])

        fieldinfo = pd.DataFrame(fieldinfo, columns = field_column_names)
        if preprocess:
            fname = os.path.join(self.path2database, self.path2fieldinfo, '{}{}.{}'.format(self.path2fieldinfo, 0, self.database_extension))
            self.save_data_file(fieldinfo, fname, key =self.path2fieldinfo)

            fieldhierarchy = pd.DataFrame(fieldhierarchy, columns = ['ParentFieldId', 'ChildFieldId'])
            fname = os.path.join(self.path2database, self.path2fieldinfo, '{}{}.{}'.format('fieldhierarchy', 0, self.database_extension))
            self.save_data_file(fieldhierarchy, fname, key ='fieldhierarchy')

        return fieldinfo



    def parse_concepts(self, preprocess = True, field_type = 'topics', specific_update='', show_progress=True):
        """
        Parse the MAG Paper Concepts raw data.

        Parameters
        ----------
        preprocess: bool, default True
            Save the processed data in new DataFrames.

        specific_update: str
            Parse only a specific update date, specified by the date in Y-M-D format, 2022-01-01.  
            If empty the full data is parsed.

        show_progress: bool, default True
            Show progress with processing of the data.

        Returns
        ----------
        DataFrame
            FieldInfo DataFrame.
        """

        if specific_update == '' or specific_update is None:
            concept_dir = os.path.join(self.path2database, 'data', 'concepts')
        else:
            concept_dir = os.path.join(self.path2database, 'data/concepts', 'updated_date={}'.format(specific_update))

        if not os.path.exists(concept_dir):
            raise ValueError("The concepts data was not found in the DataBase path.  Please download the data before parsing.")
        else:
            files_to_parse = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in os.walk(concept_dir) for file in filenames if '.gz' in file]


        field_column_names = ['FieldId', 'FieldName', 'WikiData', 'FieldLevel', 'NumberPublications', 'NumberCitations']

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, self.path2fieldinfo )):
                os.mkdir(os.path.join(self.path2database, self.path2fieldinfo ))

        fieldinfo = []
        fieldhierarchy = []
        for file_name in tqdm(files_to_parse, desc='Concepts', leave=True, disable=not show_progress):
            with gzip.open(os.path.join(concept_dir, file_name), 'r') as infile:
                for line in infile:
                    if line != '\n'.encode():
                        cline = json.loads(line)
                        fid = self.clean_openalex_ids(cline.get('id', None))
                        fielddata = [fid]
                        fielddata += [cline.get(prop, None) for prop in ['display_name', 'wikidata']]
                        fielddata += [load_int(cline.get(prop, None)) for prop in ['level', 'works_count', 'cited_by_count']]
                        
                        fieldinfo.append(fielddata)

                        for an in cline.get('ancestors', []):
                            fieldhierarchy.append([self.clean_openalex_ids(an.get('id', None)), fid])
                
                    

        fieldinfo = pd.DataFrame(fieldinfo, columns = field_column_names)
        if preprocess:
            fname = os.path.join(self.path2database, self.path2fieldinfo, '{}{}.{}'.format(self.path2fieldinfo, 0, self.database_extension))
            self.save_data_file(fieldinfo, fname, key =self.path2fieldinfo)

            fieldhierarchy = pd.DataFrame(fieldhierarchy, columns = ['ParentFieldId', 'ChildFieldId'])
            fname = os.path.join(self.path2database, self.path2fieldinfo, '{}{}.{}'.format('fieldhierarchy', 0, self.database_extension))
            self.save_data_file(fieldhierarchy, fname, key ='fieldhierarchy')

        return fieldinfo


    def parse_funders(self, preprocess = True, specific_update='', show_progress=True):
        """
        Parse the OpenAlex Sources raw data.

        Parameters
        ----------
        preprocess: bool, default True
            Save the processed data in new DataFrames.

        specific_update: str
            Parse only a specific update date, specified by the date in Y-M-D format, 2022-01-01.  
            If empty the full data is parsed.

        show_progress: bool, default True
            Show progress with processing of the data.

        Returns
        ----------
        DataFrame
            Sources DataFrame.
        """
        if specific_update == '' or specific_update is None:
            funder_dir = os.path.join(self.path2database, 'data', 'funders')
        else:
            funder_dir = os.path.join(self.path2database, 'data/funders', 'updated_date={}'.format(specific_update))

        if not os.path.exists(funder_dir):
            raise ValueError("The funders data was not found in the DataBase path.  Please download the data before parsing.")
        else:
            files_to_parse = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in os.walk(funder_dir) for file in filenames if '.gz' in file]


        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, self.path2funders)):
                os.mkdir(os.path.join(self.path2database, self.path2funders))


        funder_column_names = ['FunderId', 'WikiDataId', 'RORId', 'FullName', 'HomePage', 'Country', 'NumberGrants', 'NumberPublications', 'NumberCitations']
        funder_info = []
        ifile = 0
        for file_name in tqdm(files_to_parse, desc='Funders', leave=True, disable=not show_progress):
            with gzip.open(os.path.join(funder_dir, file_name), 'r') as infile:
                for line in infile:
                    if line != '\n'.encode():
                        vline = json.loads(line)
                        vdata = [self.clean_openalex_ids(vline['id']), vline.get('ids', {}).get('wikidata', None), vline.get('ids', {}).get('ror', None)]
                        vdata += [vline.get(idname, None) for idname in ['display_name', 'homepage_url', 'country_code']]
                        vdata += [load_int(vline.get(idname, None)) for idname in ['grants_count', 'works_count', 'cited_by_count']]
                        funder_info.append(vdata)

                funder = pd.DataFrame(funder_info, columns = funder_column_names)
                if preprocess:
                    fname = os.path.join(self.path2database, self.path2funders, '{}{}.{}'.format(self.path2funders, ifile, self.database_extension))
                    self.save_data_file(funder, fname, key =self.path2funders)

                    ifile += 1
                    funder_info = []
        
        return funder


    def load_field_hierarchy(self):
        """
        Parse the OpenAlex Field Hierarchy data and return the Topic - SubField - Field - Domain memberships.

        Parameters
        ----------
    

        Returns
        ----------
        DataFrame
            FieldHierarchy DataFrame.
        """
        fieldhier = pd.read_csv(os.path.join(path2oa, 'fieldinfo', 'fieldhierarchy0.csv.gz'))
        
        fieldhier2 = fieldhier.rename(columns = {'ParentFieldId': 'SubFieldId', 'ChildFieldId': 'TopicId'})
        fieldhier2= fieldhier2.merge(fieldhier, how='inner', left_on = 'SubFieldId', right_on = 'ChildFieldId')
        del fieldhier2['ChildFieldId']
        fieldhier2 = fieldhier2.rename(columns = {'ParentFieldId': 'FieldId'})
        fieldhier2= fieldhier2.merge(fieldhier, how='inner', left_on = 'FieldId', right_on = 'ChildFieldId')
        del fieldhier2['ChildFieldId']
        fieldhier2 = fieldhier2.rename(columns = {'ParentFieldId': 'DomainId'})

        return fieldhier2
