import os
import sys
import json
import gzip
import glob

import pandas as pd
import numpy as np
from nameparser import HumanName
import requests
import ftplib
from lxml import etree
from io import BytesIO

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


from pyscisci.datasource.readwrite import load_preprocessed_data, load_int, load_float, load_html_str, load_xml_text
from pyscisci.database import BibDataBase

class PubMed(BibDataBase):
    """
    Base class for PubMed Medline interface.

    Notes
    -------
    ~ PubMed comes as >1000 compressed XML files.
    ~ The PMID is renamed PublicationId to be consistent with the rest of pySciSci.
    ~ PubMed does not disambiguate Authors.
    ~
    """


    def __init__(self, path2database = '', database_extension='csv.gz', keep_in_memory = False, global_filter=None, 
        enable_dask=False, show_progress=True):

        self._default_init(path2database, database_extension, keep_in_memory, global_filter, enable_dask, show_progress)

        self.PublicationIdType = int
        self.AffiliationIdType = int
        self.AuthorIdType = str
        self.JournalIdType = str


    def _blank_pubmed_publication(self, PublicationId = 0):
        record = {}
        record['PublicationId'] = PublicationId
        record['Title'] = ''
        record['Year'] = 0
        record['Volume'] = 0
        record['Issue'] = ''
        record['Pages'] = ''
        record['JournalId'] = ''
        record['TeamSize'] = 0
        record['Month'] = 1
        record['Day'] = 1
        record['ISSN'] = ''
        record['Doi'] = ''
        record['PMCID'] = ''

        return record

    def _blank_pubmed_author(self):
        record = {}
        record['PublicationId'] = ''
        record['FullName'] = ''
        record['FirstName'] = ''
        record['LastName'] = ''
        record['Affiliations'] = ''
        record['ORCID'] = ''
        record['AuthorSequence'] = 0

        return record

    def _blank_pubmed_grant(self):
        record = {}
        record['PublicationId'] = ''
        record['GrantID'] = ''
        record['Acronym'] = ''
        record['Agency'] = ''
        record['Country'] = ''

        return record

    def _save_dataframes(self, ifile, publication, paa, pub2ref, pub2field, pub2abstract, pub2grant):

        publication = pd.DataFrame(publication)
        publication['PublicationId'] = publication['PublicationId'].astype(int)
        publication['Year'] = publication['Year'].astype(int)
        publication['Month'] = publication['Month'].astype(int)
        publication['Day'] = publication['Day'].astype(int)
        publication['Volume'] = pd.to_numeric(publication['Volume'])
        publication['TeamSize'] = publication['TeamSize'].astype(int)
        fname = os.path.join(self.path2database, self.path2pub, '{}{}.{}'.format(self.path2pub, ifile, self.database_extension))
        self.save_data_file(publication, fname, key =self.path2pub)


        paa = pd.DataFrame(paa)
        paa['AuthorSequence'] = paa['AuthorSequence'].astype(int)
        fname = os.path.join(self.path2database, self.path2paa, '{}{}.{}'.format(self.path2paa, ifile, self.database_extension))
        self.save_data_file(paa, fname, key =self.path2paa)

        pub2field = pd.DataFrame(pub2field, columns = ['PublicationId', 'FieldId'])
        pub2field['PublicationId'] = pub2field['PublicationId'].astype(int)
        fname = os.path.join(self.path2database, self.path2pub2field, '{}{}.{}'.format(self.path2pub2field, ifile, self.database_extension))
        self.save_data_file(pub2field, fname, key =self.path2pub2field)

        pub2ref = pd.DataFrame(pub2ref, columns = ['CitedPublicationId', 'CitingPublicationId', 'Citation'])
        fname = os.path.join(self.path2database, self.path2pub2ref, '{}{}.{}'.format(self.path2pub2ref, ifile, self.database_extension))
        self.save_data_file(pub2ref, fname, key =self.path2pub2ref)

        pub2abstract = pd.DataFrame(pub2abstract, columns = ['PublicationId', 'Abstract'])
        pub2abstract['PublicationId'] = pub2abstract['PublicationId'].astype(int)
        fname = os.path.join(self.path2database, self.path2pub2abstract, '{}{}.{}'.format(self.path2pub2abstract, ifile, self.database_extension))
        self.save_data_file(pub2abstract, fname, key =self.path2pub2abstract)

        pub2grant = pd.DataFrame(pub2grant)
        #pub2grant['PublicationId'] = pub2grant['PublicationId'].astype(int)
        fname = os.path.join(self.path2database, 'pub2grant', '{}{}.{}'.format('pub2grant', ifile, self.database_extension))
        self.save_data_file(pub2grant, fname, key ='pub2grant')

    def preprocess(self, xml_directory = 'RawXML', process_name=True, num_file_lines=10**6, show_progress=True,rewrite_existing = False):
        """
        Bulk preprocess of the PubMed raw data.

        Parameters
        ----------
        process_name: bool, default True
            If True, then when processing the raw file, the package `NameParser <https://nameparser.readthedocs.io/en/latest/>`_
            will be used to split author FullNames.

        num_file_lines: int, default 10**6
            The processed data will be saved into smaller DataFrames, each with `num_file_lines` rows.

        show_progress: bool, default True
            Show progress with processing of the data.

        rewrite_existing: bool, default False
            If True, rewrites the files in the data directory
        """

        if show_progress:
            print("Starting to preprocess the PubMed database.")

        for hier_dir_type in [self.path2pub, self.path2paa, self.path2pub2field, self.path2pub2ref, self.path2fieldinfo, self.path2pub2abstract, 'pub2grant']:

            if not os.path.exists(os.path.join(self.path2database, hier_dir_type)):
                os.mkdir(os.path.join(self.path2database, hier_dir_type))


        xmlfiles = sorted([fname for fname in os.listdir(os.path.join(self.path2database, xml_directory)) if '.xml' in fname])

        # read dtd - this takes
        path2database = self.path2database # remove self to use inside of this class
        class DTDResolver(etree.Resolver):
            def resolve(self, system_url, public_id, context):
                return self.resolve_filename(os.path.join(path2database, system_url), context)
        parser = etree.XMLParser(load_dtd=True, resolve_entities=True)

        pub2year = {}
        fieldinfo = {}

        #xmlfiles = ['pubmed22n1114.xml.gz']
        
        ifile = 0
        for xml_file_name in tqdm(xmlfiles, desc='PubMed xml files', leave=True, disable=not show_progress):

            # check if the xml file was already parsed
            dest_file_name = os.path.join(self.path2database, self.path2pub,'publication{}.hdf'.format(ifile))
            if not rewrite_existing and os.path.isfile(dest_file_name):
                ifile+=1
                continue

            publication = []
            paa = []
            pub2field = []
            pub2ref = []
            pub2abstract = []
            pub2grant = []

            xmltree = etree.parse(os.path.join(self.path2database, xml_directory, xml_file_name), parser)

            all_pubmed_articles = xmltree.findall("/PubmedArticle")
            
            for article_bucket in all_pubmed_articles:

                medline = article_bucket.find("MedlineCitation")

                # scrape the publication information
                PublicationId = load_int(load_xml_text(medline.find('PMID')))
                pub_record = self._blank_pubmed_publication(PublicationId)

                article = medline.find("Article")
                pub_record['Title'] = load_html_str(load_xml_text(article.find('ArticleTitle')))
                if article.find('Pagination') == None:
                    pub_record['Pages'] = None
                else:
                    pub_record['Pages'] = load_html_str(load_xml_text(article.find('Pagination').find("MedlinePgn")))

                journal = article.find("Journal")
                pub_record['JournalId'] = load_html_str(load_xml_text(journal.find("Title")))
                pub_record['Volume'] = load_int(load_xml_text(journal.find("JournalIssue").find("Volume")))
                pub_record['Issue'] = load_int(load_xml_text(journal.find("JournalIssue").find("Issue")))
                pub_record['ISSN'] = load_html_str(load_xml_text(journal.find("ISSN")))

                history = article_bucket.find("PubmedData/History")
                if not history is None:
                    pdate = history.find('PubMedPubDate')
                    if not pdate is None:
                        pub_record['Year'] = load_int(load_xml_text(pdate.find("Year")))
                        pub_record['Month'] = load_int(load_xml_text(pdate.find("Month")))
                        pub_record['Day'] = load_int(load_xml_text(pdate.find("Day")))


                if pub_record['Year'] > 0:
                    pub2year[PublicationId] = pub_record['Year']

                article_ids = article_bucket.find("PubmedData/ArticleIdList")
                if article_ids is not None:
                    doi = article_ids.find('ArticleId[@IdType="doi"]')
                    pub_record['Doi'] = load_xml_text(doi)

                    pmcid = article_ids.find('ArticleId[@IdType="pmc"]')
                    pub_record['PMCID'] = load_xml_text(pmcid)


                pub2abstract.append({'PublicationId':PublicationId, 'Abstract':load_xml_text(article.find("Abstract"))})

                grant_list = article.find('GrantList')
                if not grant_list is None:
                    for grant in grant_list:
                        grant_record = self._blank_pubmed_grant()
                        grant_record['PublicationId'] = PublicationId

                        grant_record['GrantID'] = load_xml_text(grant.find("GrantID"))
                        grant_record['Acronym'] = load_xml_text(grant.find("Acronym"))
                        grant_record['Agency'] = load_xml_text(grant.find("Agency"))
                        grant_record['Country'] = load_xml_text(grant.find("Country"))

                        pub2grant.append(grant_record)


                author_list = article.find('AuthorList')

                if not author_list is None:
                    for seq, author in enumerate(author_list.findall('Author')):
                        author_record = self._blank_pubmed_author()

                        author_record['PublicationId'] = PublicationId
                        author_record['FirstName'] = load_html_str(load_xml_text(author.find("ForeName")))
                        author_record['LastName'] = load_html_str(load_xml_text(author.find("LastName")))
                        author_record['FullName'] = author_record['FirstName'] + ' ' + author_record['LastName']
                        author_record['ORCID'] = load_xml_text(author.find('Identifier[@Source="ORCID"]'))
                        
                        if author.find("AffiliationInfo/Affiliation") is not None:
                            author_record['Affiliations'] = load_html_str(load_xml_text(author.find("AffiliationInfo/Affiliation")))
                            author_record['Affiliations'] = author_record['Affiliations'].replace("For a full list of the authors' affiliations please see the Acknowledgements section.","")

                        author_record['AuthorSequence'] = seq+1

                        paa.append(author_record)

                    pub_record['TeamSize'] = seq + 1

                meshterms = medline.find("MeshHeadingList")

                if meshterms is not None:
                    for term in meshterms.getchildren():
                        ui = term.find("DescriptorName").attrib.get("UI", "")
                        if len(ui)>0:
                            pub2field.append([PublicationId, ui])
                            fieldinfo[ui] = [load_xml_text(term.find("DescriptorName")), 'mesh']

                chemicals = medline.find("ChemicalList")
                if chemicals is not None:
                    for chemical in chemicals.findall("Chemical"):
                        ui = chemical.find("NameOfSubstance").attrib.get("UI", "")
                        if len(ui)>0:
                            pub2field.append([PublicationId, ui])
                            fieldinfo[ui] = [load_xml_text(chemical.find("NameOfSubstance")), 'chem']

                references = article_bucket.find("PubmedData/ReferenceList")
                if not references is None:
                    for ref in references.findall("Reference"):
                        citation = load_xml_text(ref.find("Citation"))
                        if not ref.find('ArticleIdList') is None:
                            pmid = load_int(load_xml_text(ref.find('ArticleIdList').find('ArticleId[@IdType="pubmed"]')))
                        else:
                            pmid = ""
                        pub2ref.append([PublicationId, pmid, citation])

                publication.append(pub_record)

            self._save_dataframes(ifile, publication, paa, pub2ref, pub2field, pub2abstract, pub2grant)
            ifile += 1

        # if rewriting
        #dest_file_name = os.path.join(self.path2database, self.path2fieldinfo,'fieldinfo0.hdf')
        if rewrite_existing:
            # save field info dictionary
            mesh_id_list = list(fieldinfo.values())
            for i, j in enumerate(fieldinfo.keys()):
                mesh_id_list[i].insert(0, j)

            fieldinfo = pd.DataFrame(mesh_id_list, columns = ['FieldId', 'FieldName', 'FieldType'], dtype=int)
            fname = os.path.join(self.path2database, self.path2fieldinfo, '{}{}.{}'.format(self.path2fieldinfo, 0, self.database_extension))
            self.save_data_file(fieldinfo, fname, key =self.path2fieldinfo)

        with gzip.open(os.path.join(self.path2database, 'pub2year.json.gz'), 'w') as outfile:
            outfile.write(json.dumps(pub2year).encode('utf8'))


    def download_from_source(self, source_url='ftp.ncbi.nlm.nih.gov', dtd_url = 'https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_190101.dtd',
        rewrite_existing = False, show_progress=True):
        """
        Download the Pubmed raw xml files and the dtd formating information from [PubMed](https://www.nlm.nih.gov/databases/download/pubmed_medline.html).
            1. pubmed/baseline - the directory containing the baseline compressed xml files
            2. pubmed_190101.dtd - the dtd containing xml syntax

        The files will be saved to the path specified by `path2database` into RawXML.

        Parameters
        ----------
        source_url: str, default 'ftp.ncbi.nlm.nih.gov'
            The base url for the ftp server from which to download.

        dtd_url: str, default 'pubmed_190101.dtd'
            The url for the dtd file.

        rewrite_existing: bool, default False
            If True, overwrite existing files or if False, only download any missing files.

        show_progress: bool, default True
            Show progress with processing of the data.

        """

        FTP_USER = "anonymous"
        FTP_PASS = ""

        ftp = ftplib.FTP(source_url, FTP_USER, FTP_PASS)
        ftp.encoding = "utf-8"
        ftp.cwd("pubmed/baseline/")

        files2download = sorted([fname for fname in ftp.nlst() if '.xml.gz' in fname and not '.md5' in fname])

        if not os.path.exists(os.path.join(self.path2database, 'RawXML')):
            os.mkdir(os.path.join(self.path2database, 'RawXML'))

        if not rewrite_existing:
            files_already_downloaded = os.listdir(os.path.join(self.path2database, 'RawXML'))
            files2download = [fname for fname in files2download if not fname in files_already_downloaded]

        for xml_file_name in tqdm(files2download, disable=not show_progress):
            with open(os.path.join(self.path2database, 'RawXML', xml_file_name), "wb") as outfile:
                ftp.retrbinary('RETR %s' % xml_file_name, outfile.write)

        with open(os.path.join(self.path2database, 'RawXML', 'pubmed_190101.dtd'), 'w') as outfile:
            outfile.write(requests.get(dtd_url).content.decode('utf-8'))

        ftp.quit()


    def parse_affiliations(self, preprocess = False):
        raise NotImplementedError("PubMed artciles are stored with all information in an xml file.  Run preprocess to parse the file.")

    def parse_publications(self, xml_directory = 'RawXML',preprocess = True, num_file_lines=10**7,rewrite_existing = False):
        """
        Parse the PubMed publication raw data.
        
        Parameters
        ----------
        preprocess: bool, default True
            Save the processed data in new DataFrames.
        process_name: bool, default True
            If True, then when processing the raw file, the package `NameParser <https://nameparser.readthedocs.io/en/latest/>`_
            will be used to split author FullNames.
        num_file_lines: int, default 5*10**6
            The processed data will be saved into smaller DataFrames, each with `num_file_lines` rows.
        show_progress: bool, default True
            Show progress with processing of the data.
        
        Returns
        ----------
        DataFrame
            Publication metadata DataFrame.
        """

        # process publication files through xml
        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, self.path2pub)):
                os.mkdir(os.path.join(self.path2database, self.path2pub))

            xmlfiles = sorted([fname for fname in os.listdir(os.path.join(self.path2database, xml_directory)) if '.xml' in fname])

            # read dtd - this takes
            path2database = self.path2database # remove self to use inside of this class
            class DTDResolver(etree.Resolver):
                def resolve(self, system_url, public_id, context):
                    return self.resolve_filename(os.path.join(path2database, system_url), context)

            parser = etree.XMLParser(load_dtd=True, resolve_entities=True)

            ifile = 0
            for xml_file_name in tqdm(xmlfiles, desc='PubMed publication xml files', leave=True, disable=not show_progress):

                # check if the xml file was already parsed
                dest_file_name = os.path.join(self.path2database, self.path2pub,'{}{}.{}'.format(self.path2pub, ifile, self.database_extension))
                if not rewrite_existing and os.path.isfile(dest_file_name):
                    ifile+=1
                    continue

                publication = []

                all_pubmed_articles = xmltree.findall("/PubmedArticle")

                for article_bucket in all_pubmed_articles:

                    medline = article_bucket.find("MedlineCitation")

                    # scrape the publication information
                    PublicationId = load_int(load_xml_text(medline.find('PMID')))

                    pub_record = self._blank_pubmed_publication(PublicationId)

                    article = medline.find("Article")
                    pub_record['Title'] = load_html_str(load_xml_text(article.find('ArticleTitle')))
                    if article.find('Pagination') == None:
                        pub_record['Pages'] = None
                    else:
                        pub_record['Pages'] = load_html_str(load_xml_text(article.find('Pagination').find("MedlinePgn")))

                    journal = article.find("Journal")
                    pub_record['JournalId'] = load_html_str(load_xml_text(journal.find("Title")))
                    pub_record['Volume'] = load_int(load_xml_text(journal.find("JournalIssue").find("Volume")))
                    pub_record['Issue'] = load_int(load_xml_text(journal.find("JournalIssue").find("Issue")))
                    pub_record['ISSN'] = load_html_str(load_xml_text(journal.find("ISSN")))

                    history = article_bucket.find("PubmedData/History")
                    if not history is None:
                        pdate = history.find('PubMedPubDate')
                        if not pdate is None:
                            pub_record['Year'] = load_int(load_xml_text(pdate.find("Year")))
                            pub_record['Month'] = load_int(load_xml_text(pdate.find("Month")))
                            pub_record['Day'] = load_int(load_xml_text(pdate.find("Day")))

                    article_ids = article_bucket.find("PubmedData/ArticleIdList")
                    if article_ids is not None:
                        doi = article_ids.find('ArticleId[@IdType="doi"]')
                        pub_record['Doi'] = load_xml_text(doi)


                    author_list = article.find('AuthorList')

                    if not author_list is None:
                        pub_record['TeamSize'] = len(author_list.findall('Author'))

                    publication.append(pub_record)

                # save publication dataframe
                publication = pd.DataFrame(publication)
                publication['PublicationId'] = publication['PublicationId'].astype(int)
                publication['Year'] = publication['Year'].astype(int)
                publication['Month'] = publication['Month'].astype(int)
                publication['Day'] = publication['Day'].astype(int)
                publication['Volume'] = pd.to_numeric(publication['Volume'])
                publication['TeamSize'] = publication['TeamSize'].astype(int)
                fname = os.path.join(self.path2database, self.path2pub, '{}{}.{}'.format(self.path2pub, ifile, self.database_extension))
                self.save_data_file(publication, fname, key =self.path2pub)

                ifile += 1

        ## load publication dataframe into a large file
        pub_files_list = glob.glob(os.path.join(self.path2database, self.path2pub, + '{}*.{}'.format(self.path2pub, self.database_extension)))

        pub = pd.DataFrame()

        print("Parsing files...")
        for tmp_pub in tqdm(paa_files_list, desc='PubMed author files', leave=True, disable=not show_progress):
            pub = pub.append(self.read_data_file(tmp_pub), ignore_index = True)

        return pub

    def parse_references(self, xml_directory='RawXML',preprocess = True, num_file_lines=10**7, rewrite_existing=False,show_progress=True):
        """
        Parse the PubMed References raw data.
        
        Parameters
        ----------
        preprocess: bool, default True
            Save the processed data in new DataFrames.
        process_name: bool, default True
            If True, then when processing the raw file, the package `NameParser <https://nameparser.readthedocs.io/en/latest/>`_
            will be used to split author FullNames.
        num_file_lines: int, default 5*10**6
            The processed data will be saved into smaller DataFrames, each with `num_file_lines` rows.
        show_progress: bool, default True
            Show progress with processing of the data.
        
        Returns
        ----------
        DataFrame
            Citations DataFrame.
        """

        # process author files through xml
        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, self.path2pub2ref)):
                os.mkdir(os.path.join(self.path2database, self.path2pub2ref))

            xmlfiles = sorted([fname for fname in os.listdir(os.path.join(self.path2database, xml_directory)) if '.xml' in fname])

            # read dtd - this takes
            path2database = self.path2database # remove self to use inside of this class
            class DTDResolver(etree.Resolver):
                def resolve(self, system_url, public_id, context):
                    return self.resolve_filename(os.path.join(path2database, system_url), context)

            parser = etree.XMLParser(load_dtd=True, resolve_entities=True)

            ifile = 0
            for xml_file_name in tqdm(xmlfiles, desc='PubMed reference xml files', leave=True, disable=not show_progress):

                xmltree = etree.parse(os.path.join(self.path2database, xml_directory, xml_file_name), parser)

                # check if the xml file was already parsed
                dest_file_name = os.path.join(self.path2database, self.path2pub2ref,'{}{}.{}'.format(self.path2pub2ref, ifile, self.database_extension))
                if not rewrite_existing and os.path.isfile(dest_file_name):
                    ifile+=1
                    continue

                pub2ref = []

                all_pubmed_articles = xmltree.findall("/PubmedArticle")

                for article_bucket in all_pubmed_articles:

                    medline = article_bucket.find("MedlineCitation")

                    # scrape the publication information
                    PublicationId = load_int(load_xml_text(medline.find('PMID')))


                    references = article_bucket.find("PubmedData/ReferenceList")
                    if not references is None:
                        for ref in references.findall("Reference"):
                            citation = load_xml_text(ref.find("Citation"))
                            if not ref.find('ArticleIdList') is None:
                                pmid = load_int(load_xml_text(ref.find('ArticleIdList').find('ArticleId[@IdType="pubmed"]')))
                            else:
                                pmid = ""
                            pub2ref.append([PublicationId, pmid, citation])

                # save file
                pub2ref = pd.DataFrame(pub2ref, columns = ['CitedPublicationId', 'CitingPublicationId', 'Citation'], dtype=int)
                fname = os.path.join(self.path2database, self.path2pub2ref, '{}{}.{}'.format(self.path2pub2ref, ifile, self.database_extension))
                self.save_data_file(pub2ref, fname, key =self.path2pub2ref)


        # load the citations into a large dataframe

        pub2ref_files = glob.glob(os.path.join(self.path2database, self.path2pub2ref, '{}*.{}'.format(self.path2pub2ref, self.database_extension)))

        pub2ref = pd.DataFrame()

        print("parsing citation data...")
        for pub2ref_tmp in tqdm(pub2ref_files,desc='PubMed citation xml files', leave=True, disable=not show_progress):
            pub2ref = pub2ref.append(self.read_data_file(pub2ref_tmp), ignore_index=True)

        return pub2ref

    def parse_publicationauthoraffiliation(self, xml_directory = 'RawXML',preprocess = True, num_file_lines=10**7, rewrite_existing = False):
        """
        Parse the PubMed publication-author raw data.
        
        Parameters
        ----------
        preprocess: bool, default True
            Save the processed data in new DataFrames.
        process_name: bool, default True
            If True, then when processing the raw file, the package `NameParser <https://nameparser.readthedocs.io/en/latest/>`_
            will be used to split author FullNames.
        num_file_lines: int, default 5*10**6
            The processed data will be saved into smaller DataFrames, each with `num_file_lines` rows.
        show_progress: bool, default True
            Show progress with processing of the data.
       
        Returns
        ----------
        DataFrame
            Publication-Author DataFrame.
        """

        # process author files through xml
        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, self.paa)):
                os.mkdir(os.path.join(self.path2database, self.paa))

            xmlfiles = sorted([fname for fname in os.listdir(os.path.join(self.path2database, xml_directory)) if '.xml' in fname])

            # read dtd - this takes
            path2database = self.path2database # remove self to use inside of this class
            class DTDResolver(etree.Resolver):
                def resolve(self, system_url, public_id, context):
                    return self.resolve_filename(os.path.join(path2database, system_url), context)

            parser = etree.XMLParser(load_dtd=True, resolve_entities=True)

            ifile = 0
            for xml_file_name in tqdm(xmlfiles, desc='PubMed author xml files', leave=True, disable=not show_progress):

                # check if the xml file was already parsed
                dest_file_name = os.path.join(self.path2database, self.path2paa,'{}{}.{}'.format(self.path2paa, ifile, self.database_extension))
                if not rewrite_existing and os.path.isfile(dest_file_name):
                    ifile+=1
                    continue

                paa = []

                all_pubmed_articles = xmltree.findall("/PubmedArticle")

                for article_bucket in all_pubmed_articles:

                    medline = article_bucket.find("MedlineCitation")

                    # scrape the publication information
                    PublicationId = load_int(load_xml_text(medline.find('PMID')))

                    author_list = article.find('AuthorList')

                    if not author_list is None:
                        for seq, author in enumerate(author_list.findall('Author')):
                            author_record = self._blank_pubmed_author()

                            author_record['PublicationId'] = PublicationId
                            author_record['FirstName'] = load_html_str(load_xml_text(author.find("ForeName")))
                            author_record['LastName'] = load_html_str(load_xml_text(author.find("LastName")))
                            author_record['FullName'] = author_record['FirstName'] + ' ' + author_record['LastName']

                            if author.find("AffiliationInfo/Affiliation") is not None:
                                author_record['Affiliations'] = load_html_str(load_xml_text(author.find("AffiliationInfo/Affiliation")))
                                author_record['Affiliations'] = author_record['Affiliations'].replace("For a full list of the authors' affiliations please see the Acknowledgements section.","")

                            author_record['AuthorSequence'] = seq+1

                            paa.append(author_record)
                paa = pd.DataFrame(paa)
                paa['AuthorSequence'] = paa['AuthorSequence'].astype(int)
                self.save_data_file( dest_file_name, key = self.path2paa)
                ifile += 1


        ## load publication author dataframe into a large file
        paa_files_list = glob.glob(os.path.join(self.path2database, self.path2paa, '{}*.{}'.format(self.path2paa, self.database_extension)))

        paa = pd.DataFrame()

        print("Parsing files...")
        for tmp_paa in tqdm(paa_files_list, desc='PubMed author files', leave=True, disable=not show_progress):
            paa = paa.append(self.read_data_file(tmp_paa), ignore_index = True)

        return paa

    def parse_fields(self, preprocess = True, num_file_lines=10**7, rewrite_existing=False,xml_directory = 'RawXML'):
        """
        Parse the PubMed field (mesh term) raw data.
        
        Parameters
        ----------
        preprocess: bool, default True
            Save the processed data in new DataFrames.
        process_name: bool, default True
            If True, then when processing the raw file, the package `NameParser <https://nameparser.readthedocs.io/en/latest/>`_
            will be used to split author FullNames.
        num_file_lines: int, default 5*10**6
            The processed data will be saved into smaller DataFrames, each with `num_file_lines` rows.
        show_progress: bool, default True
            Show progress with processing of the data.
        
        Returns
        ----------
        DataFrame
            Publication-Term ID DataFrame and Term ID - Term DataFrame
        """

        if preprocess:
            for hier_dir_type in [self.path2pub2field, self.path2fieldinfo]:
                if not os.path.exists(os.path.join(self.path2database, hier_dir_type)):
                    os.mkdir(os.path.join(self.path2database, hier_dir_type))

            xmlfiles = sorted([fname for fname in os.listdir(os.path.join(self.path2database, xml_directory)) if '.xml' in fname])

            # read dtd - this takes
            path2database = self.path2database # remove self to use inside of this class
            class DTDResolver(etree.Resolver):
                def resolve(self, system_url, public_id, context):
                    return self.resolve_filename(os.path.join(path2database, system_url), context)
            parser = etree.XMLParser(load_dtd=True, resolve_entities=True)

            # global id to term mapping
            fieldinfo = {}

            ifile = 0
            for xml_file_name in tqdm(xmlfiles, desc='PubMed xml files', leave=True, disable=not show_progress):

                # check if the xml file was already parsed
                dest_file_name = os.path.join(self.path2database, self.path2pub2field,'{}{}.{}'.format(self.path2pub2field, ifile, self.database_extension))
                if not rewrite_existing and os.path.isfile(dest_file_name):
                    ifile+=1
                    continue

                pub2field = []
                all_pubmed_articles = xmltree.findall("/PubmedArticle")

                for article_bucket in all_pubmed_articles:

                    medline = article_bucket.find("MedlineCitation")

                    # scrape the publication information
                    PublicationId = load_int(load_xml_text(medline.find('PMID')))

                    meshterms = medline.find("MeshHeadingList")

                    if meshterms is not None:
                        for term in meshterms.getchildren():
                            ui = term.find("DescriptorName").attrib.get("UI", "")
                            if len(ui)>0:
                                pub2field.append([PublicationId, ui])
                                fieldinfo[ui] = [load_xml_text(term.find("DescriptorName")), 'mesh']

                    chemicals = medline.find("ChemicalList")
                    if chemicals is not None:
                        for chemical in chemicals.findall("Chemical"):
                            ui = chemical.find("NameOfSubstance").attrib.get("UI", "")
                            if len(ui)>0:
                                pub2field.append([PublicationId, ui])
                                fieldinfo[ui] = [load_xml_text(chemical.find("NameOfSubstance")), 'chem']

                # save the pub-field id
                pub2field = pd.DataFrame(pub2field, columns = ['PublicationId', 'FieldId'], dtype=int)
                self.save_data_file(dest_file_name, key=self.path2pub2field)

                ifile += 1

            # if rewriting
            dest_file_name = os.path.join(self.path2database, self.path2fieldinfo, '{}{}.{}'.format(self.path2fieldinfo, 0, self.database_extension))
            if rewrite_existing:
                # save field info dictionary
                mesh_id_list = list(fieldinfo.values())
                for i, j in enumerate(fieldinfo.keys()):
                    mesh_id_list[i].insert(0, j)

                fieldinfo = pd.DataFrame(mesh_id_list, columns = ['FieldId', 'FieldName', 'FieldType'], dtype=int)
                self.save_data_file(fieldinfo, dest_file_name, key=self.path2fieldinfo)

        # load the dataframes
        # pub2field
        pub2field_files = glob.glob(os.path.join(self.path2database, self.path2pub2field, '{}*.{}'.format(self.path2pub2field, self.database_extension)))
        pub2field = pd.DataFrame()

        for pub2field_tmp_file in tqdm(pub2field_files, desc='PubMed pub2field files', leave=True, disable=not show_progress):
            pub2field = pub2field.append(self.read_data_file(pub2field_tmp_file), ignore_index=True)

        # field info map
        fieldinfo = self.read_data_file(os.path.join(self.path2database, self.path2fieldinfo, '{}{}.{}'.format(self.path2fieldinfo, 0, self.database_extension)))

        return pub2field, fieldinfo
