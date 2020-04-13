# -*- coding: utf-8 -*-
"""
.. module:: interface_mag
    :synopsis: functions to facilitate working with the Microsoft Academic Graph

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
"""

import os
from nameparser import HumanName

from pyscisci.publication import Publication
from pyscisci.author import Author
from pyscisci.journal import Journal
from pyscisci.affiliation import Affiliation

from .utils import load_long, load_str, load_int, load_float

default_filenames = {
    'affiliaions' : ('mag/Affiliations.txt', ['AffiliationId', 'Rank', 'NormalizedName', 'DisplayName', 'GridId', 'OfficialPage', 'WikiPage', 'PaperCount', 'CitationCount:long', 'Latitude', 'Longitude:float?', 'CreatedDate:DateTime']),
    'authors' : ('mag/Authors.txt', ['AuthorId', 'Rank', 'NormalizedName', 'DisplayName', 'LastKnownAffiliationId', 'PaperCount:long', 'CitationCount:long', 'CreatedDate:DateTime']),
    'ConferenceInstances' : ('mag/ConferenceInstances.txt', ['ConferenceInstanceId:long', 'NormalizedName', 'DisplayName:string', 'ConferenceSeriesId:long', 'Location:string', 'OfficialUrl:string', 'StartDate:DateTime?', 'EndDate:DateTime?', 'AbstractRegistrationDate:DateTime?', 'SubmissionDeadlineDate:DateTime?', 'NotificationDueDate:DateTime?', 'FinalVersionDueDate:DateTime?', 'PaperCount:long', 'CitationCount:long', 'Latitude:float?', 'Longitude:float?', 'CreatedDate:DateTime']),
    'ConferenceSeries' : ('mag/ConferenceSeries.txt', ['ConferenceSeriesId:long', 'Rank:uint', 'NormalizedName:string', 'DisplayName:string', 'PaperCount', 'CitationCount:long', 'CreatedDate:DateTime']),
    'EntityRelatedEntities' : ('advanced/EntityRelatedEntities.txt', ['EntityId:long', 'EntityType:string', 'RelatedEntityId:long', 'RelatedEntityType:string', 'RelatedType:int', 'Score:float']),
    'FieldOfStudyChildren' : ('advanced/FieldOfStudyChildren.txt', ['FieldOfStudyId', 'ChildFieldOfStudyId']),
    'FieldOfStudyExtendedAttributes' : ('advanced/FieldOfStudyExtendedAttributes.txt', ['FieldOfStudyId', 'AttributeType:int', 'AttributeValue:string']),
    'fields' : ('advanced/FieldsOfStudy.txt', ['FieldOfStudyId', 'Rank', 'NormalizedName', 'DisplayName', 'MainType', 'Level', 'PaperCount:long', 'CitationCount:long', 'CreatedDate:DateTime']),
    'journals' : ('mag/Journals.txt', ['JournalId', 'Rank', 'NormalizedName', 'DisplayName', 'Issn', 'Publisher', 'Webpage:string', 'PaperCount:long', 'CitationCount:long', 'CreatedDate:DateTime']),
    'PaperAbstractsInvertedIndex' : ('nlp/PaperAbstractsInvertedIndex.txt.{*}', ['PaperId:long', 'IndexedAbstract:string']),
    'publicationauthoraffiliation' : ('mag/PaperAuthorAffiliations.txt', ['PaperId', 'AuthorId', 'AffiliationId', 'AuthorSequenceNumber', 'OriginalAuthor', 'OriginalAffiliation']),
    'PaperCitationContexts' : ('nlp/PaperCitationContexts.txt', ['PaperId', 'PaperReferenceId:long', 'CitationContext:string']),
    'PaperExtendedAttributes' : ('mag/PaperExtendedAttributes.txt', ['PaperId', 'AttributeType:int', 'AttributeValue:string']),
    'paperfields' : ('advanced/PaperFieldsOfStudy.txt', ['PaperId', 'FieldOfStudyId', 'Score']),
    'PaperRecommendations' : ('advanced/PaperRecommendations.txt', ['PaperId:long', 'RecommendedPaperId:long', 'Score:float']),
    'publicationreferences' : ('mag/PaperReferences.txt', ['PaperId', 'PaperReferenceId']),
    'PaperResources' : ('mag/PaperResources.txt', ['PaperId:long', 'ResourceType:int', 'ResourceUrl:string', 'SourceUrl:string', 'RelationshipType:int']),
    'PaperUrls' : ('mag/PaperUrls.txt', ['PaperId:long', 'SourceType:int?', 'SourceUrl:string', 'LanguageCode:string']),
    'publications' : ('mag/Papers.txt', ['PaperId', 'Rank', 'Doi', 'DocType', 'PaperTitle', 'OriginalTitle', 'BookTitle:string', 'Year', 'Date', 'Publisher:string', 'JournalId', 'ConferenceSeriesId:long?', 'ConferenceInstanceId:long?', 'Volume', 'Issue', 'FirstPage', 'LastPage', 'ReferenceCount:long', 'CitationCount:long', 'EstimatedCitation:long', 'OriginalVenue:string', 'FamilyId', 'CreatedDate:DateTime']),
    'RelatedFieldOfStudy' : ('advanced/RelatedFieldOfStudy.txt', ['FieldOfStudyId1:long', 'Type1:string', 'FieldOfStudyId2:long', 'Type2:string', 'Rank:float']),
  }

default_datatypes = {'PaperId':load_long, 'Doi':load_str, 'DocType':load_str, 'PaperTitle':load_str, 'PaperCount':load_long,
'OriginalTitle':load_str, 'AffiliationId':load_long, 'Year':load_int, 'Rank':load_long, 'AuthorId':load_long,
'JournalId':load_long, 'NormalizedName':load_str, 'FieldOfStudyId':load_long, 'DisplayName':load_str, 'GridId':load_str,
'OfficialPage':load_str, 'WikiPage':load_str, 'Latitude':load_str, 'Longitude':load_str, 'City':load_str,
'Country':load_str, 'LastKnownAffiliationId':load_long, 'Issn':load_str, 'ChildFieldOfStudyId':load_long, 'MainType':load_str,
'Latitude':load_float, 'Longitude':load_float, 'Level':load_int, 'PaperReferenceId':load_long, 'Date':load_str,
'Volume':load_str, 'Issue':load_str, 'FirstPage':load_str, 'LastPage':load_str, 'Publisher':load_str, 'Score':load_float,
'FamilyId':load_long
}


def get_author_name(sline, newauthor, author_idx):
    newauthor.fullname = default_datatypes['NormalizedName'](sline[author_idx['NormalizedName']])
    hname = HumanName(newauthor.fullname)
    newauthor.lastname = hname.last
    newauthor.firstname = hname.first
    newauthor.middlename = hname.middle
    return newauthor

def load_mag_affiliations(database, path2files = '', filename_dict = None):
    affiliation_idx = {dn:filename_dict['affiliaions'][1].index(dn) for dn in ['AffiliationId', 'NormalizedName', 'GridId', 'OfficialPage', 'WikiPage', 'Latitude', 'Longitude']}
    with open(os.path.join(path2files, filename_dict['affiliaions'][0]), 'r') as pubreffile:
        for line in pubreffile:
            sline = line.replace('\n', '').split('\t')
            newaffiliation = Affiliation(database)
            newaffiliation.id = default_datatypes['AffiliationId'](sline[affiliation_idx['AffiliationId']])
            newaffiliation.fullname = default_datatypes['NormalizedName'](sline[affiliation_idx['NormalizedName']])
            newaffiliation.gridid = default_datatypes['GridId'](sline[affiliation_idx['GridId']])
            newaffiliation.webpage = default_datatypes['OfficialPage'](sline[affiliation_idx['OfficialPage']])
            newaffiliation.wikipage = default_datatypes['WikiPage'](sline[affiliation_idx['WikiPage']])
            newaffiliation.latitude = default_datatypes['Latitude'](sline[affiliation_idx['Latitude']])
            newaffiliation.longitude = default_datatypes['Longitude'](sline[affiliation_idx['Longitude']])
            database.add_affiliation(newaffiliation)

def load_mag_authors(database, author_subset = None, path2files = '', filename_dict = None, full_info = False, keep_affiliation = True):
    author_idx = {dn:filename_dict['authors'][1].index(dn) for dn in ['AuthorId', 'NormalizedName']}
    with open(os.path.join(path2files, filename_dict['authors'][0]), 'r') as authorfile:
        for line in authorfile:
            sline = line.replace('\n', '').split('\t')
            authorid = default_datatypes['AuthorId'](sline[author_idx['AuthorId']])
            if author_subset is None or authorid in author_subset:
                newauthor = Author(database)
                newauthor.id = authorid
                if full_info:
                    newauthor = get_author_name(sline, newauthor, author_idx)
                database.add_author()

    paa_idx = {dn:filename_dict['publicationauthoraffiliation'][1].index(dn) for dn in ['PaperId', 'AuthorId', 'AffiliationId']}
    with open(os.path.join(path2files, filename_dict['publicationauthoraffiliation'][0]), 'r') as authorfile:
        for line in authorfile:
            sline = line.replace('\n', '').split('\t')
            authorid = default_datatypes['AuthorId'](sline[paa_idx['AuthorId']])
            if author_subset is None or authorid in author_subset:
                pubid = default_datatypes['PaperId'](sline[paa_idx['PaperId']])
                database.get_author(authorid).add_publication(pubid)
                if keep_affiliation:
                    affid = default_datatypes['AffiliationId'](sline[paa_idx['AffiliationId']])
                    database.get_author(authorid).affiliation2pub[affid].append(pubid)
                    database.get_affiliation(affid).authors.add(authorid)
                    database.get_affiliation(affid).publications.add(pubid)

def get_publication_info(sline, newpub, pub_idx):
    newpub.journal = default_datatypes['JournalId'](sline[pub_idx['JournalId']])
    newpub.title = default_datatypes['PaperTitle'](sline[pub_idx['PaperTitle']])
    newpub.sort_date = default_datatypes['Date'](sline[pub_idx['Date']])
    newpub.doctype = default_datatypes['DocType'](sline[pub_idx['DocType']])
    newpub.volume = default_datatypes['Volume'](sline[pub_idx['Volume']])
    newpub.issue = default_datatypes['Issue'](sline[pub_idx['Issue']])
    newpub.pages = default_datatypes['FirstPage'](sline[pub_idx['FirstPage']]) + '-' + default_datatypes['LastPage'](sline[pub_idx['LastPage']])
    newpub.metadata['familyId'] = default_datatypes['FamilyId'](sline[pub_idx['FamilyId']])
    return newpub

def load_mag_pubs(database, publication_subset = None, path2files = '', filename_dict = None, full_info=False):
    pub_idx = {dn:filename_dict['publications'][1].index(dn) for dn in ['PaperId', 'Year', 'Doi', 'DocType', 'PaperTitle', 'Date', 'Volume', 'Issue', 'FirstPage', 'LastPage', 'FamilyId']}
    with open(os.path.join(path2files, filename_dict['publications'][0]), 'r') as pubfile:
        for line in pubfile:
            sline = line.replace('\n', '').split('\t')
            pubid = default_datatypes['PaperId'](sline[pub_idx['PaperId']])
            if publication_subset is None or pubid in publication_subset:
                newpub = Publication(database)
                newpub.id = pubid
                newpub.year = default_datatypes['Year'](sline[pub_idx['Year']])
                if full_info:
                    newpub = get_publication_info(sline, newpub, pub_idx)
                database.add_pub(newpub)

def load_mag_journals(database, path2files = '', filename_dict=None):
    jour_idx = {dn:filename_dict['journals'][1].index(dn) for dn in ['JournalId', 'NormalizedName', 'Issn', 'Publisher']}
    with open(os.path.join(path2files, filename_dict['journals'][0]), 'r') as jfile:
        for line in jfile:
            sline = line.replace('\n', '').split('\t')
            newjournal = Journal(database)
            newjournal.id = default_datatypes['JournalId'](sline[jour_idx['JournalId']])
            newjournal.fullname = default_datatypes['NormalizedName'](sline[jour_idx['NormalizedName']])
            newjournal.issn = default_datatypes['Issn'](sline[jour_idx['Issn']])
            newjournal.publisher = default_datatypes['Publisher'](sline[jour_idx['Publisher']])
            database.add_journal(newjournal)

def load_mag_references(database, publication_subset = None, path2files = '', filename_dict = None):
    ref_idx = {dn:filename_dict['publicationreferences'][1].index(dn) for dn in ['PaperId', 'PaperReferenceId']}
    with open(os.path.join(path2files, filename_dict['publicationreferences'][0]), 'r') as pubreffile:
        for line in pubreffile:
            sline = line.replace('\n', '').split('\t')
            citingid = default_datatypes['PaperId'](sline[ref_idx['PaperId']])
            citedid = default_datatypes['PaperReferenceId'](sline[ref_idx['PaperReferenceId']])
            if publication_subset is None:
                database.get_pub(citingid).references.append(citedid)
                database.get_pub(citedid).citations.append(citingid)
            else:
                if citedid in publication_subset:
                    database.get_pub(citedid).citations.append(citingid)
                if citingid in publication_subset:
                    database.get_pub(citingid).citations.append(citedid)

def load_mag_fields(database, publication_subset = None, path2files = '', filename_dict = None):
    if False:
        # TODO
        with open(os.path.join(path2files, filenames['fields']), 'r') as pubreffile:
            for line in pubreffile:
                sline = line.replace('\n', '').split('\t')


def load_mag(database, path2files = '', filename_dict = None, files2load = None, author_subset = None,
    publication_subset = None, full_info = False, keep_affiliation = True):

    filenames = default_filenames
    if isinstance(filename_dict, dict):
        for filetype, filename in filename_dict.items():
            filename[filetype] = filename

    if files2load is None:
        files2load = list(default_filenames.keys())

    if 'affiliaions' in files2load:
        load_mag_affiliations(database, path2files, filename_dict)

    if 'authors' in files2load:
        load_mag_authors(database, author_subset, path2files, filename_dict, full_info, keep_affiliation)

    if 'publications' in files2load:
        if not author_subset is None:
            if publication_subset is None:
                publication_subset = []
            publication_subset = list(set(p for a in database.author_generator() for p in a.publications).union(set(publication_subset)))

        load_mag_pubs(database, publication_subset, path2files, filename_dict, full_info)

        load_mag_references(database, publication_subset, path2files, filename_dict)

    if 'journals' in files2load:
            load_mag_journals(database, path2files, filename_dict)

    if 'fields' in files2load:
        load_mag_fields(database, publication_subset, path2files, filename_dict)


