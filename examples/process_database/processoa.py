import os
import numpy as np
import pandas as pd

from pyscisci.datasource.OpenAlex import OpenAlex
from pyscisci.utils import isin_sorted, rank_array

print("Starting Processing OpenAlex", flush=True)


path2oa = '/project/connecteddatahub/OpenAlex1125'

myoa = OpenAlex(path2oa)

myoa.download_from_source(rewrite_existing = False, dataframe_list=['all'])
myoa.preprocess(dataframe_list=['all'])

pub2doctype = pd.read_csv(os.path.join(path2oa, 'pub2doctype.csv.gz'))
pub2year = pd.read_csv(os.path.join(path2oa, 'pub2year.csv.gz'))
print(len(pub2year), flush=True)

pub2doctype = pub2doctype[pub2doctype['DocType']=='article']
pub2doctype = pub2doctype.merge(pub2year, how='inner', on='PublicationId')
pub2doctype.dropna(inplace=True)
pubarticles = np.sort(pub2doctype['PublicationId'].values)
print(pubarticles.shape, flush=True)

pub2year = pub2year[isin_sorted(pub2year['PublicationId'].values, pubarticles)]

pub2ref = myoa.load_references()
pub2ref = pub2ref.merge(pub2year.rename(columns={'PublicationId':'CitedPublicationId', 'Year':'CitedYear'}), how='inner', on='CitedPublicationId')
pub2ref = pub2ref.merge(pub2year.rename(columns={'PublicationId':'CitingPublicationId', 'Year':'CitingYear'}), how='inner', on='CitingPublicationId')

oa_c = pub2ref.groupby(by=['CitedPublicationId'], as_index=False)['CitingPublicationId'].nunique().rename(columns={'CitingPublicationId':'Ctotal'})
oa_c.to_csv(os.path.join(path2oa, 'precomputed_data', 'oa_ctotal.csv.gz'), index=False, header=True, compression='gzip')

pub2ref = pub2ref[pub2ref['CitingYear'] >= pub2ref['CitedYear']]
for window in [5, 10]:
    oa_c_window = pub2ref[pub2ref['CitingYear'] <= pub2ref['CitedYear'] + window].groupby(by=['CitedPublicationId'], as_index=False)['CitingPublicationId'].nunique().rename(columns={'CitingPublicationId':'C{}'.format(window)})
    oa_c_window.to_csv(os.path.join(path2oa, 'precomputed_data', 'oa_c{}.csv.gz'.format(window)), index=False, header=True, compression='gzip')


fieldhier = pd.read_csv(os.path.join(path2oa, 'fieldinfo', 'fieldhierarchy0.csv.gz'))
print(fieldhier.shape)
fieldhier2 = fieldhier.rename(columns = {'ParentFieldId': 'SubFieldId', 'ChildFieldId': 'TopicId'})
fieldhier2= fieldhier2.merge(fieldhier, how='inner', left_on = 'SubFieldId', right_on = 'ChildFieldId')
del fieldhier2['ChildFieldId']
fieldhier2 = fieldhier2.rename(columns = {'ParentFieldId': 'FieldId'})
fieldhier2= fieldhier2.merge(fieldhier, how='inner', left_on = 'FieldId', right_on = 'ChildFieldId')
del fieldhier2['ChildFieldId']
fieldhier2 = fieldhier2.rename(columns = {'ParentFieldId': 'DomainId'})

pub2field = myoa.load_pub2field(filter_dict={'PublicationId':pubarticles}, columns=['PublicationId', 'FieldId'])
pub2field.rename(columns={'FieldId':'TopicId', 'PublicationId':'CitingPublicationId'}, inplace=True)
print(pub2field.shape, flush=True)

pub2field = pub2field.merge(fieldhier2[['TopicId', 'SubFieldId', 'FieldId']], how='left', on='TopicId')

citing_fields = pub2ref.merge(pub2field.rename(columns={'PublicationId':'CitingPublicationId'}), how='inner', on='CitingPublicationId')
citing_fields.rename(columns={'SubFieldId':'CitingSubFieldId', 'FieldId':'CitingFieldId', 'TopicId':'CitingTopicId'}, inplace=True)
print(citing_fields.shape, 'reference with fields')

for field_type in ['CitingFieldId']:
    print(field_type)
    fcites = citing_fields.groupby(['CitedPublicationId', field_type, 'CitedYear'], as_index=False)['CitingPublicationId'].nunique()
    fcites = fcites.rename(columns={'CitingPublicationId':'FieldCitations'})
    fcites.to_csv(os.path.join(path2oa, 'precomputed_data', "oa_c5_{}.csv.gz".format(field_type)), 
        compression='gzip', mode='w', header=True, index=False)

pub2field.rename(columns={'FieldId':'TopicId', 'CitingPublicationId':'CitedPublicationId'}, inplace=True)
pub2field = pub2field.merge(pub2year, how='inner', on='PublicationId')

for window in [5, 10, 500]:
    oa_c_window = pub2ref[pub2ref['CitingYear'] <= pub2ref['CitedYear'] + window].groupby(by=['CitedPublicationId', 'CitedYear'], as_index=False)['CitingPublicationId'].nunique().rename(columns={'CitingPublicationId':'C{}'.format(window)})
    print(oa_c_window.shape, 'reference window')
    if window == 500:
        window = 'total'

    for y in np.sort(oa_c_window['CitedYear'].unique()):
        ycited_fields = pub2field[pub2field['Year'] == y].merge(oa_c_window[oa_c_window['CitedYear'] == y][['CitedPublicationId', 'C{}'.format(window)]], how='left', on='CitedPublicationId')
        ycited_fields.fillna(0, inplace=True)

        for field_type in ['TopicId', 'SubFieldId', 'FieldId']:
            ycited_fields['{}C{}Rank'.format(field_type, window)] = ycited_fields.groupby(field_type, as_index=False)['C{}'.format(window)].transform(lambda x: rank_array(x, ascending=True, normed=True))
        ycited_fields.to_csv(os.path.join(path2oa, 'precomputed_data/fieldrank', "oa_c{}rank{}.csv.gz".format(window, y)), 
            compression='gzip', mode='w', header=True, index=False)

   