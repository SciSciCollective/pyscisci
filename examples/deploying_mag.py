
import os
import pandas as pd

from pyscisci.datasource.MAG import MAG
from pyscisci.citationanalysis import compute_disruption_index
from pyscisci.datasource.readwrite import append_to_preprocessed_df



mymag = MAG(path2database='/home/ajgates42/MAG', keep_in_memory=False)

#mymag.preprocess(dflist = ['fields'])
#mymag.preprocess(dflist = ['publication', 'reference', 'publicationauthoraffiliation'])
#mymag.filter_doctypes(doctypes = ['j', 'b', 'bc', 'c'])
#mymag.compute_teamsize(save2pubdf=True)
#mymag.remove_selfcitations(preprocess=True)
#mymag.compute_impact(preprocess=True, citation_horizons = [5,10], noselfcite = True)
#mymag.compute_yearly_citations(preprocess=True, verbose=True)

disruption_df = compute_disruption_index(mymag.pub2refnoself_df, show_progress=True)

print("DisruptionIndex complete, saving")

#disruption_df.to_hdf(os.path.join(mymag.path2database, 'advancedimpact', 'disruption.hdf'), key = 'advancedimpact', mode = 'w')

Nfiles = sum('impact' in fname for fname in os.listdir(os.path.join(mymag.path2database, 'impact')))
startcount = 1800
for ifile in range(Nfiles):
    datadf = pd.read_hdf(os.path.join(mymag.path2database, 'impact', 'impact{}.hdf'.format(ifile+startcount)))
    datadf = datadf.merge(disruption_df, how = 'left', on='PublicationId')
    datadf[['PublicationId', 'Year', 'DisruptionIndex']].to_hdf(os.path.join(mymag.path2database, 'disruption',
        'disruption{}.hdf'.format(ifile+startcount)), key = 'disruption', mode = 'w')
