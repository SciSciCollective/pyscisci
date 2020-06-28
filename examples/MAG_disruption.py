"""
Compute the disruption index on the entire MAG.

"""

import os
import pandas as pd

from pyscisci.datasource.MAG import MAG
from pyscisci.metrics import compute_disruption_index


mymag = MAG(path2database='/home/ajgates42/MAG', keep_in_memory=False)

disruption_df = compute_disruption_index(mymag.pub2refnoself_df, show_progress=True)

print("DisruptionIndex complete, saving")

Nfiles = sum('impact' in fname for fname in os.listdir(os.path.join(mymag.path2database, 'impact')))
startcount = 1800
for ifile in range(Nfiles):
    datadf = pd.read_hdf(os.path.join(mymag.path2database, 'impact', 'impact{}.hdf'.format(ifile+startcount)))
    datadf = datadf.merge(disruption_df, how = 'left', on='PublicationId')
    datadf[['PublicationId', 'Year', 'DisruptionIndex']].to_hdf(os.path.join(mymag.path2database, 'disruption',
        'disruption{}.hdf'.format(ifile+startcount)), key = 'disruption', mode = 'w')
