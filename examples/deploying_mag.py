
from pyscisci.datasource.MAG import MAG
from pyscisci.citationanalysis import compute_disruption_index
from pyscisci.datasource.readwrite import append_to_preprocessed_df

import pandas as pd

mymag = MAG(path2database='/home/ajgates42/MAG', keep_in_memory=False)

mymag.preprocess(dflist = ['fields'])
#mymag.preprocess(dflist = ['publication', 'reference', 'publicationauthoraffiliation'])
#mymag.filter_doctypes(doctypes = ['j', 'b', 'bc', 'c'])
#mymag.compute_teamsize(save2pubdf=True)
#mymag.remove_selfcitations(preprocess=True)
#mymag.compute_impact(preprocess=True, citation_horizons = [5,10], noselfcite = True)
#mymag.compute_yearly_citations(preprocess=True, verbose=True)

disruption_df = compute_disruption_index(mymag.pub2refnoself_df)
append_to_preprocessed_df(disruption_df, mymag.path2database, 'impact')