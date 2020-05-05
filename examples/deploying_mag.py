import pyscisci.datasource as scidata
import pandas as pd

mymag = scidata.MAG(path2database='/home/ajgates42/MAG')

#print(list(pd.read_hdf('/home/ajgates42/MAG/impact/impact1950.hdf')))
#mymag.preprocess()
#mymag.compute_teamsize(save2pubdf=True)
#mymag.remove_selfcitations(preprocess=True)
#mymag.compute_impact(preprocess=True, citation_horizons = [5,10], noselfcite = True)
mymag.compute_yearly_citations(preprocess=True)

