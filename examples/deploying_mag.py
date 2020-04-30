import pyscisci.datasource as scidata

mymag = scidata.MAG(path2database='/home/ajgates42/MAG')

#mymag.preprocess()
#mymag.compute_teamsize(save2pubdf=True)
#mymag.remove_selfcitations(preprocess=True)
mymag.compute_impact(preprocess=True, citation_horizons = [5,10], noselfcite = True)

