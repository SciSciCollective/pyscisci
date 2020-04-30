import pyscisci.datasource as scidata

mymag = scidata.MAG(path2database='/home/ajgates42/MAG')

mymag.compute_impact(preprocess=True, citation_horizons = [5,10], noselfcite = True)

