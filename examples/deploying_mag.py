"""
A first example in using Microsoft Academic Graph (MAG).

"""

import os
import pandas as pd

from pyscisci.datasource.MAG import MAG

mymag = MAG(path2database='/home/ajgates42/MAG', keep_in_memory=False)

mymag.preprocess() # first preprocess the raw MAG files

mymag.filter_doctypes(doctypes = ['j', 'b', 'bc', 'c'])  #filter the publications to only these doc types (MAG has a lot of other file types by default)
mymag.compute_teamsize(save2pubdf=True)
mymag.remove_selfcitations(preprocess=True)
mymag.compute_impact(preprocess=True, citation_horizons = [5,10], noselfcite = True)  # some commonly used impact measures C5 and C10

