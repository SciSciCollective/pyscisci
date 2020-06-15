
import os
import pandas as pd

from pyscisci.datasource.DBLP import DBLP
from pyscisci.network import coauthorship_network


mydblp = DBLP(path2database='/home/ajgates42/DBLP', keep_in_memory=False, show_progress=True)

a2p_df = mydblp.author2pub_df

# coauthorship network of Mark E. J. Newman --- 299546
coauthornet, author2int = coauthorship_network(a2p_df, focus_author_ids = [299546], focus_constraint='ego', show_progress=True)


print(coauthornet.nnz)  # 70 edges
print(author2int)  # between 16 authors