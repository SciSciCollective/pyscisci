# -*- coding: utf-8 -*-
"""
.. module:: scinlp
    :synopsis: The Natual Langauge Processing for SciSci

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>

 """
import os
import sys
import pandas as pd
import numpy as np

try:
    import sklearn
except ImportError:
    print('Please install sklearn to take full advantage of the NLP tools.')

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

