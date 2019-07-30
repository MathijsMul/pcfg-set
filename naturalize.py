"""
To run Stanford CoreNLP:

1. kill all processes at port 9000 (lsof -ti:9000 | xargs kill)
2. in terminal, move to directory of stanford corenlp
3. run "java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000"

Naturalization:

1. get WMT data
2. get tree statistics of WMT data using Stanford dependency parser
3. generate a lot of PCFG data using random probabilities for each sample
4. transform:
    i. partition WMT data in several ways (grid with different variable intervals)
    ii. reshape PCFG data to match the distribution over partitionings of i
    iii. fit bivariate Gaussian to reshaped PCFG data (and to WMT)
    iv. choose reshaped data with lowest KL divergence from WMT
5. (if necessary) infer PCFG parameters from optimized data with MLE (Expectation Maximization?)
6. generate more 'optimized' PCFG data with inferred parameters
7. apply 4 to optimized PCFG data

to generate more: repeat 6 and 7 with higher nr samples

"""

from pycorenlp import StanfordCoreNLP
import collections
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import operator
import random
import itertools
import sys

from utils import DataLoader
from interpret_set import interpret

DEPTHS_WMT_TEST = [3, 5, 9, 5, 6, 5, 5, 6, 3, 5, 6, 7, 3, 5, 4, 4, 7, 4, 3, 3, 4, 9, 5, 7, 5, 8, 9, 4, 3, 3, 4, 5, 3, 3, 5, 4, 4, 4, 6, 4, 4, 2, 4, 3, 4, 3, 4, 6, 6, 4, 4, 4, 5, 4, 5, 3, 4, 5, 4, 6, 3, 6, 6, 5, 6, 7, 5, 4, 5, 4, 7, 4, 7, 5, 5, 10, 4, 11, 5, 8, 3, 4, 5, 9, 4, 7, 4, 6, 4, 5, 4, 3, 3, 6, 3, 4, 3, 3, 4, 3, 4, 7, 4, 8, 5, 3, 5, 4, 3, 9, 8, 6, 6, 2, 3, 8, 3, 7, 5, 9, 4, 5, 6, 9, 4, 11, 5, 7, 5, 5, 7, 7, 7, 4, 7, 6, 4, 4, 6, 5, 6, 5, 6, 8, 10, 7, 8, 8, 9, 5, 4, 4, 7, 10, 3, 4, 4, 2, 5, 5, 3, 5, 7, 4, 5, 4, 4, 3, 4, 6, 9, 3, 6, 7, 4, 5, 6, 4, 3, 7, 3, 4, 4, 5, 7, 4, 5, 13, 5, 8, 10, 4, 7, 6, 5, 5, 9, 4, 6, 4, 5, 5, 4, 6, 5, 4, 5, 6, 6, 4, 10, 4, 6, 5, 7, 9, 7, 11, 5, 4, 5, 3, 6, 5, 4, 5, 7, 8, 5, 3, 4, 7, 3, 6, 5, 4, 6, 6, 7, 3, 4, 2, 5, 7, 8, 3, 4, 4, 5, 6, 4, 3, 5, 5, 4, 9, 6, 4, 3, 5, 4, 7, 5, 5, 2, 7, 9, 5, 3, 4, 6, 7, 5, 6, 3, 6, 4, 6, 5, 5, 5, 5, 8, 5, 8, 4, 9, 4, 3, 8, 5, 6, 3, 4, 6, 8, 4, 6, 4, 5, 5, 7, 8, 6, 5, 3, 9, 5, 4, 6, 8, 3, 3, 5, 5, 4, 5, 6, 6, 4, 6, 9, 3, 4, 4, 5, 3, 3, 3, 4, 5, 3, 4, 5, 3, 3, 6, 4, 5, 6, 5, 2, 6, 3, 10, 3, 9, 7, 10, 5, 5, 5, 5, 4, 3, 4, 5, 6, 3, 8, 6, 4, 3, 5, 4, 6, 7, 5, 4, 6, 5, 5, 4, 5, 8, 5, 4, 5, 12, 6, 4, 5, 2, 10, 8, 7, 14, 6, 9, 4, 5, 5, 5, 4, 3, 6, 3, 2, 3, 10, 5, 7, 3, 3, 3, 3, 4, 3, 3, 3, 3, 2, 7, 4, 9, 6, 6, 5, 8, 7, 7, 11, 6, 5, 7, 2, 8, 9, 4, 5, 4, 9, 6, 5, 6, 5, 5, 6, 5, 5, 8, 6, 7, 6, 7, 9, 3, 4, 8, 8, 5, 4, 5, 9, 5, 3, 4, 5, 4, 5, 6, 4, 4, 5, 4, 4, 6, 4, 7, 8, 3, 9, 4, 6, 3, 3, 7, 5, 2, 4, 5, 4, 6, 5, 5, 7, 6, 4, 5, 4, 5, 5, 7, 6, 5, 4, 7, 5, 5, 4, 5, 6, 7, 7, 5, 4, 8, 5, 3, 6, 6, 6, 6, 4, 3, 4, 5, 5, 5, 5, 7, 3, 6, 7, 3, 4, 5, 4, 4, 5, 5, 5, 6, 6, 4, 3, 5, 6, 4, 4, 7, 4, 3, 4, 4, 6, 7, 3, 5, 4, 3, 2, 3, 7, 4, 3, 5, 4, 6, 11, 5, 7, 5, 8, 5, 5, 9, 4, 5, 5, 4, 2, 7, 5, 11, 7, 7, 7, 5, 6, 4, 6, 5, 7, 6, 6, 4, 7, 7, 13, 4, 6, 6, 8, 6, 6, 6, 4, 5, 7, 5, 8, 9, 10, 5, 9, 11, 8, 4, 5, 6, 5, 7, 7, 4, 7, 5, 8, 9, 5, 6, 6, 7, 7, 6, 10, 7, 14, 6, 8, 8, 6, 5, 5, 5, 6, 5, 8, 4, 6, 8, 6, 5, 6, 4, 5, 6, 5, 8, 7, 13, 4, 5, 7, 6, 7, 4, 7, 4, 4, 5, 4, 6, 5, 4, 6, 8, 5, 14, 5, 5, 6, 7, 6, 4, 3, 6, 6, 7, 3, 5, 4, 7, 3, 6, 12, 4, 7, 5, 5, 4, 6, 6, 6, 6, 9, 9, 4, 7, 6, 8, 7, 6, 6, 9, 4, 8, 9, 6, 6, 10, 9, 3, 8, 6, 6, 4, 4, 5, 6, 5, 7, 4, 10, 9, 6, 2, 4, 7, 4, 6, 8, 5, 8, 6, 5, 6, 8, 5, 6, 3, 4, 5, 4, 9, 6, 8, 4, 5, 4, 8, 5, 7, 6, 5, 12, 5, 6, 14, 5, 8, 12, 4, 5, 5, 5, 6, 5, 4, 8, 7, 4, 18, 5, 5, 5, 6, 5, 6, 4, 4, 5, 3, 4, 4, 5, 5, 4, 9, 5, 6, 7, 7, 6, 7, 6, 8, 6, 6, 5, 8, 8, 9, 7, 4, 5, 3, 6, 7, 6, 8, 3, 8, 6, 6, 6, 9, 5, 8, 6, 3, 6, 5, 7, 3, 6, 4, 10, 5, 5, 5, 3, 7, 4, 6, 8, 3, 5, 4, 5, 5, 7, 3, 5, 4, 3, 4, 6, 4, 4, 3, 6, 5, 5, 6, 5, 8, 5, 10, 5, 3, 3, 5, 7, 3, 7, 5, 4, 6, 5, 6, 3, 3, 6, 4, 6, 6, 7, 3, 5, 7, 6, 7, 4, 4, 6, 4, 5, 9, 5, 5, 6, 3, 4, 7, 7, 4, 4, 3, 4, 3, 3, 7, 5, 6, 4, 4, 4, 3, 4, 4, 2, 3, 5, 5, 5, 5, 4, 3, 4, 3, 5, 5, 6, 4, 7, 4, 4, 5, 7, 5, 8, 8, 10, 5, 5, 4, 5, 4, 5, 6, 5, 6, 5, 6, 4, 5, 5, 7, 4, 3, 4, 10, 5, 6, 3, 6, 7, 5, 6, 9, 4, 4, 6, 7, 5, 5, 6, 5, 5, 4, 4, 5, 5, 6, 7, 6, 7, 6, 3, 6, 3, 7, 5, 6, 5, 11, 6, 5, 6, 5, 8, 5, 6, 6, 10, 5, 3, 8, 7, 5, 7, 4, 7, 5, 10, 6, 7, 5, 4, 5, 4, 4, 5, 6, 3, 6, 8, 5, 2, 3, 5, 6, 7, 4, 5, 6, 8, 5, 4, 7, 4, 4, 4, 8, 4, 6, 4, 7, 5, 4, 4, 3, 3, 4, 5, 8, 3, 6, 7, 8, 5, 4, 4, 5, 5, 6, 6, 4, 4, 4, 4, 4, 3, 6, 4, 4, 3, 7, 7, 3, 10, 7, 4, 12, 4, 8, 3, 9, 5, 12, 4, 5, 9, 7, 6, 4, 6, 3, 4, 4, 4, 3, 4, 3, 9, 4, 4, 4, 5, 4, 7, 4, 5, 9, 2, 3, 4, 5, 4, 7, 5, 5, 4, 4, 5, 5, 10, 5, 4, 9, 8, 6, 4, 5, 7, 4, 7, 3, 5, 5, 4, 5, 7, 9, 4, 9, 8, 4, 5, 6, 11, 7, 6, 3, 4, 5, 5, 3, 5, 9, 5, 3, 6, 5, 5, 8, 9, 7, 5, 5, 3, 7, 7, 6, 4, 3, 3, 5, 8, 5, 4, 10, 2, 13, 4, 6, 10, 10, 3, 3, 4, 6, 13, 9, 9, 2, 6, 11, 5, 5, 5, 5, 6, 5, 9, 7, 7, 4, 5, 7, 10, 6, 9, 8, 4, 4, 6, 7, 5, 5, 7, 9, 7, 3, 7, 3, 6, 5, 6, 14, 9, 7, 2, 7, 11, 10, 4, 9, 3, 5, 7, 5, 8, 10, 9, 9, 5, 8, 9, 5, 8, 5, 6, 7, 3, 4, 8, 7, 5, 6, 5, 6, 8, 3, 6, 1, 7, 5, 4, 4, 7, 2, 4, 5, 5, 4, 6, 4, 4, 5, 5, 3, 7, 4, 5, 5, 3, 4, 3, 3, 9, 9, 8, 5, 5, 6, 6, 5, 8, 4, 6, 5, 4, 5, 9, 8, 6, 4, 6, 3, 4, 3, 4, 5, 7, 5, 3, 5, 4, 5, 6, 4, 5, 3, 8, 13, 4, 6, 5, 6, 6, 5, 4, 3, 6, 10, 4, 6, 10, 3, 6, 3, 3, 3, 3, 11, 5, 2, 7, 3, 4, 7, 6, 6, 4, 5, 7, 4, 5, 6, 3, 5, 3, 5, 8, 4, 7, 4, 5, 4, 4, 3, 4, 5, 3, 4, 5, 4, 5, 5, 6, 5, 4, 6, 6, 3, 6, 8, 10, 7, 4, 5, 5, 4, 5, 3, 2, 5, 5, 6, 5, 4, 8, 5, 5, 7, 7, 3, 3, 3, 4, 5, 6, 4, 4, 4, 4, 3, 3, 3, 4, 5, 3, 4, 8, 2, 5, 4, 5, 6, 5, 7, 6, 8, 9, 6, 5, 3, 5, 4, 3, 4, 4, 6, 4, 8, 3, 4, 8, 4, 3, 3, 3, 5, 4, 4, 6, 6, 5, 7, 3, 9, 4, 6, 4, 5, 7, 5, 6, 9, 4, 6, 5, 7, 6, 7, 8, 6, 3, 5, 6, 5, 3, 4, 8, 4, 7, 9, 5, 8, 5, 6, 4, 8, 5, 6, 4, 5, 5, 4, 9, 4, 9, 5, 7, 4, 7, 5, 4, 7, 6, 5, 4, 5, 8, 3, 3, 3, 4, 4, 5, 5, 9, 4, 5, 5, 7, 6, 6, 8, 5, 7, 5, 5, 7, 5, 5, 4, 5, 5, 3, 3, 3, 6, 4, 7, 5, 6, 7, 11, 4, 4, 4, 5, 5, 6, 6, 6, 7, 3, 5, 5, 5, 6, 5, 6, 5, 4, 5, 4, 4, 7, 5, 7, 6, 3, 12, 6, 4, 5, 6, 5, 7, 4, 7, 9, 8, 4, 8, 6, 5, 5, 5, 5, 4, 6, 9, 6, 5, 7, 7, 3, 8, 4, 5, 7, 5, 6, 4, 7, 3, 7, 6, 6, 5, 6, 4, 5, 6, 6, 4, 4, 4, 5, 6, 5, 6, 7, 4, 3, 5, 3, 2, 8, 5, 7, 9, 5, 2, 7, 5, 4, 7, 4, 5, 7, 2, 8, 11, 3, 5, 4, 5, 5, 4, 7, 5, 7, 5, 5, 4, 4, 3, 5, 5, 6, 3, 4, 6, 3, 6, 4, 7, 6, 5, 4, 4, 4, 3, 5, 6, 6, 4, 3, 4, 4, 4, 3, 9, 6, 6, 4, 4, 3, 3, 4, 4, 4, 7, 7, 5, 6, 7, 8, 6, 4, 4, 3, 3, 3, 11, 8, 5, 5, 7, 4, 4, 3, 4, 7, 3, 5, 6, 4, 8, 4, 4, 5, 7, 4, 4, 8, 5, 8, 8, 7, 3, 5, 3, 4, 5, 8, 6, 4, 7, 5, 5, 4, 9, 4, 8, 8, 6, 6, 5, 7, 11, 7, 5, 10, 5, 5, 3, 5, 6, 6, 4, 7, 9, 5, 5, 3, 5, 7, 7, 6, 3, 6, 9, 6, 6, 8, 7, 3, 6, 8, 7, 4, 7, 8, 6, 5, 9, 5, 15, 6, 5, 9, 10, 3, 6, 10, 5, 5, 5, 4, 8, 13, 5, 6, 4, 4, 5, 8, 6, 5, 6, 9, 7, 9, 4, 5, 5, 7, 9, 9, 14, 5, 7, 5, 5, 4, 5, 7, 6, 4, 7, 9, 4, 6, 7, 9, 2, 6, 5, 5, 6, 4, 5, 3, 4, 5, 6, 3, 7, 2, 2, 7, 4, 3, 5, 6, 10, 3, 8, 4, 3, 4, 3, 7, 6, 2, 5, 6, 5, 6, 4, 5, 3, 6, 4, 6, 4, 8, 8, 5, 6, 3, 3, 3, 7, 11, 5, 6, 6, 7, 9, 3, 3, 6, 5, 4, 5, 6, 5, 5, 5, 8, 3, 10, 4, 6, 3, 5, 3, 6, 6, 2, 7, 3, 3, 10, 5, 5, 7, 7, 3, 7, 8, 5, 6, 3, 10, 3, 5, 3, 10, 4, 6, 6, 6, 5, 4, 6, 5, 4, 8, 7, 6, 4, 3, 9, 5, 5, 7, 4, 3, 8, 6, 11, 10, 4, 6, 10, 6, 4, 4, 7, 3, 8, 5, 13, 6, 5, 6, 11, 8, 9, 11, 12, 9, 6, 5, 5, 4, 5, 3, 6, 4, 7, 4, 5, 5, 3, 6, 4, 7, 4, 5, 3, 4, 3, 5, 2, 4, 5, 5, 3, 4, 4, 4, 5, 7, 6, 3, 6, 5, 6, 7, 7, 5, 3, 9, 3, 4, 5, 5, 5, 7, 4, 4, 6, 7, 4, 6, 4, 5, 5, 6, 11, 6, 8, 3, 3, 6, 5, 3, 4, 4, 7, 7, 10, 9, 4, 5, 6, 9, 4, 3, 5, 4, 6, 8, 3, 6, 7, 6, 4, 2, 5, 10, 5, 5, 6, 5, 4, 3, 8, 6, 5, 3, 2, 3, 6, 4, 4, 4, 6, 4, 7, 5, 4, 11, 5, 7, 7, 8, 4, 4, 5, 5, 5, 2, 3, 4, 5, 4, 6, 5, 4, 7, 6, 6, 3, 5, 6, 5, 5, 9, 8, 7, 3, 5, 6, 4, 6, 5, 6, 5, 6, 4, 9, 3, 4, 5, 8, 4, 4, 7, 6, 4, 11, 4, 4, 4, 8, 9, 6, 7, 11, 5, 3, 7, 6, 5, 6, 3, 3, 5, 7, 6, 5, 5, 4, 5, 5, 4, 8, 6, 7, 4, 7, 6, 9, 8, 7, 11, 11, 6, 6, 6, 7, 5, 6, 7, 8, 6, 4, 6, 5, 5, 7, 7, 6, 9, 7, 4, 7, 7, 3, 11, 3, 7, 6, 5, 5, 5, 5, 3, 3, 9, 8, 3, 5, 7, 5, 3, 7, 8, 4, 6, 7, 9, 7, 6, 4, 7, 6, 5, 5, 4, 2, 5, 5, 5, 5, 4, 7, 5, 3, 3, 5, 3, 6, 6, 5, 8, 8, 7, 4, 5, 6, 5, 9, 4, 6, 6, 7, 4, 4, 6, 6, 6, 7, 4, 4, 5, 4, 4, 7, 7, 5, 7, 4, 4, 8, 5, 5, 4, 6, 7, 3, 6, 5, 4, 4, 6, 3, 6, 6, 5, 6, 6, 5, 4, 5, 3, 10, 8, 4, 5, 7, 5, 5, 5, 4, 4, 7, 4, 7, 7, 3, 7, 5, 8, 4, 6, 4, 4, 7, 6, 6, 4, 4, 6, 6, 5, 4, 5, 8, 5, 4, 5, 4, 2, 4, 5, 5, 4, 5, 3, 5, 7, 6, 8, 10, 3, 6, 4, 3, 4, 4, 6, 6, 4, 3, 6, 5, 5, 6, 7, 6, 8, 11, 5, 8, 9, 3, 4, 5, 6, 6, 3, 8, 7, 6, 9, 6, 4, 6, 4, 6, 7, 8, 6, 7, 6, 5, 4, 5, 7, 8, 11, 7, 4, 7, 3, 12, 5, 5, 6, 4, 10, 7, 4, 6, 3, 4, 5, 4, 4, 4, 4, 3, 4, 5, 5, 7, 4, 6, 8, 4, 6, 6, 6, 6, 9, 7, 8, 7, 6, 4, 7, 7, 6, 5, 7, 7, 7, 5, 8, 4, 4, 6, 3, 3, 7, 5, 6, 7, 6, 6, 4, 6, 4, 6, 5, 5, 5, 4, 11, 4, 6, 3, 6, 4, 5, 4, 2, 3, 5, 5, 7, 4, 8, 5, 3, 5, 5, 5, 5, 5, 6, 3, 4, 4, 6, 7, 4, 5, 4, 5, 3, 5, 5, 5, 6, 5, 5, 4, 5, 7, 4, 3, 4, 8, 5, 7, 5, 7, 5, 4, 5, 5, 8, 8, 8, 5, 10, 4, 3, 9, 9, 8, 4, 6, 8, 7, 9, 7, 7, 7, 6, 8, 6, 3, 4, 6, 5, 6, 7, 3, 8, 7, 6, 2, 5, 4, 9, 8, 6, 10, 6, 3, 5, 10, 7, 4, 4, 4, 5, 4, 5, 12, 5, 7, 6, 4, 7, 5, 3, 6, 6, 8, 9, 6, 5, 3, 6, 4, 8, 6, 7, 5, 6, 3, 7, 6, 9, 7, 10, 5, 4, 7, 8, 8, 7, 4, 8, 6, 4, 6, 8, 12, 3, 6, 4, 4, 7, 3, 7, 6, 7, 8, 5, 7, 9, 5, 2, 6, 4, 3, 7, 5, 7, 5, 5, 7, 6, 4, 5, 4, 5, 8, 3, 5, 3, 6, 3, 4, 9, 6, 4, 8, 5, 3, 5, 8, 7, 4, 7, 5, 4, 9, 7, 4, 8, 5, 4, 5, 4, 7, 5, 8, 9, 5, 6, 4, 4, 4, 5, 4, 5, 5, 5, 4, 9, 4, 7, 5, 4, 6, 4, 3, 5, 11, 4, 3, 10, 11, 4, 9, 6, 6, 7, 7, 8, 9, 7, 7, 6, 5, 5, 7, 7, 5, 3, 10, 5, 3, 4, 7, 3, 12, 6, 6, 7, 5, 7, 3, 4, 10, 7, 6, 7, 4, 8, 9, 6, 7, 4, 5, 5, 3, 4, 4, 5, 6, 5, 3, 5, 3, 7, 4, 6, 3, 5, 5, 6, 7, 3, 6, 4, 3, 8, 6, 6, 4, 7, 6, 3, 4, 4, 5, 4, 3, 6, 5, 8, 5, 8, 3, 8, 5, 5, 4, 8, 6, 9, 6, 4, 7, 6, 6, 6, 5, 4, 6, 8, 7, 6, 4, 5, 6, 5, 6, 5, 6, 5, 5, 5, 5, 4, 4, 10, 5, 5, 7, 3, 3, 4, 4, 5, 6, 7, 9, 4, 7, 6, 10, 6, 3, 6, 6, 9, 6, 3, 6, 7, 9, 4, 7, 2, 6, 3, 8, 4, 7, 3, 4, 4, 3, 5, 5, 7, 4, 4, 9, 9, 7, 7, 5, 5, 5, 3, 12, 5, 3, 5, 3, 9, 4, 5, 7, 14, 5, 4, 4, 4, 8, 5, 9, 5, 8, 3, 11, 4, 3, 7, 5, 6, 6, 9, 5, 10, 5, 4, 8, 5, 7, 10, 9, 5, 6, 9, 4, 10, 6, 6, 7, 6, 6, 8, 10, 6, 6, 4, 8, 6, 6, 7, 3, 8, 4, 12, 5, 7, 3, 4, 4, 10, 8, 5, 3, 3, 4, 3, 5, 6, 7, 14, 7, 6, 3, 4, 4, 7, 7, 3, 8, 3, 5, 8, 4, 5, 3, 3, 4, 6, 3, 3, 5, 5, 3, 5, 5, 2, 3, 5, 5, 6, 3, 4, 6, 4, 7, 9, 3, 10, 3, 5, 8, 3, 3, 4, 8, 4, 7, 5, 3, 4, 5, 5, 4, 9, 5, 8, 5, 5, 3, 4, 4, 7, 4, 4, 2, 5, 4, 7, 5, 4, 4, 2, 4, 5, 5, 4, 6, 4, 4, 6, 4, 5]
LENGTHS_WMT_TEST = [8, 22, 24, 25, 17, 37, 14, 21, 14, 17, 13, 17, 9, 19, 10, 9, 24, 13, 5, 9, 13, 18, 13, 13, 19, 27, 28, 19, 9, 7, 6, 27, 7, 7, 22, 9, 7, 11, 20, 19, 10, 5, 20, 8, 15, 10, 11, 15, 22, 8, 15, 23, 43, 10, 18, 9, 11, 15, 21, 17, 7, 25, 17, 8, 18, 22, 9, 7, 13, 16, 27, 9, 14, 16, 20, 30, 15, 25, 9, 22, 9, 13, 10, 32, 17, 18, 11, 29, 10, 16, 9, 4, 8, 17, 9, 9, 6, 6, 7, 4, 7, 24, 8, 29, 17, 6, 10, 8, 8, 25, 34, 21, 16, 4, 9, 23, 9, 18, 21, 21, 12, 15, 16, 20, 13, 22, 18, 22, 25, 15, 25, 25, 26, 16, 23, 17, 9, 8, 21, 25, 12, 17, 13, 35, 30, 34, 25, 25, 28, 17, 7, 8, 27, 29, 5, 18, 19, 3, 18, 18, 7, 15, 29, 7, 12, 10, 8, 8, 11, 13, 30, 4, 14, 17, 7, 10, 24, 10, 6, 24, 7, 17, 9, 20, 14, 10, 17, 38, 21, 16, 37, 13, 22, 27, 17, 14, 36, 10, 18, 10, 16, 18, 14, 21, 16, 15, 20, 17, 21, 13, 34, 13, 13, 25, 28, 34, 27, 23, 15, 11, 13, 12, 22, 14, 14, 12, 10, 19, 18, 5, 12, 30, 8, 27, 21, 20, 24, 21, 18, 10, 20, 6, 24, 28, 27, 8, 13, 14, 14, 28, 13, 7, 26, 14, 12, 29, 26, 4, 8, 22, 15, 25, 17, 11, 3, 26, 42, 17, 9, 11, 22, 26, 19, 28, 12, 14, 9, 18, 17, 17, 10, 10, 37, 21, 26, 16, 30, 13, 7, 18, 14, 30, 7, 13, 18, 23, 14, 14, 13, 22, 16, 29, 29, 40, 21, 11, 31, 12, 12, 23, 17, 6, 10, 17, 22, 16, 18, 18, 28, 16, 18, 23, 7, 6, 17, 21, 9, 8, 9, 7, 13, 5, 11, 9, 9, 8, 19, 8, 19, 20, 17, 6, 15, 10, 35, 11, 22, 23, 24, 14, 12, 11, 17, 15, 5, 11, 18, 27, 9, 23, 15, 9, 9, 17, 15, 12, 24, 15, 11, 21, 12, 14, 9, 9, 32, 12, 15, 16, 38, 29, 15, 15, 5, 31, 19, 23, 47, 18, 37, 9, 12, 20, 15, 15, 10, 12, 8, 2, 4, 23, 12, 36, 6, 8, 3, 3, 8, 3, 3, 3, 3, 6, 29, 22, 28, 28, 18, 22, 44, 28, 23, 35, 23, 18, 35, 4, 31, 25, 13, 15, 18, 36, 25, 13, 20, 17, 14, 41, 25, 28, 32, 22, 23, 20, 25, 22, 10, 10, 19, 34, 18, 8, 13, 24, 13, 4, 6, 15, 19, 10, 13, 8, 15, 6, 8, 10, 13, 9, 17, 23, 6, 16, 9, 7, 4, 6, 21, 15, 3, 8, 11, 8, 27, 24, 25, 25, 11, 13, 19, 17, 24, 16, 24, 25, 15, 14, 21, 9, 11, 6, 19, 18, 26, 17, 16, 11, 27, 13, 12, 16, 20, 15, 36, 14, 10, 16, 19, 17, 18, 9, 22, 11, 26, 19, 5, 15, 9, 12, 10, 12, 23, 22, 23, 12, 13, 9, 12, 25, 10, 12, 27, 11, 7, 16, 18, 25, 23, 6, 16, 13, 9, 3, 12, 28, 17, 6, 21, 18, 27, 49, 16, 22, 23, 24, 8, 29, 31, 18, 13, 21, 10, 6, 43, 12, 39, 29, 25, 43, 26, 26, 17, 14, 18, 35, 21, 18, 18, 30, 35, 71, 10, 24, 14, 21, 23, 25, 25, 19, 20, 22, 15, 23, 32, 46, 16, 30, 28, 31, 9, 17, 26, 26, 34, 17, 11, 24, 10, 22, 32, 19, 25, 23, 23, 20, 11, 32, 21, 40, 23, 23, 26, 11, 17, 15, 11, 14, 10, 20, 9, 17, 15, 21, 19, 18, 13, 14, 31, 12, 27, 29, 35, 24, 29, 28, 29, 16, 11, 35, 22, 24, 16, 22, 40, 19, 16, 17, 28, 16, 44, 12, 25, 31, 29, 25, 10, 3, 30, 18, 18, 10, 25, 15, 15, 8, 17, 27, 11, 18, 12, 19, 11, 17, 18, 17, 20, 26, 27, 14, 12, 16, 25, 23, 21, 22, 33, 9, 28, 17, 23, 24, 23, 32, 6, 24, 29, 21, 10, 10, 20, 25, 21, 27, 20, 38, 31, 31, 2, 10, 23, 13, 26, 39, 20, 22, 29, 25, 16, 22, 18, 39, 8, 12, 11, 10, 27, 31, 20, 6, 11, 15, 29, 21, 17, 15, 12, 38, 26, 13, 42, 14, 12, 43, 11, 22, 25, 20, 24, 17, 15, 22, 22, 9, 42, 16, 17, 18, 14, 26, 17, 7, 13, 12, 5, 14, 10, 17, 17, 15, 35, 23, 30, 24, 24, 18, 16, 23, 17, 22, 20, 13, 15, 23, 23, 24, 14, 12, 11, 16, 19, 22, 32, 10, 36, 21, 23, 40, 31, 20, 26, 22, 7, 21, 28, 30, 9, 25, 17, 35, 30, 20, 30, 11, 15, 14, 23, 27, 9, 18, 22, 23, 19, 18, 9, 9, 13, 6, 13, 18, 12, 10, 11, 29, 24, 11, 22, 20, 26, 29, 28, 20, 4, 9, 21, 18, 7, 24, 15, 8, 13, 12, 28, 10, 6, 13, 18, 18, 18, 29, 12, 25, 27, 30, 18, 15, 21, 18, 13, 19, 35, 32, 13, 17, 7, 17, 33, 21, 13, 14, 7, 26, 5, 11, 30, 24, 12, 11, 11, 12, 8, 16, 8, 5, 5, 19, 11, 15, 14, 8, 6, 14, 6, 13, 16, 23, 21, 25, 14, 8, 8, 20, 17, 25, 26, 42, 24, 17, 13, 29, 14, 21, 20, 22, 18, 17, 18, 7, 10, 16, 23, 22, 5, 7, 37, 17, 18, 6, 26, 26, 20, 26, 50, 25, 15, 18, 16, 18, 13, 36, 20, 12, 17, 15, 14, 13, 30, 32, 22, 16, 19, 5, 23, 5, 27, 21, 14, 23, 31, 28, 13, 19, 16, 25, 24, 23, 17, 24, 30, 18, 21, 22, 20, 25, 10, 24, 18, 39, 20, 23, 18, 19, 19, 13, 13, 19, 12, 8, 15, 24, 14, 5, 5, 30, 17, 31, 14, 12, 19, 31, 9, 20, 18, 12, 13, 11, 22, 8, 19, 8, 24, 28, 11, 12, 6, 10, 8, 12, 11, 10, 9, 21, 45, 11, 12, 17, 17, 10, 16, 33, 7, 10, 20, 10, 12, 7, 23, 8, 11, 7, 22, 24, 8, 25, 20, 18, 38, 12, 20, 8, 25, 24, 28, 8, 12, 28, 20, 18, 15, 14, 11, 10, 7, 7, 5, 16, 14, 45, 11, 10, 8, 11, 13, 27, 10, 17, 32, 4, 9, 9, 9, 12, 26, 21, 22, 24, 12, 19, 14, 26, 17, 11, 40, 23, 18, 8, 15, 17, 17, 35, 7, 17, 16, 11, 13, 18, 34, 15, 44, 28, 11, 29, 28, 37, 20, 30, 6, 15, 25, 20, 10, 11, 33, 13, 8, 23, 18, 19, 28, 36, 18, 19, 19, 9, 25, 36, 14, 11, 7, 5, 21, 30, 19, 12, 38, 5, 47, 5, 37, 37, 50, 6, 6, 7, 19, 47, 34, 27, 3, 20, 29, 16, 21, 12, 9, 23, 16, 29, 20, 15, 8, 14, 25, 24, 23, 25, 30, 9, 9, 10, 14, 19, 14, 36, 22, 19, 9, 14, 4, 25, 9, 18, 36, 36, 23, 5, 33, 33, 39, 11, 40, 8, 15, 29, 11, 28, 55, 29, 36, 17, 27, 25, 12, 45, 13, 24, 35, 8, 18, 29, 18, 14, 13, 18, 12, 37, 9, 44, 1, 24, 15, 14, 19, 19, 8, 10, 18, 13, 15, 13, 11, 15, 17, 13, 9, 31, 8, 19, 10, 7, 14, 9, 5, 41, 29, 41, 19, 24, 17, 17, 17, 21, 15, 21, 23, 14, 39, 25, 41, 24, 16, 36, 7, 13, 5, 9, 9, 32, 17, 8, 19, 18, 20, 17, 8, 18, 7, 32, 40, 10, 17, 13, 29, 29, 21, 13, 16, 29, 51, 14, 24, 48, 8, 17, 6, 8, 6, 6, 38, 12, 4, 26, 5, 13, 31, 34, 20, 11, 19, 15, 12, 12, 25, 7, 11, 8, 17, 26, 10, 15, 9, 10, 7, 15, 10, 6, 7, 7, 13, 10, 15, 13, 15, 14, 25, 11, 33, 15, 11, 35, 37, 42, 21, 15, 22, 29, 8, 18, 12, 6, 18, 23, 20, 21, 21, 27, 15, 13, 23, 26, 8, 11, 8, 8, 25, 16, 14, 15, 14, 9, 8, 7, 6, 10, 8, 8, 11, 38, 4, 15, 10, 11, 29, 21, 40, 20, 28, 25, 24, 18, 10, 15, 12, 10, 12, 8, 26, 20, 29, 9, 9, 23, 12, 9, 6, 12, 11, 14, 13, 29, 15, 23, 21, 9, 43, 8, 13, 12, 23, 20, 14, 22, 38, 10, 15, 21, 21, 21, 27, 25, 24, 8, 21, 17, 14, 6, 10, 22, 9, 21, 22, 13, 19, 13, 17, 10, 23, 13, 18, 14, 13, 23, 11, 26, 13, 33, 18, 27, 16, 27, 17, 13, 18, 17, 14, 13, 16, 30, 8, 8, 15, 11, 17, 26, 24, 26, 6, 16, 16, 23, 25, 20, 18, 16, 20, 16, 12, 32, 16, 14, 13, 14, 19, 9, 5, 9, 11, 8, 27, 11, 35, 24, 36, 15, 15, 16, 18, 15, 32, 21, 16, 29, 4, 16, 15, 34, 28, 15, 23, 16, 12, 27, 10, 17, 18, 21, 27, 29, 11, 32, 27, 10, 24, 19, 10, 34, 16, 23, 27, 17, 12, 20, 20, 21, 21, 11, 18, 8, 20, 24, 22, 16, 22, 37, 10, 27, 15, 17, 29, 26, 22, 14, 24, 13, 33, 26, 22, 12, 27, 11, 18, 24, 33, 14, 12, 17, 16, 26, 16, 22, 17, 8, 10, 11, 7, 3, 21, 8, 19, 28, 15, 5, 30, 15, 11, 30, 15, 17, 20, 3, 26, 37, 9, 18, 17, 25, 11, 10, 18, 15, 10, 14, 10, 12, 12, 8, 8, 15, 18, 8, 14, 23, 7, 30, 13, 42, 29, 31, 13, 9, 10, 12, 12, 24, 18, 7, 6, 11, 16, 18, 5, 39, 13, 22, 7, 9, 6, 8, 12, 8, 17, 16, 28, 16, 20, 19, 32, 17, 11, 12, 8, 11, 11, 34, 39, 14, 25, 27, 16, 15, 10, 16, 25, 8, 15, 17, 14, 21, 20, 18, 16, 27, 14, 11, 22, 10, 33, 24, 23, 6, 19, 12, 13, 16, 45, 22, 18, 24, 18, 26, 9, 23, 18, 34, 20, 14, 27, 28, 21, 69, 25, 20, 26, 25, 40, 9, 14, 27, 28, 19, 31, 42, 16, 26, 8, 15, 25, 17, 16, 7, 26, 32, 18, 26, 21, 16, 6, 30, 23, 15, 11, 26, 27, 38, 26, 28, 17, 58, 13, 18, 34, 34, 8, 42, 47, 24, 47, 26, 13, 23, 51, 25, 25, 12, 19, 32, 21, 17, 16, 25, 32, 20, 30, 14, 21, 28, 30, 30, 30, 57, 15, 41, 14, 29, 16, 13, 35, 26, 6, 13, 34, 17, 24, 23, 28, 5, 41, 18, 13, 16, 11, 12, 5, 18, 33, 34, 11, 27, 5, 3, 23, 11, 11, 17, 21, 37, 6, 42, 16, 9, 18, 11, 23, 40, 6, 27, 22, 18, 24, 13, 26, 12, 37, 15, 14, 8, 28, 28, 19, 14, 6, 5, 6, 17, 39, 10, 25, 18, 21, 41, 10, 4, 15, 14, 5, 20, 16, 16, 10, 12, 33, 14, 33, 19, 32, 9, 17, 9, 29, 15, 5, 26, 5, 4, 16, 29, 25, 34, 16, 6, 28, 33, 11, 14, 7, 26, 6, 16, 10, 35, 12, 22, 13, 20, 13, 16, 17, 13, 7, 29, 22, 22, 14, 8, 23, 17, 23, 22, 11, 8, 43, 18, 29, 38, 17, 27, 33, 21, 13, 11, 24, 8, 32, 18, 46, 21, 30, 32, 29, 40, 33, 33, 37, 34, 31, 17, 17, 8, 20, 10, 20, 11, 33, 10, 18, 21, 6, 26, 12, 19, 10, 16, 9, 9, 4, 15, 4, 11, 15, 13, 10, 13, 12, 13, 16, 24, 12, 11, 21, 15, 25, 17, 21, 13, 11, 31, 6, 12, 12, 13, 13, 18, 7, 10, 20, 21, 19, 16, 16, 10, 33, 24, 38, 23, 16, 7, 8, 15, 9, 3, 11, 17, 17, 31, 29, 38, 8, 14, 18, 20, 13, 6, 10, 10, 28, 30, 9, 19, 21, 14, 12, 6, 23, 41, 18, 24, 28, 11, 8, 5, 29, 20, 16, 10, 4, 8, 14, 11, 12, 15, 8, 15, 24, 21, 9, 34, 13, 19, 17, 23, 18, 16, 27, 16, 11, 4, 10, 16, 20, 12, 21, 23, 15, 28, 26, 20, 6, 26, 19, 11, 16, 31, 31, 32, 14, 14, 24, 14, 23, 38, 24, 26, 19, 12, 37, 8, 13, 21, 21, 8, 13, 18, 22, 8, 29, 15, 6, 12, 27, 45, 25, 32, 39, 22, 5, 28, 27, 21, 16, 8, 10, 19, 49, 25, 21, 16, 12, 11, 14, 21, 24, 21, 38, 15, 30, 12, 29, 44, 27, 33, 26, 14, 34, 26, 25, 17, 12, 35, 52, 21, 12, 22, 21, 23, 27, 16, 22, 36, 21, 12, 22, 41, 11, 31, 7, 25, 21, 10, 34, 17, 19, 9, 7, 25, 48, 7, 14, 42, 16, 11, 23, 19, 12, 20, 19, 27, 29, 33, 15, 33, 27, 19, 16, 20, 2, 14, 11, 9, 16, 12, 23, 15, 8, 7, 19, 4, 12, 13, 16, 21, 22, 23, 8, 10, 26, 10, 23, 15, 32, 22, 28, 8, 26, 26, 19, 21, 33, 15, 11, 17, 15, 17, 18, 19, 16, 26, 7, 17, 41, 11, 11, 18, 38, 24, 8, 37, 17, 8, 13, 16, 8, 19, 22, 29, 21, 23, 19, 7, 18, 5, 42, 44, 10, 24, 27, 20, 30, 15, 18, 10, 31, 11, 27, 22, 8, 21, 15, 17, 10, 20, 10, 28, 38, 20, 22, 12, 9, 15, 18, 28, 9, 18, 31, 26, 10, 14, 13, 4, 7, 21, 15, 12, 14, 12, 18, 23, 23, 27, 49, 7, 16, 12, 7, 8, 11, 14, 15, 18, 4, 12, 23, 20, 30, 29, 21, 34, 33, 10, 47, 23, 8, 12, 15, 20, 20, 9, 36, 34, 23, 21, 17, 15, 20, 22, 24, 25, 39, 31, 26, 19, 12, 14, 22, 28, 19, 60, 50, 17, 33, 11, 33, 27, 15, 17, 19, 32, 31, 10, 22, 9, 17, 22, 14, 9, 12, 11, 5, 6, 17, 12, 21, 9, 33, 38, 12, 19, 30, 19, 22, 42, 29, 31, 19, 27, 10, 36, 26, 21, 28, 27, 13, 50, 18, 21, 11, 6, 13, 13, 10, 23, 11, 30, 20, 27, 21, 14, 17, 10, 24, 17, 17, 23, 12, 24, 15, 20, 6, 17, 17, 19, 9, 5, 14, 8, 22, 26, 9, 33, 29, 15, 12, 15, 11, 10, 14, 17, 4, 15, 21, 16, 19, 12, 17, 11, 15, 9, 35, 23, 15, 12, 11, 10, 11, 14, 20, 10, 9, 20, 33, 11, 17, 23, 22, 23, 14, 16, 15, 31, 47, 17, 21, 31, 16, 9, 28, 43, 34, 11, 46, 30, 28, 23, 30, 18, 25, 21, 22, 24, 9, 13, 29, 18, 26, 20, 5, 31, 22, 15, 6, 8, 17, 41, 32, 16, 41, 21, 7, 15, 26, 24, 15, 17, 10, 25, 12, 31, 29, 16, 16, 21, 13, 27, 16, 6, 29, 24, 36, 33, 21, 14, 6, 16, 9, 19, 22, 40, 25, 19, 8, 28, 22, 27, 25, 30, 21, 14, 18, 43, 25, 29, 11, 28, 30, 9, 26, 26, 51, 9, 20, 8, 14, 17, 11, 18, 17, 14, 28, 21, 21, 28, 24, 6, 19, 28, 7, 17, 23, 17, 23, 11, 38, 24, 13, 21, 19, 21, 23, 11, 24, 10, 21, 7, 14, 39, 14, 14, 20, 17, 8, 20, 43, 18, 20, 23, 14, 18, 38, 19, 20, 27, 17, 14, 16, 12, 25, 20, 39, 19, 20, 29, 13, 13, 5, 20, 15, 12, 23, 10, 10, 44, 16, 32, 19, 23, 21, 12, 5, 21, 37, 11, 7, 23, 32, 11, 35, 27, 15, 14, 19, 17, 40, 23, 30, 43, 13, 20, 26, 23, 15, 6, 28, 8, 9, 17, 22, 10, 36, 17, 18, 20, 19, 27, 4, 9, 27, 30, 27, 24, 14, 25, 32, 29, 30, 11, 8, 14, 5, 11, 9, 15, 24, 17, 7, 16, 18, 19, 13, 22, 8, 15, 9, 15, 23, 9, 19, 11, 7, 29, 22, 20, 9, 16, 16, 7, 13, 13, 13, 12, 9, 20, 7, 23, 16, 18, 8, 21, 14, 21, 8, 23, 16, 25, 15, 10, 21, 17, 17, 24, 20, 13, 16, 16, 26, 28, 9, 14, 14, 25, 30, 14, 20, 18, 17, 18, 16, 14, 10, 32, 17, 17, 26, 16, 6, 17, 12, 18, 24, 25, 44, 27, 26, 24, 33, 14, 5, 23, 12, 25, 30, 8, 18, 18, 22, 7, 34, 5, 41, 9, 27, 19, 34, 5, 7, 12, 10, 12, 15, 36, 9, 9, 33, 35, 18, 29, 29, 19, 14, 11, 40, 10, 6, 9, 12, 24, 11, 13, 16, 56, 12, 20, 23, 18, 39, 19, 26, 14, 25, 9, 34, 8, 5, 22, 15, 18, 29, 27, 10, 42, 22, 10, 31, 26, 36, 48, 50, 16, 29, 34, 12, 36, 14, 14, 28, 31, 34, 43, 40, 16, 29, 16, 36, 18, 27, 23, 16, 33, 13, 35, 28, 20, 8, 10, 9, 29, 30, 16, 9, 10, 9, 12, 13, 22, 29, 34, 23, 15, 17, 10, 13, 17, 13, 8, 19, 7, 14, 14, 10, 13, 9, 10, 11, 17, 9, 8, 15, 27, 8, 26, 13, 5, 6, 10, 8, 16, 4, 10, 17, 10, 25, 24, 5, 27, 7, 20, 27, 7, 8, 12, 14, 9, 20, 24, 5, 7, 9, 16, 14, 23, 10, 17, 16, 7, 13, 20, 8, 20, 8, 15, 2, 18, 10, 29, 12, 24, 11, 3, 15, 18, 11, 13, 33, 9, 19, 22, 11, 19]

class DependencyParseNL():
    def __init__(self, parser_output):
        self.dep_parse = parser_output['sentences'][0]['basicDependencies']
        self.arcs = self.get_arcs()
        self.length = len(self.arcs)
        self.tree = self.construct_tree()
        self.depth = self.get_tree_depth(self.tree)

    def get_arcs(self):
        arcs = []
        for dep in self.dep_parse:
            arcs += [(dep['dependent'], dep['governor'])]
        return(arcs)

    def construct_tree(self):
        """
        Given a list of edges [child, parent], return trees.
        from http://xahlee.info/python/python_construct_tree_from_edge.html
        """

        trees = collections.defaultdict(dict)

        for child, parent in self.arcs:
            trees[parent][child] = trees[child]

        # Find roots
        children, parents = zip(*self.arcs)
        roots = set(parents).difference(children)

        return ({root: trees[root] for root in roots})

    def get_tree_depth(self, tree):
        if tree == {}:
            # Account for ROOT by taking -1 instead of 0
            return -1
        else:
            return(max([self.get_tree_depth(tree[subtree]) for subtree in tree]) + 1)

class DependencyParsePCFG():
    def __init__(self, sample, alphabet):
        self.sample = self.process_sample(sample)
        self.length = sum([1 for item in self.sample if not item in ['(', ')'] ]) # ignore brackets
        self.leaves = alphabet
        #self.nr_leaves = len([item for item in self.sample if item in self.leaves])
        self.depth = self.get_tree_depth(self.sample)

    def process_sample(self, sample):
        input = sample.split('\t')[0]
        input = input.split()
        return(input)

    def get_tree_depth(self, sample):
        open_bracket_counts = []
        open_brackets = 0
        for item in sample:
            if item == '(':
                open_brackets += 1
            elif item == ')':
                open_brackets -= 1
            open_bracket_counts += [open_brackets]

        depth = max(open_bracket_counts) + 1
        return(depth)

class DataNaturalization():
    def __init__(self, alphabet, unary_functions, binary_functions):
        sys.setrecursionlimit(500)
        self.alphabet = alphabet
        self.unary_functions = unary_functions
        self.binary_functions = binary_functions

    def get_tree_statistics(self, file, type):
        depths = []
        lengths = []

        if type == 'nl':
            dl = DataLoader(file)
            dl.load_data()
            nlp = StanfordCoreNLP('http://localhost:9000')

            for idx, sentence in enumerate(dl.data):
                if idx % 1000 == 0:
                    print('Processing sentence ' + str(idx + 1))
                output = nlp.annotate(sentence, properties={
                    'annotators': 'tokenize,ssplit,depparse',
                    'outputFormat': 'json'
                })

                d = DependencyParseNL(output)
                depths += [d.depth]
                lengths += [d.length]

        elif type == 'pcfg':
            with open(file, 'r') as f:
                for idx, line in enumerate(f):
                    d = DependencyParsePCFG(line, self.alphabet)
                    depths += [d.depth]
                    lengths += [d.length]

        return(depths, lengths)

    def plot_dist(self, var1, var2, name1, name2):
        sns.set(style="white", color_codes=True, font_scale=1.5)
        df = pd.DataFrame({name1: np.array(var1), name2: np.array(var2)})
        g = sns.JointGrid(x=name1, y=name2, data=df, space=0, xlim=(0,15), ylim=(0,40))
        g = g.plot_joint(sns.kdeplot, shade_lowest=False, shade=True)

        sns.kdeplot(df[name1], color="b", shade=True, ax=g.ax_marg_x, legend=False, bw=0.25)
        sns.kdeplot(df[name2], color="b", shade=True, vertical=True, ax=g.ax_marg_y, legend=False)

        #g.ax_joint.legend_.remove()
        g.ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(5))
        g.ax_joint.yaxis.set_major_locator(ticker.MultipleLocator(10))

        g.fig.set_dpi(300)

        plt.savefig('dist_wmt_largefont.pdf', format='pdf')

    def kl_divergence(self, mean1, cov1, mean2, cov2):
        kl_div = 0.5 * (
            np.trace(np.linalg.inv(cov2) @ cov1)
            + (np.transpose(mean2 - mean1) @ np.linalg.inv(cov2) @ (mean2 - mean1))
            - 2
            + np.log(np.linalg.det(cov2) / np.linalg.det(cov1)))
        return(kl_div)

    def get_pcfg_params(self, file):
        # Maximum Likelihood Estimation
        samples = open(file, 'r').readlines()
        inputs = [sample.split('\t')[0].split() for sample in samples]

        unary_names = ['copy', 'shift', 'echo', 'reverse']
        binary_names = ['append', 'prepend']

        nr_samples = len(inputs)
        unary_count, binary_count, string_count = 0, 0, 0
        for input in inputs:
            within_string = False
            for item in input:
                if item in unary_names:
                    unary_count += 1
                elif item in binary_names:
                    binary_count += 1
                elif item in self.alphabet:
                    if not within_string:
                        string_count += 1
                        within_string = True
                elif item in [')', ',']:
                    within_string = False

        function_count = unary_count + binary_count
        prob_unary = unary_count / function_count
        prob_func = (function_count - nr_samples) / (function_count - nr_samples + string_count)
        return(prob_unary, prob_func)

    def force_dist_on_data(self, data_gold_dist, data_to_be_transformed, depth_interval, length_interval):
        """
        Using depth and length intervals, consider distribution not per (depth, length)
        combination, but in regions, for otherwise too many data instances would be discarded.
        """

        if not data_gold_dist is None:
            # To infer from file instead of using stored
            depths_nl, lengths_nl = self.get_tree_statistics(data_gold_dist, type='nl')
        else:
            depths_nl, lengths_nl = DEPTHS_WMT_TEST, LENGTHS_WMT_TEST

        #print(depths_nl, lengths_nl)
        depths_lengths_comb_nl = zip(depths_nl, lengths_nl)
        depths_lengths_data_nl = {}

        for comb in depths_lengths_comb_nl:
            depth_cat = comb[0] // depth_interval
            length_cat = comb[1] // length_interval
            if (depth_cat, length_cat) in depths_lengths_data_nl:
                depths_lengths_data_nl[(depth_cat, length_cat)] += 1
            else:
                depths_lengths_data_nl[(depth_cat, length_cat)] = 1
        sorted_depth_lengths_nl = sorted(depths_lengths_data_nl.items(), key=operator.itemgetter(1), reverse=True)
        most_likely_comb, highest_freq = sorted_depth_lengths_nl[0]

        pcfg_data = open(data_to_be_transformed, 'r').readlines()
        depths_pcfg, lengths_pcfg = self.get_tree_statistics(data_to_be_transformed, type='pcfg')
        depths_lengths_comb_pcfg = zip(depths_pcfg, lengths_pcfg)
        depths_lengths_data_pcfg = {}

        for idx, comb in enumerate(depths_lengths_comb_pcfg):
            depth_cat = comb[0] // depth_interval
            length_cat = comb[1] // length_interval
            if (depth_cat, length_cat) in depths_lengths_data_pcfg:
                depths_lengths_data_pcfg[(depth_cat, length_cat)] += [pcfg_data[idx]]
            else:
                depths_lengths_data_pcfg[(depth_cat, length_cat)] = [pcfg_data[idx]]
        pcfg_size_most_likely_comb = len(depths_lengths_data_pcfg[most_likely_comb])

        transformed_data = []
        for comb in depths_lengths_data_pcfg:
            if comb in depths_lengths_data_nl:
                include_size = int((depths_lengths_data_nl[comb] / highest_freq) * pcfg_size_most_likely_comb)
                transformed_data += depths_lengths_data_pcfg[comb][:include_size]

        output_file = data_to_be_transformed.split('.')[0] + '_transformed_intervals_depth_'+ str(depth_interval) + '_length_' + str(length_interval) +'.txt'

        with open(output_file, 'w') as f:
            for sample in transformed_data:
                f.write(sample)

        depths_trans, length_trans = self.get_tree_statistics(output_file, type='pcfg')
        array_trans = np.array(list(zip(depths_trans, length_trans)))
        array_nl = np.array(list(zip(depths_nl, lengths_nl)))

        # Compute KL divergence
        mean_nl, cov_nl = np.mean(array_nl, axis=0), np.cov(array_nl, rowvar=0)
        mean_trans, cov_trans = np.mean(array_trans, axis=0), np.cov(array_trans, rowvar=0)

        kl_div = self.kl_divergence(mean_nl, cov_nl, mean_trans, cov_trans)
        print('Depth interval: ' + str(depth_interval))
        print('Length interval: ' + str(length_interval))
        print('Nr of samples: ' + str(len(depths_trans)))
        print('KL divergence: ' + str(kl_div))
        print('##################################')

        #self.plot_dist(depths_trans, length_trans, 'depth', 'length')
        #return (transformed_data)
        return(kl_div, output_file)

    def finalize(self, file, factor=1, remove_brackets=True, add_args=True, output_file=True, plot_dist=False):
        new_file_brackets = file.split('.')[0] + '_times' + str(factor) + '_brackets.txt'
        if remove_brackets:
            new_file = file.split('.')[0] + '_times' + str(factor) + '.txt'

        with open(file, 'r') as f:
            if add_args:
                args_used = []
            new_lines = []
            for idx, line in enumerate(f):
                line = line.split()
                for i in range(factor):
                    new_line = []
                    arg_count = 0
                    for item in line:
                        if arg_count != 0 and item != 'X':
                            try:
                                if add_args:
                                    candidate_arg = [random.choice(self.alphabet) for i in range(arg_count)]
                                    while candidate_arg in args_used:
                                        candidate_arg = [random.choice(self.alphabet) for i in range(arg_count)]
                                    new_line += candidate_arg
                                    args_used += [candidate_arg]
                                    arg_count = 0
                                else:
                                    new_line += ['X' for i in range(arg_count)]
                                    arg_count = 0
                            except:
                                break
                        if factor > 1:
                            if item in self.unary_functions:
                                new_line += [random.choice(self.unary_functions)]
                            elif item in self.binary_functions:
                                 new_line += [random.choice(self.binary_functions)]
                        elif item == 'X':
                            arg_count += 1
                        else:
                            new_line += [item]
                    try:
                        output = interpret(new_line)
                    except:
                        break

                    new_lines += [' '.join(new_line) + '\t' + ' '.join(output) + '\n']

        if output_file:
            with open(new_file_brackets, 'w') as nf_brackets:
                for line in new_lines:
                    nf_brackets.write(line)
                print('Final file with brackets located at: ')
                print(new_file_brackets)

            if remove_brackets:
                with open(new_file, 'w') as nf:
                    for line in new_lines:
                        nf.write(line.replace('( ', '').replace(' )', ''))
                print('Final file without brackets located at: ')
                print(new_file)

        if plot_dist:
            depths, lengths = self.get_tree_statistics(new_file_brackets, type='pcfg')
            self.plot_dist(depths, lengths, 'depth', 'length')

        total_nr_str_sequences = len(args_used)
        print('Total nr of string sequences: ' + str(total_nr_str_sequences))

# dn = DataNaturalization(alphabet=None, unary_functions=None, binary_functions=None)
# depth, length = dn.get_tree_statistics('data/pcfg_set/10K/pcfg_10funcs_520letters_brackets.txt', type='pcfg')
# depth = [d - 1 for d in depth]
# dn.plot_dist(depth, length, 'depth', 'length')

# dn = DataNaturalization(alphabet=None, unary_functions=None, binary_functions=None)
# depth, length = dn.get_tree_statistics('/Users/mathijs/Documents/Studie/AI/Thesis/data/wmt_ende_sp/test.en', type='nl')
# depth = [d - 1 for d in depth]
# dn.plot_dist(depth, length, 'depth', 'length')
