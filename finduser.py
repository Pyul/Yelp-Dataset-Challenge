import json, sklearn, pickle, random, copy, collabf, csp, util, regressor
#import numpy as np
#import pandas as pd
#from collections import Counter
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
#from scipy.sparse import csr_matrix
# import pca, heatplot
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

random.seed(42)

###############  Start code ###############
user = pickle.load(open('user_list'))
i = 8
str = ''
for review in user[i].reviews:
    str = str+review.text
print str

f = open('userrev%d' %(i), 'w')
f.write(str)
