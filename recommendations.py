import json, sklearn, pickle, random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

random.seed(42)

def getCosineSimilarityMatrix(reviewTextArray):
    X = TfidfVectorizer().fit_transform(reviewTextArray)
    return linear_kernel(X, X)

def getDivergentBizs(user):
    pastReviews = user['reviews']
    combinedText = ""
    for review in pastReviews:
        combinedText += "\n\n\n\"" + review['text']
    





users = pickle.load(open('user_list'))
bizs = pickle.load(open('business_list'))
reviews = pickle.load(open('review_list'))
# bizIdToReview = pickle.load('biz_id_to_review')
# bizIdToText = pickle.load('biz_id_to_review_text')

# hist = np.histogram(reviews, bins=[1,10,30,50,100,300,800])
#NLTK - NLP library
bizIds = []
for review in reviews:
    bizIds.append(review['business_id'])

bizIdToReviewText = dict.fromkeys(bizIds, "")
for review in reviews:
    bizIdToReviewText[review["business_id"]] += "\n\n\n" + review['text'].strip()

bizIds = np.array(bizIdToReviewText.keys())
# combinedTexts = np.array(bizIdToReviewText.values())
combinedTexts = []
for bizId in bizIds:
    combinedTexts.append(bizIdToReviewText[bizId])
data = np.mat([np.transpose(bizIds), np.transpose(combinedTexts)])

#data[0, :] is the array of all biz id's
#data[1, :] is the array of all the reviews

cosSim = getCosineSimilarityMatrix(combinedTexts)
print max([(cosSim, index) for index, cosSim in enumerate(cosSim[0][1:])])
# nZeros = 0
# for i in xrange(len(cos_sim)):
#     for j in xrange(len(cos_sim[0])):
#         if cos_sim[i, j] == 0:
#             nZeros += 1
#
# print cos_sim
# print 1.0*nZeros/(len(cos_sim)*len(cos_sim[0]))

queryUser = random.choice(users)
pastBizsBySimilarity = getDivergentBizs(bizIds, queryUser)