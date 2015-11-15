import json, sklearn, pickle, random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import csr_matrix

random.seed(42)

# def getCosineSimilarityMatrix(reviewTextArray):
#     X = TfidfVectorizer().fit_transform(reviewTextArray)
#     return linear_kernel(X, X)

def getDivergentBizs(user):
    pastReviews = user['reviews']

#Can be used for user or restaurant
def setCombinedText(x):
    if 'combined_text' in x.keys():
        return
    else:
        combinedText = ""
        for review in x['reviews']:
            combinedText += "\n\n\n\"" + review['text']
        x['combined_text'] = combinedText

#Can be used for user or restaurant
def getReviewIds(x):
    reviewIds = []
    for review in x['reviews']:
        reviewIds.append(review['review_id'])
    return reviewIds

#adds all the reviews for
def combineVectors(x, vectorizedReviewTexts, reviewIdToIndex):
    xReviewIds = getReviewIds(x)
    reviewIndexes = []
    for reviewId in xReviewIds:
        reviewIndexes.append(reviewIdToIndex[reviewId])
    combinedVecs = csr_matrix(np.zeros(vectorizedReviewTexts[0].shape))
    for reviewIndex in reviewIndexes:
        combinedVecs = combinedVecs + vectorizedReviewTexts[reviewIndex]
    return combinedVecs

#Can be used for any combination of users or restaurants
def findSimilarity(x, y, vectorizedReviewTexts, reviewIdToIndex):
    xCombinedVectors = combineVectors(x, vectorizedReviewTexts, reviewIdToIndex)
    yCombinedVectors = combineVectors(y, vectorizedReviewTexts, reviewIdToIndex)
    return cosine_similarity(xCombinedVectors, yCombinedVectors)


users = pickle.load(open('user_list'))
bizs = pickle.load(open('business_list'))
reviews = pickle.load(open('review_list'))
# bizIdToReview = pickle.load('biz_id_to_review')
# bizIdToText = pickle.load('biz_id_to_review_text')

# hist = np.histogram(reviews, bins=[1,10,30,50,100,300,800])
#NLTK - NLP library
reviewIdToIndex = {}
reviewIds = []
reviewCorpus = []
for i in xrange(len(reviews)):
    reviewIds.append(reviews[i]['review_id'])
    reviewIdToIndex[reviews[i]['review_id']] = i
    reviewCorpus.append(reviews[i]['text'])

# combinedTexts = np.array(bizIdToReviewText.values())
data = np.mat([np.transpose(reviewIds), np.transpose(reviewCorpus)])

#data[0, :] is the array of all biz id's
#data[1, :] is the array of all the reviews

vectorizedReviewTexts = TfidfVectorizer().fit_transform(reviewCorpus)
# cosSim = linear_kernel(vectorizedReviewTexts, vectorizedReviewTexts)
# print cosSim
# print max([(cosSim, index) for index, cosSim in enumerate(cosSim[0][1:])])
# nZeros = 0
# for i in xrange(len(cos_sim)):
#     for j in xrange(len(cos_sim[0])):
#         if cos_sim[i, j] == 0:
#             nZeros += 1
#
# print cos_sim
# print 1.0*nZeros/(len(cos_sim)*len(cos_sim[0]))

queryUser = random.choice(users)
neighborIndexesBySim = []
for i in xrange(len(users)):
    if users[i] is not queryUser:
        sim = findSimilarity(users[i], queryUser, vectorizedReviewTexts, reviewIdToIndex)
        neighborIndexesBySim.append((sim, i))

neighborIndexesBySim.sort(reverse=True)
for neighbor in neighborIndexesBySim:
    print neighbor

nearestNeighbor = users[neighborIndexesBySim[1]]
