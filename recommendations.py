import json, sklearn, pickle, random, copy, collabf, csp, util, regressor, math
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import csr_matrix
import pca, heatplot
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

random.seed(42)

# def getCosineSimilarityMatrix(reviewTextArray):
#     X = TfidfVectorizer().fit_transform(reviewTextArray)
#     return linear_kernel(X, X)

def evaluateRecommendations(user, divRecommendations, removedBizs, bizs, vectorizedReviewTexts, reviewIdToIndex):
    combinedUser = combineVectors(user, vectorizedReviewTexts, reviewIdToIndex)
    randomBizs = []
    n = 0
    limit = xrange(len(divRecommendations))
    for biz in bizs:
        combinedBiz = combineVectors(biz, vectorizedReviewTexts, reviewIdToIndex)
        sim = cosine_similarity(combinedUser, combinedBiz)
        if sim < 0.8:
            randomBizs.append(biz)
            n += 1
            if n == limit:
                break
    nRecMatches = 0
    nRandomMatches = 0
    for removedBiz in removedBizs:
        if removedBiz in divRecommendations:
            nRecMatches += 1
        if removedBiz in randomBizs:
            nRandomMatches += 1
    return nRecMatches/nRandomMatches

def makeEvalUser(queryUser, vectorizedReviewTexts, reviewIdToIndex, bizIdToBiz, nRecs):
    queryUserBizs = findUserBizs(queryUser, bizIdToBiz)
    divergentUserBizs = divergentBizs(queryUser, queryUserBizs, vectorizedReviewTexts, reviewIdToIndex, nRecs)
    userCopy = copy.deepcopy(queryUser)
    remainingReviews = []
    divBizIds = set()
    for divBiz in divergentUserBizs:
        divBizIds.add(divBiz['business_id'])
    for review in userCopy['reviews']:
        if review['business_id'] not in divBizIds:
            remainingReviews.append(review)
    userCopy['reviews'] = remainingReviews
    return userCopy, divergentUserBizs


def divergentBizs(queryUser, bizs, vectorizedReviewTexts, reviewIdToIndex, nRecs):
    combinedUser = combineVectors(queryUser, vectorizedReviewTexts, reviewIdToIndex)
    bizsByDist = []
    for i in xrange(len(bizs)):
        combinedBiz = combineVectors(bizs[i], vectorizedReviewTexts, reviewIdToIndex)
        sim = cosine_similarity(combinedUser, combinedBiz)
        bizsByDist.append((sim, i))
    bizsByDist.sort()
    divRecommendations = []
    limit = min(nRecs, len(bizsByDist))
    for i in xrange(limit):
        divRecommendations.append(bizs[bizsByDist[i][1]])
    return divRecommendations


def nearestNeighbors(queryUser, users, vectorizedReviewTexts, reviewIdToIndex):
    neighborIndexesBySim = []
    for i in xrange(len(users)):
        if users[i]['user_id'] != queryUser['user_id']:
            sim = findSimilarity(users[i], queryUser, vectorizedReviewTexts, reviewIdToIndex)
            neighborIndexesBySim.append((sim, i))
        neighborIndexesBySim.sort(reverse=True)
    return neighborIndexesBySim

def findUserBizs(user, bizIdToBiz):
    bizs = []
    for review in user['reviews']:
        bizId = review['business_id']
        if bizId in bizIdToBiz.keys():
            bizs.append(bizIdToBiz[bizId])
    return bizs

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


# bizIdToReview = pickle.load('biz_id_to_review')
# bizIdToText = pickle.load('biz_id_to_review_text')

# hist = np.histogram(reviews, bins=[1,10,30,50,100,300,800])
#NLTK - NLP library

# reviewIdToIndex = {}
# reviewIds = []
# reviewCorpus = []
# for i in xrange(len(reviews)):
#     reviewIds.append(reviews[i]['review_id'])
#     reviewIdToIndex[reviews[i]['review_id']] = i
#     reviewCorpus.append(reviews[i]['text'])
#
# # combinedTexts = np.array(bizIdToReviewText.values())
# data = np.mat([np.transpose(reviewIds), np.transpose(reviewCorpus)])
#
# #data[0, :] is the array of all biz id's
# #data[1, :] is the array of all the reviews
#
# vectorizedReviewTexts = TfidfVectorizer().fit_transform(reviewCorpus)
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

###############  Start code ###############
rec = pickle.load(open('pickledRecommender'))
inArrayForm = []
rec.regress()

#
# minSim = 0.5
# while minSim < 1:
#     error = 0
#     rec.minSim = minSim
#     for i in xrange(10):
#         error += math.sqrt(rec.evalRecommendations())
#     print 'error: {}, minSim: {}'.format(error/5, minSim)
#     minSim += 0.05

# ### Start plotting ###
# users = rec.getUsers()
# bizs = rec.getBizs()
# reviews = rec.getReviews()
# queryUser = random.choice(users)
# # recommendations = rec.recommend([queryUser])
#
# print users[0].id
# numusers = 20
# numrestaurants = 20
#
# # Make a heat plot of restaurant-restaurant similarities
# lenbiz = bizs[0].vectorizedText.shape[1]
# allbizsvectorizedText = np.zeros((numusers,lenbiz))
# for i in xrange(numusers):
#     vector = bizs[i].vectorizedText.toarray()
#     allbizsvectorizedText[i,:]=vector
#
# reslabels = []
# for i in range(numrestaurants):
#     if len(bizs[i].categories)>=2:
#         bizs[i].categories.remove('Restaurants')
#         reslabel = random.sample(bizs[i].categories, 1)[0]
#         reslabel.encode('ascii','ignore')
#     else:
#         reslabel = 'Restaurants'
#     reslabels.append(reslabel)
#
# similaritymatrix = np.zeros((numrestaurants,numrestaurants))
# for i in range(numrestaurants):
#     for j in range(numrestaurants):
#         if i==j:
#             similaritymatrix[i,j] = 1
#         else:
#             similaritymatrix[i,j] = cosine_similarity(allbizsvectorizedText[i,:],allbizsvectorizedText[j,:])
# heatplot.heatplot(similaritymatrix,reslabels,reslabels)
#
# # Make a heat plot of user-user similarities
# len = users[0].vectorizedText.shape[1]
# allusersvectorizedText = np.zeros((numusers,len))
# for i in xrange(numusers):
#     vector = users[i].vectorizedText.toarray()
#     allusersvectorizedText[i,:]=vector
#
# userlabels = []
# for i in range(numusers):
#     userlabels.append('user'+str(i))
#
# similaritymatrix = np.zeros((numusers,numusers))
# for i in range(numusers):
#     for j in range(numusers):
#         if i==j:
#             similaritymatrix[i,j] = 1
#         else:
#             similaritymatrix[i,j] = cosine_similarity(allusersvectorizedText[i,:],allusersvectorizedText[j,:])
# heatplot.heatplot(similaritymatrix,userlabels,userlabels)
#
# # Make a heat plot of user-restaurant similarities
# similaritymatrix = np.zeros((numusers,numrestaurants))
# for i in range(numusers):
#     for j in range(numrestaurants):
#         similaritymatrix[i,j] = cosine_similarity(allbizsvectorizedText[i,:],allusersvectorizedText[j,:])
# heatplot.heatplot(similaritymatrix,reslabels,userlabels)















# Make a PCA plot
# pca.plotpca(allusersvectorizedText)
# pca.plotpca(allbizsvectorizedText)

# bizIdToBiz = {}
# for biz in bizs:
#     bizIdToBiz[biz['business_id']] = biz
#
# queryUser = random.choice(users)
# userForEval, removedBizs = makeEvalUser(queryUser, vectorizedReviewTexts, reviewIdToIndex, bizIdToBiz, 10)
# neighborIndexesBySim = nearestNeighbors(userForEval, users, vectorizedReviewTexts, reviewIdToIndex)
#
# bizIdToBiz = {}
# for biz in bizs:
#     bizIdToBiz[biz['business_id']] = biz
#
# queryUser = random.choice(users)
# userForEval, removedBizs = makeEvalUser(queryUser, vectorizedReviewTexts, reviewIdToIndex, bizIdToBiz, 10)
# neighborIndexesBySim = nearestNeighbors(userForEval, users, vectorizedReviewTexts, reviewIdToIndex)
#
# for neighbor in neighborIndexesBySim:
#     print neighbor
#
# nearestNeighbor = users[neighborIndexesBySim[0][1]]
#
# neighborBizs = findUserBizs(nearestNeighbor, bizIdToBiz)
# divRecommendations = divergentBizs(userForEval, neighborBizs, vectorizedReviewTexts, reviewIdToIndex, 10)
#
# evalScore = evaluateRecommendations(userForEval, divRecommendations, removedBizs, bizs, vectorizedReviewTexts, reviewIdToIndex)
#
# print evalScore
#
