#import json, collections, numpy as np
import json, collections
#from nltk.corpus import stopwords
#from sklearn.feature_extraction.text import TfidfVectorizer
#import pickle

class User:

    def __init__(self, userJson):
        self.id = userJson['user_id']
        self.name = userJson['name']
        self.reviewCount = userJson['review_count']
        self.averageStars = userJson['average_stars']
        self.votes = userJson['votes']
        self.reviews = []

    def addReview(self, review):
        self.reviews.append(review)
        review.user = self

class Biz:

    def __init__(self, bizJson):
        self.id = bizJson['business_id']
        self.name = bizJson['name']
        self.neighborhoods = bizJson['neighborhoods']
        self.city = bizJson['city']
        self.state = bizJson['state']
        self.lat = bizJson['latitude']
        self.lon = bizJson['longitude']
        self.stars = bizJson['stars']
        self.reviewCount = bizJson['review_count']
        self.categories = bizJson['categories']
        self.open = bizJson['open']
        self.reviews = []

    def addReview(self, review):
        self.reviews.append(review)
        review.biz = self

class Review:

    def __init__(self, reviewJson):
        self.bizId = reviewJson['business_id']
        self.userId = reviewJson['user_id']
        self.stars = reviewJson['stars']
        self.text = reviewJson['text']
        self.date = reviewJson['date']
        self.votes = reviewJson['votes']
        self.user = None
        self.biz = None



'''

def getUsers(directory, minReviews = 100, maxUsers = -1):
    userRead = open(directory)
    users = []
    n = 0
    for line in userRead:
        user = json.loads(line)
        if user['review_count'] >= minReviews:
            users.append(user)
        n += 1
        if n == maxUsers:
            break
    return users

def getBizs(directory, maxBizs = -1, city = None):
    bizRead = open(directory)
    bizs = []
    n = 0
    for line in bizRead:
        biz = json.loads(line)
        if 'Restaurants' in biz['categories'] and (city == None or city == biz['city']):
            bizs.append(biz)
            n += 1
            if n == maxBizs:
                break
    return bizs

def getReviews(directory, maxReviews = -1):
    reviewsRead = open(directory)
    reviews = []
    bizIdToReviews = dict.fromkeys(all, "")
    n = 0
    for line in reviewsRead:
        review = json.loads(line)
        review["text"] = review["text"].lower()
        reviews.append(review)
        bizIdToReviews[review['business_id']] = review
        n += 1
        if n == maxReviews:
            break
    bizIdToText = {}
    #for bizId in bizIdToReviews.keys():

    return reviews, bizIdToReviews

'''

# def extractTextFeatures(bizIdToRewiews):
#     wordCounter = collections.Counter()
#     for review in bizIdToRewiews
#     for word in review.split():
#         wordCounter[word.strip()] += 1
#     words = wordCounter.keys()
#     feature = np.zeros(len(words))
#     for i in xrange(len(words)):
#         feature[i] = wordCounter[words[i]]
#     return words, feature
#
#
# def getBizClusterSimilarity(cluster):
#     categoryCounts = collections.Counter()
#     for biz in cluster:
#         for cat in biz['categories']:
#             if cat != 'Restaurants':
#                 categoryCounts[cat] += 1
#     categories = categoryCounts.values().sort()
#     mostPopularCategory = max(categoryCounts, key=categoryCounts.get)
#     nBizInCategory = 0
#     for biz in cluster:
#         if mostPopularCategory in biz['categories']:
#             nBizInCategory += 1
#     similarity = 1.0*nBizInCategory/len(cluster)
#     assert similarity <= 1
#     return similarity