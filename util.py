#import json, collections, numpy as np
#from nltk.corpus import stopwords
#from sklearn.feature_extraction.text import TfidfVectorizer
#import pickle


class User:

    def __init__(self, userJson):
        self.id = userJson['user_id']
        self.name = userJson['name']
        self.yelpReviewCount = userJson['review_count']
        self.yelpAverageStars = userJson['average_stars']
        self.extractedAverageStars = None
        self.votes = userJson['votes']
        self.reviews = []

    def addReview(self, review):
        self.reviews.append(review)
        review.user = self

    def getId(self):
        return self.id

    def setId(self, Id):
        self.id = Id

    def setName(self, name):
        self.name = name

    def getName(self):
        return self.name

    def getYelpReviewCount(self):
        return self.yelpReviewCount

    def getExtractedReviewCount(self):
        return len(self.reviews)

    def getYelpAverageStars(self):
        return self.yelpAverageStars

    def getReviews(self):
        return self.reviews

    def findAverageStars(self):
        if self.extractedAverageStars is not None:
            return self.extractedAverageStars
        else:
            reviews = self.getReviews()
            if reviews:
                totalStars = 0
                for review in reviews:
                    totalStars += review.stars
                averageStars = 1.0*totalStars/len(reviews)
                self.extractedAverageStars = averageStars
                return averageStars
            else:
                return None

    def getVotes(self):
        return self.votes

    def __str__(self):
        return 'id: {} name: {} averageStars: {} reviewCount: {} votes: {} reviews: {}'.format(self.id, self.name,
            self.extractedAverageStars, self.getExtractedReviewCount(), self.votes, self.reviews)


class Biz:

    def __init__(self, bizJson):
        self.id = bizJson['business_id']
        self.name = bizJson['name']
        self.neighborhoods = set(bizJson['neighborhoods'])
        self.city = bizJson['city']
        self.state = bizJson['state']
        self.lat = bizJson['latitude']
        self.lon = bizJson['longitude']
        self.attributes = bizJson['attributes']
        self.yelpStars = bizJson['stars']
        self.extractedAverageStars = None
        self.yelpReviewCount = bizJson['review_count']
        self.categories = set(bizJson['categories'])
        self.open = bizJson['open']
        self.reviews = []

    def setId(self, Id):
        self.id = Id

    def getId(self):
        return self.id

    def setName(self, name):
        self.name = name

    def getName(self):
        return self.name

    def getNeighborhoods(self):
        return self.neighborhoods

    def getCity(self):
        return self.city

    def getState(self):
        return self.state

    def getLat(self):
        return self.lat

    def getLon(self):
        return self.lon

    def getYelpStars(self):
        return self.yelpStars

    def getCategories(self):
        return self.categories

    def getOpen(self):
        return self.open

    def getReviews(self):
        return self.reviews

    def getExtractedReviewCount(self):
        return len(self.reviews)

    def findAverageStars(self):
        if self.extractedAverageStars is not None:
            return self.extractedAverageStars
        else:
            reviews = self.getReviews()
            if reviews:
                totalStars = 0
                for review in reviews:
                    totalStars += review.stars
                averageStars = 1.0*totalStars/len(reviews)
                self.extractedAverageStars = averageStars
                return averageStars
            else:
                return None

    def addReview(self, review):
        self.reviews.append(review)
        review.biz = self

    def __str__(self):
        return 'id: {}\nname: {}\nneighborhoods: {}\ncity: {}\n'.format(self.id, self.name, self.neighborhoods, self.city) + \
               ' state: {}\nlat: {}\nlon: {}stars: {}\n'.format(self.state, self.lat, self.lon, self.extractedAverageStars) + \
            'reviewCount: {}\ncategories: {}\nopen: {}\nreviews: {}'.format(self.getExtractedReviewCount(), self.categories, self.open, self.reviews)


class Review:

    def __init__(self, reviewJson):
        self.bizId = reviewJson['business_id']
        self.userId = reviewJson['user_id']
        self.id = self.userId + self.bizId
        self.stars = reviewJson['stars']
        self.text = reviewJson['text']
        self.date = reviewJson['date']
        self.votes = reviewJson['votes']
        self.user = None
        self.biz = None

    def getBizId(self):
        return self.bizId

    def getUserId(self):
        return self.userId

    def getStars(self):
        return self.stars

    def getText(self):
        return self.text

    def setText(self, text):
        self.text = text

    def getId(self):
        return self.id

    def getDate(self):
        return self.date

    def getUser(self):
        return self.user

    def getBiz(self):
        return self.biz

    def __str__(self):
        return 'bizId: {}\nuserId: {}\nstars: {}\ndate'.format(self.bizId, self.userId, self.stars, self.date) + \
            'votes: {}\ntext: {}, user: {}\nbiz: {}'.format(self.votes, self.text, self.user, self.biz)

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
