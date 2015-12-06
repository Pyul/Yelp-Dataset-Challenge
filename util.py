import json, collections, numpy as np, random, csp
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

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
        self.vectorizedText = None

    def getVectorizedText(self):
        return self.vectorizedText

    def setVectorizedText(self, vectorizedText):
        self.vectorizedText = vectorizedText

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

class User:

    def __init__(self, userJson):
        self.id = userJson['user_id']
        self.name = userJson['name']
        self.yelpReviewCount = userJson['review_count']
        self.yelpAverageStars = userJson['average_stars']
        self.extractedAverageStars = None
        self.votes = userJson['votes']
        self.reviews = []
        self.vectorizedText = None
        self.reviewedBizIds = set()
        self.reviewedBizs = []
        self.bizIdToReview = {}

    def addReview(self, review):
        self.reviews.append(review)
        review.user = self
        self.reviewedBizIds.add(review.bizId)
        self.reviewedBizs.append(review.biz)
        self.bizIdToReview[review.bizId] = review

    def reviewFromBizId(self, Id):
        if Id in self.reviewedBizIds:
            return self.bizIdToReview[Id]
        else:
            return None

    def getReviewedBizs(self):
        return self.reviewedBizs

    def getReviewedBizIds(self):
        return self.reviewedBizIds

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

    def combineVectorizedReviews(self):
        if self.reviews:
            review0 = self.reviews[0]
            self.vectorizedText = csr_matrix(np.zeros(review0.getVectorizedText().shape))
            for review in self.reviews:
                if review.vectorizedText != None:
                    self.vectorizedText = self.vectorizedText + review.getVectorizedText()

    def getVectorizedText(self):
        return self.vectorizedText

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
        self.yelpStars = bizJson['stars']
        self.extractedAverageStars = None
        self.yelpReviewCount = bizJson['review_count']
        self.categories = set(bizJson['categories'])
        self.open = bizJson['open']
        self.reviews = []
        self.vectorizedText = None
        self.reviewerIds = set()

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
        self.reviewerIds.add(review.userId)

    def combineVectorizedReviews(self):
        if self.reviews:
            review0 = self.reviews[0]
            self.vectorizedText = csr_matrix(np.zeros(review0.getVectorizedText().shape))
            for review in self.reviews:
                if review.vectorizedText != None:
                    self.vectorizedText = self.vectorizedText + review.getVectorizedText()

    def getVectorizedText(self):
        return self.vectorizedText

    def __str__(self):
        return 'id: {}\nname: {}\nneighborhoods: {}\ncity: {}\n'.format(self.id, self.name, self.neighborhoods, self.city) + \
               ' state: {}\nlat: {}\nlon: {}stars: {}\n'.format(self.state, self.lat, self.lon, self.extractedAverageStars) + \
            'reviewCount: {}\ncategories: {}\nopen: {}\nreviews: {}'.format(self.getExtractedReviewCount(), self.categories, self.open, self.reviews)


random.seed(42)


class Recommender:

    NSIMILAR_USERS = 50

    def __init__(self, users, bizs, reviews, k=10):
        self.users = users
        self.bizs = bizs
        self.reviews = reviews
        self.finalK = k
        self.initialK = 10*k
        self.simTheta = None

    def getUsers(self):
        return self.users

    def getBizs(self):
        return self.bizs

    def getReviews(self):
        return self.reviews

    def train(self):
        review0 = self.reviews[0]
        self.simTheta = csr_matrix(np.zeros(review0.getVectorizedText().shape))


    def recommend(self, queryUsers, constraints=None):
        recommendations = {}
        for user in queryUsers:
            recs = self.topKRecommendations(user)
            if user in constraints and constraints[user] != None:
                recs = csp.reduceBizs(recs, constraints[user])
            recommendations[user.id] = recs
        return recommendations

    def topKRecommendations(self, user):
        # returns initialK recommendations (list of (score, Biz) tuples)
        collabFilteringList = self.userUserFilter(user)
        # narrows down to finalK recommendations
        # collabFilteringList = collabf.similarityFilter(user, collabFilteringList, self)
        return collabFilteringList

    # returns list of tuples (predicted rating of user for a biz, that biz)
    def userUserFilter(self, user):
        #similarUsers is list of tuples of (user-user similarity score, user)
        similarUsers = self.rankedSimilarUsers(user)
        #candidateBizs is list uf tuples of (user-user sim* user's stars, biz)
        candidateBizs = self.findCandidateBizs(user, similarUsers)
        rankedRecommendations = self.recommendFromCandidates(user, candidateBizs, similarUsers)
        return rankedRecommendations

    # performs prediction of the queryUser's stars for some candidate bizs. Current formula from
    # page 13 of http://files.grouplens.org/papers/FnT%20CF%20Recsys%20Survey.pdf
    def recommendFromCandidates(self, user, candidateBizs, similarUsers):
        recommendations = []
        userAvgStars = user.findAverageStars()
        for _, biz in candidateBizs:
            simSum = 0
            p_ui = 0
            nOthersWithReview = 0
            for simScore, other in similarUsers:
                reviewOther = other.bizIdToReview[biz.id]
                if reviewOther is not None:
                    p_ui += simScore*(reviewOther.getStars() - other.findAverageStars())
                    simSum += simScore
                    nOthersWithReview += 1
            if nOthersWithReview >= 3:
                p_ui /= simSum
                p_ui += userAvgStars
                recommendations.append((p_ui, biz))
        recommendations.sort(reverse=True)
        return recommendations

     # returns list of tuples of (user-user similarity score, user)
    def rankedSimilarUsers(self, user):
        similarUsers = []
        for other in self.users:
            if user is other:
                continue
            sim = cosine_similarity(user.getVectorizedText(), other.getVectorizedText())
            similarUsers.append((sim, other))
        similarUsers.sort(reverse=True)
        limit = min(self.NSIMILAR_USERS, len(similarUsers))
        return similarUsers[:limit]

    # scans similar user's rated businesses and assembles a list of candidates
    # returns list of tuples (similarity score * stars that a similar reviewer gave for a biz, that biz)
    def findCandidateBizs(self, user, similarUsers):
        userBizIds = user.getReviewedBizIds()
        candidateBizIds = set()
        candidateBizs = []
        for simScore, other in similarUsers:
            otherBizs = other.getReviewedBizs()
            for biz in otherBizs:
                review = other.reviewFromBizId(biz.id)
                # if it's not already a candidate and the query user does not know it and the other gave it a good score
                if biz.id not in candidateBizIds and biz.id not in userBizIds and review.getStars() > other.findAverageStars():
                    candidateBizIds.add(biz.id)
                    candidateBizs.append((simScore*review.getStars(), biz))
        limit = min(len(candidateBizs), self.initialK)
        return candidateBizs[:limit]


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
    # for bizId in bizIdToReviews.keys():

    return reviews, bizIdToReviews

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