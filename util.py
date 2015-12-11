import json, collections, numpy as np, random, csp, copy, regressor, stat
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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
        self.starStdDev = None
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

    def getReviewFromBizId(self, Id):
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
            if self.reviews:
                self.computeStats()
                return self.extractedAverageStars
            else:
                return None

    def findStarStdDev(self):
        if self.starStdDev is not None:
            return self.starStdDev
        else:
            if self.reviews:
                self.computeStats()
                return self.starStdDev
            else:
                return None

    def computeStats(self):
        if self.reviews:
            starsList = []
            for review in self.reviews:
                starsList.append(review.stars)
            self.extractedAverageStars = np.mean(starsList)
            self.starStdDev = np.std(starsList)
            return True
        else:
            return False


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
        self.starStdDev = None
        self.yelpReviewCount = bizJson['review_count']
        self.categories = set(bizJson['categories'])
        self.featurizedCategories = None
        self.open = bizJson['open']
        self.reviews = []
        self.vectorizedText = None
        self.attributes = bizJson['attributes']
        self.featurizedAttributes = None
        self.reviewerIds = set()

    def setId(self, Id):
        self.id = Id

    def setFeaturizedAttributes(self, phi):
        self.featurizedAttributes = phi

    def getFeaturizedAttributes(self):
        return self.featurizedAttributes

    def setFeaturizedCategories(self, phi):
        self.featurizedCategories = phi

    def getFeaturizedCategories(self):
        return self.featurizedCategories

    def getAttributes(self):
        return self.attributes

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
            if self.reviews:
                self.computeStats()
                return self.extractedAverageStars
            else:
                return None

    def findStarStdDev(self):
        if self.starStdDev is not None:
            return self.starStdDev
        else:
            if self.reviews:
                self.computeStats()
                return self.starStdDev
            else:
                return None

    def computeStats(self):
        if self.reviews:
            starsList = []
            for review in self.reviews:
                starsList.append(review.stars)
            self.extractedAverageStars = np.mean(starsList)
            self.starStdDev = np.std(starsList)
            return True
        else:
            return False

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
    KFOLDS = 5
    MIN_USER_USER_COS_SIM = 0.7

    def __init__(self, users, bizs, reviews, k=10):
        self.users = users
        self.bizs = bizs
        self.reviews = reviews
        self.finalK = k
        self.initialK = 10*k
        self.simTheta = None
        self.minSim = 0.5
        self.UIPairs = []
        self.reviewStars = []
        for user in users:
            for biz in user.getReviewedBizs():
                review = user.getReviewFromBizId(biz.getId())
                self.reviewStars.append(review.getStars())
                self.UIPairs.append((user, biz))

    def getUsers(self):
        return self.users

    def getBizs(self):
        return self.bizs

    def getReviews(self):
        return self.reviews

    def evalRecommendations(self):
        users = set(copy.deepcopy(self.users))
        samples = []
        nUsers = len(users)/self.KFOLDS
        for i in xrange(self.KFOLDS-1):
            sample = random.sample(users, nUsers)
            samples.append(sample)
            for item in sample:
                users.remove(item)
        samples.append(users)
        sumSquaredError = 0
        nErrorsConsidered = 0
        for i in xrange(self.KFOLDS):
            testUsers = samples[i]
            trainUsers = samples[:i].extend(samples[i+1:])
            for testUser in testUsers:
                singleUserError = self.evalUser(testUser, trainUsers)
                if singleUserError is not None:
                    sumSquaredError += singleUserError
                    nErrorsConsidered += 1
        avgSquarredError = sumSquaredError / nErrorsConsidered
        return avgSquarredError

    def evalUser(self, userGiven, others):
        user = copy.deepcopy(userGiven)
        bizs = user.getReviewedBizs()
        toPredict = random.choice(bizs)
        bizs.remove(toPredict)
        similarUsers = self.rankedSimilarUsers(user, others)
        recommendations = self.recommendFromCandidates(user, [toPredict], similarUsers)
        if recommendations:
            p_ui = recommendations[0][0]
            reviewForPredicted = user.bizIdToReview[toPredict.getId()]
            r_ui = reviewForPredicted.getStars()
            squaredError = (p_ui - r_ui)**2
            return squaredError
        else:
            return None

    def regress(self):
        np.random.seed(17411)
        # load traning set
        # first column = y
        # second to end = x
        Xtrain, Ytrain = regressor.preprocess(self.UIPairs, self.reviewStars, self.users, self.bizs)

        # start training
        regr,trainscore,testscore = regressor.train(Xtrain, Ytrain, model='Linear Regression')
        print("LR // Training error = "+str(trainscore)+' // Test error = ' + str(testscore))

        regr,trainscore,testscore = regressor.train(Xtrain, Ytrain, model='Random Forest', n_estimators=7)
        print("RF // Training error = "+str(trainscore)+' // Test error = ' + str(testscore))

        #regr,trainscore,testscore = regressor.train(Xtrain, Ytrain, model='SVM', kernel='linear')
        #print("SVM LNR // Training error = "+str(trainscore)+' // Test error = ' + str(testscore))
        regr,trainscore,testscore = regressor.train(Xtrain, Ytrain, model='SVM', kernel='poly',degree=3)
        print("SVM POLY deg3 // Training error = "+str(trainscore)+' // Test error = ' + str(testscore))
        #regr,trainscore,testscore = regressor.train(Xtrain, Ytrain, model='SVM', kernel='poly',degree=4)
        #print("SVM POLY deg4 // Training error = "+str(trainscore)+' // Test error = ' + str(testscore))
        #regr,trainscore,testscore = regressor.train(Xtrain, Ytrain, model='SVM', kernel='poly',degree=5)
        #print("SVM POLY deg5 // Training error = "+str(trainscore)+' // Test error = ' + str(testscore))

        regr,trainscore,testscore = regressor.train(Xtrain, Ytrain, model='NN')
        print("Neural Nets // Training error = "+str(trainscore)+' // Test error = ' + str(testscore))

        # # learning curve for different n_estimators
        # trainscores=[]
        # testscores=[]
        # treenum=range(1,16,1)
        # for n_estimator in treenum:
        #     trainscore = 0.
        #     testscore = 0.
        #     niter = 10
        #     for i in range(niter):
        #         regr,score1,score2 = \
        #             regressor.train(Xtrain, Ytrain, model='Random Forest', n_estimators=n_estimator)
        #         trainscore += score1
        #         testscore += score2
        #     print("No. of trees = "+str(n_estimator)+\
        #               " -> Training error = "+str(trainscore/float(niter))+\
        #               ' // Test error = ' + str(testscore/float(niter)))
        #     trainscores.append(trainscore/float(niter))
        #     testscores.append(testscore/float(niter))
        # plt.plot(treenum,trainscores,'g^',label='Training Error')
        # plt.plot(treenum,testscores,'bs',label='Test Error')
        # plt.xlabel('No. of trees')
        # plt.ylabel('Root-mean-squared error')
        # plt.legend()
        # plt.show()
        #
        # # learning curve for different no. of units for neural nets
        #trainscores=[]
        #testscores=[]
        #unitnum=range(1,100,10)
        #for n_units in unitnum:
        #     trainscore = 0.
        #     testscore = 0.
        #     niter = 10
        #     for i in range(niter):
        #         regr,score1,score2 = \
        #             regressor.train(Xtrain, Ytrain, model='NN', n_units=n_units)
        #         trainscore += score1
        #         testscore += score2
        #     print("No. of units = "+str(n_units)+\
        #               " -> Training error = "+str(trainscore/float(niter))+\
        #               ' // Test error = ' + str(testscore/float(niter)))
        #     trainscores.append(trainscore/float(niter))
        #     testscores.append(testscore/float(niter))
        #plt.plot(unitnum,trainscores,'g^-',label='Training Error')
        #plt.plot(unitnum,testscores,'bs-',label='Test Error')
        #plt.xlabel('No. of units')
        #plt.ylabel('Root-mean-squared error')
        #plt.legend()
        #plt.show()

        # # learning curve for different number of folds
        # trainscores=[]
        # testscores=[]
        # kfold=np.linspace(0.1, 0.9, num=9)
        # for train_size in kfold:
        #     trainscore = 0.
        #     testscore = 0.
        #     niter = 10
        #     for i in range(niter):
        #         regr,score1,score2 = \
        #             regressor.train(Xtrain, Ytrain, model='Random Forest', train_size=train_size, n_estimators = 5)
        #         trainscore += score1
        #         testscore += score2
        #     print("No. of k-fold = "+str(train_size)+\
        #               " -> Training error = "+str(trainscore/float(niter))+' // Test error = ' + str(testscore/float(niter)))
        #     trainscores.append(trainscore/float(niter))
        #     testscores.append(testscore/float(niter))
        # plt.plot(kfold,trainscores,'g^',label='Training Error')
        # plt.plot(kfold,testscores,'bs',label='Test Error')
        # plt.xlabel('K-fold Cross Validation')
        # plt.legend()
        # plt.show()

        # load test set
        # Xtest = np.random.rand(5, 3)

        # make prediction
        # y_pred = regr.predict(Xtest)
        print(" - Finished.")

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
        candidateBizs = [candidateBizs[1] for biz in candidateBizs]
        rankedRecommendations = self.recommendFromCandidates(user, candidateBizs, similarUsers)
        return rankedRecommendations

    # performs prediction of the queryUser's stars for some candidate bizs. Current formula from
    # page 13 of http://files.grouplens.org/papers/FnT%20CF%20Recsys%20Survey.pdf
    def recommendFromCandidates(self, user, candidateBizs, similarUsers):
        recommendations = []
        userAvgStars = user.findAverageStars()
        for biz in candidateBizs:
            simSum = 0
            p_ui = 0
            nOthersWithReview = 0
            for simScore, other in similarUsers:
                otherReviewedBizIds = other.bizIdToReview.keys()
                if biz.id in otherReviewedBizIds:
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
    def rankedSimilarUsers(self, user, others=None):
        if others is None:
            others = self.users
        similarUsers = []
        for other in others:
            if user is other:
                continue
            sim = cosine_similarity(user.getVectorizedText(), other.getVectorizedText())
            if sim > self.minSim:
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
                    candidateBizs.append((simScore*(review.getStars() - other.findAverageStars()), biz))
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
