import pickle, json
from util import User, Biz, Review, Recommender
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


userRead = open('../yelp_academic_dataset_user.json')
minUserReviews = 100
minBizReviews = 100
maxUsers = 1000
maxBizs = 50
maxReviewsPerBiz = 20
city = 'Pittsburgh'


bizRead = open('../yelp_academic_dataset_business.json')
bizs = []
bizIdToBiz = {}
n = 0
for line in bizRead:
    jsonBiz = json.loads(line)
    if 'Restaurants' in jsonBiz['categories'] and jsonBiz['review_count'] >= minBizReviews and (city == None or city == jsonBiz['city']):
        biz = Biz(jsonBiz)
        bizs.append(biz)
        bizIdToBiz[biz.id] = biz
        n += 1
    if n >= maxBizs:
        break
bizRead.close()


#get all the reviews for the extracted businesses
reviewsRead = open('../yelp_academic_dataset_review.json')
reviews = []
# bizIdToReviews = dict.fromkeys(all, "")
userIdsWithReviews = set()
bizIds = bizIdToBiz.keys()
bizToReviewCount = dict.fromkeys(bizIds, 0)
reviewIds = set()
for line in reviewsRead:
    jsonReview = json.loads(line)
    if jsonReview['business_id'] in bizIds:
        if bizToReviewCount[jsonReview['business_id']] <= maxReviewsPerBiz:
            jsonReview["text"] = jsonReview["text"].lower()
            review = Review(jsonReview)
            reviews.append(review)
            userIdsWithReviews.add(review.userId)
            reviewIds.add(review.id)
            bizToReviewCount[review.bizId] += 1
reviewsRead.close()

#get all the users that wrote reviews for the businesses above
users = []
userIdToUser = {}
n = 0
for line in userRead:
    jsonUser = json.loads(line)
    if jsonUser['review_count'] >= minUserReviews and jsonUser['user_id'] in userIdsWithReviews:
        user = User(jsonUser)
        users.append(user)
        userIdToUser[user.id] = user
        n += 1
        if n >= maxUsers:
            break

# # get all the other reviews for the users in our set that are NOT for the restaurants we have
userIds = userIdToUser.keys()
# reviewsRead = open('../yelp_academic_dataset_review.json')
# for line in reviewsRead:
#     jsonReview = json.loads(line)
#     if jsonReview['user_id'] in userIds and (jsonReview['user_id'] + jsonReview['business_id']) not in reviewIds:
#         jsonReview["text"] = jsonReview["text"].lower()
#         review = Review(jsonReview)
#         reviews.append(review)
# reviewsRead.close()



reviewIdToIndex = {}
reviewIds = []
reviewCorpus = []
for i in xrange(len(reviews)):
    reviewIds.append(reviews[i].getId())
    reviewIdToIndex[reviews[i].getId()] = i
    reviewCorpus.append(reviews[i].getText())
    reviews[i].setText(None)

# combinedTexts = np.array(bizIdToReviewText.values())
data = np.mat([np.transpose(reviewIds), np.transpose(reviewCorpus)])

#data[0, :] is the array of all biz id's
#data[1, :] is the array of all the reviews

vectorizedReviewTexts = TfidfVectorizer().fit_transform(reviewCorpus)
for i in xrange(len(reviews)):
    reviews[i].setVectorizedText(vectorizedReviewTexts[i])

#make all the reviews for businesses and users accessible to each other
for review in reviews:
    if review.bizId in bizIds:
        biz = bizIdToBiz[review.bizId]
        biz.addReview(review)
    if review.userId in userIds:
        user = userIdToUser[review.userId]
        user.addReview(review)


#filter for users with a minimum number of reviews in our list of reviews (different from their reviewCount field)
usersWithManyReviews = []
for user in users:
    if len(user.reviews) >= 20:
        usersWithManyReviews.append(user)

for user in usersWithManyReviews:
    user.combineVectorizedReviews()
for biz in bizs:
    biz.combineVectorizedReviews()

recommender = Recommender(usersWithManyReviews, bizs, reviews)
print len(users)
print len(usersWithManyReviews)
print len(bizs)
print len(reviews)
pickle.dump(recommender, open('pickledRecommender', 'wb'))

# pickle.dump(bizs, open('business_list', 'wb'))
# pickle.dump(usersWithManyReviews, open('user_list', 'wb'))
# pickle.dump(reviews, open('review_list', 'wb'))
# pickle.dump(bizIdToReviews, open('biz_id_to_review', 'wb'))
# pickle.dump(bizIdToText, open('biz_id_to_review_text', 'wb'))