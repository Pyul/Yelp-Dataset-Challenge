import pickle, json

userRead = open('../yelp_academic_dataset_user.json')
minUserReviews = 100
minBizReviews = 10
maxUsers = 100
maxBizs = 10
maxReviews = 10
city = 'Pittsburgh'


bizRead = open('../yelp_academic_dataset_business.json')
bizs = []
bizIdToBiz = {}
n = 0
for line in bizRead:
    biz = json.loads(line)
    if 'Restaurants' in biz['categories'] and biz['review_count'] >= minBizReviews and (city == None or city == biz['city']):
        biz['reviews'] = []
        bizs.append(biz)
        bizIdToBiz[biz['business_id']] = biz
        n += 1
    if n >= maxBizs:
        break
bizRead.close()


#get all the reviews for the businesses
reviewsRead = open('../yelp_academic_dataset_review.json')
reviews = []
# bizIdToReviews = dict.fromkeys(all, "")
userIdsWithReviews = set()
bizIds = bizIdToBiz.keys()
bizToReviewCount = dict.fromkeys(bizIds, 0)
reviewIds = set()
for line in reviewsRead:
    review = json.loads(line)
    if review['business_id'] in bizIds:
        if bizIdToBiz[review['business_id']]['review_count'] > minBizReviews:
            if bizToReviewCount[review['business_id']] <= 10:
                review["text"] = review["text"].lower()
                reviews.append(review)
                userIdsWithReviews.add(review['user_id'])
                reviewIds.add(review['user_id'] + review['business_id'])
                bizToReviewCount[review['business_id']] += 1
reviewsRead.close()

#get all the users that wrote reviews for the businesses above
users = []
userIdToUser = {}
n = 0
for line in userRead:
    user = json.loads(line)
    if user['review_count'] >= minUserReviews and user['user_id'] in userIdsWithReviews:
        user['reviews'] = []
        users.append(user)
        userIdToUser[user['user_id']] = user
        n += 1
    if n >= maxUsers:
        break

# get all the other reviews for the users in our set that are NOT for the restaurants we have
userIds = userIdToUser.keys()
reviewsRead = open('../yelp_academic_dataset_review.json')
for line in reviewsRead:
    review = json.loads(line)
    if review['user_id'] in userIds and (review['user_id'] + review['business_id']) not in reviewIds:
        review["text"] = review["text"].lower()
        reviews.append(review)
reviewsRead.close()

#make all the reviews for businesses and users accessible to each
for review in reviews:
    review['review_id'] = review['user_id'] + review['business_id']
    if review['business_id'] in bizIds:
        biz = bizIdToBiz[review['business_id']]
        biz['reviews'].append(review)
    if review['user_id'] in userIds:
        user = userIdToUser[review['user_id']]
        user['reviews'].append(review)

#filter for users with a minimum number of reviews in our list of reviews (different from their review_count field)
usersWithManyReviews = []
for user in users:
    if len(user['reviews']) > 50:
        usersWithManyReviews.append(user)
# bizIdToText = {}
# for bizId in bizIdToReviews.keys():
#     text = ''
#     for review in bizIdToReviews[bizId]:
#         text += '\n' + review['text']


pickle.dump(bizs, open('business_list', 'wb'))
pickle.dump(usersWithManyReviews, open('user_list', 'wb'))
pickle.dump(reviews, open('review_list', 'wb'))
# pickle.dump(bizIdToReviews, open('biz_id_to_review', 'wb'))
# pickle.dump(bizIdToText, open('biz_id_to_review_text', 'wb'))