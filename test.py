import pickle, json

userRead = open('yelp_academic_dataset_user.json')
minUserReviews = 100
minBizReviews = 10
maxUsers = 100
maxBizs = 10
maxReviews = 10
city = 'Pittsburgh'


bizRead = open('yelp_academic_dataset_business.json')
bizs = []
bizIdToBiz = {}
n = 0
for line in bizRead:
    biz = json.loads(line)
    if 'Restaurants' in biz['categories'] and biz['review_count'] >= minBizReviews and (city == None or city == biz['city']):
        bizs.append(biz)
        bizIdToBiz[biz['business_id']] = biz
        n += 1
        if n == maxBizs:
            break



users = []
userIdToUser = {}
n = 0
for line in userRead:
    user = json.loads(line)
    if user['review_count'] >= minUserReviews:
        users.append(user)
        userIdToUser[user['user_id']] = user
    n += 1
    if n == maxUsers:
        break

reviewsRead = open('yelp_academic_dataset_review.json')
reviews = []
# bizIdToReviews = dict.fromkeys(all, "")
n = 0
for line in reviewsRead:
    review = json.loads(line)
    if review['business_id'] in bizIdToBiz.keys() and review['user_id'] in userIdToUser.keys():
        review["text"] = review["text"].lower()
        reviews.append(review)
        biz = bizIdToBiz[review['business_id']]
        user = userIdToUser[review['user_id']]

        #add review to business review list
        if 'reviews' in biz.keys():
            bizReviews = biz['reviews']
            bizReviews.append(review)
        else:
            biz['reviews'] = [review]

        #add review to user review list
        if 'reviews' in user.keys():
            userReviews = user['reviews']
            userReviews.append(review)
        else:
            user['reviews'] = [review]
        n += 1
        if n == maxReviews:
            break

# bizIdToText = {}
# for bizId in bizIdToReviews.keys():
#     text = ''
#     for review in bizIdToReviews[bizId]:
#         text += '\n' + review['text']


pickle.dump(bizs, open('business_list', 'wb'))
pickle.dump(users, open('user_list', 'wb'))
pickle.dump(reviews, open('review_list', 'wb'))
# pickle.dump(bizIdToReviews, open('biz_id_to_review', 'wb'))
# pickle.dump(bizIdToText, open('biz_id_to_review_text', 'wb'))