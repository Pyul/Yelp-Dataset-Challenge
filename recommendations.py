import json, codecs

# bizf = open('/Users/sigberto/Documents/yelpCSProject/yelp_academic_dataset_business.json')
# popBizs = {}
# popBiz = None
# for line in bizf:
#     biz = json.loads(line)
#     if biz["review_count"] > 50 and "Food" in biz["categories"] or "Restaurant" in biz["categories"]:
#         popBiz = biz
#         break
# bizf.close()
#
# print popBiz

# Output all reviews from our prolific user
# outputf = open('sampleReviews', 'w')
# reviewsf = open('/Users/sigberto/Documents/yelpCSProject/yelp_academic_dataset_review.json')
#
# userIds = set()
# reviews = []
# # bizIds = popBizs.keys()
# for line in reviewsf:
#     review = json.loads(line)
#     # reviews.append(review)
#     if review["user_id"] == 'nEYPahVwXGD2Pjvgkm7QqQ':
#         outputf.write(review['text'].lower().encode("UTF-8"))
#         # userIds.add(review["user_id"])
# reviewsf.close()
# outputf.close()

#Output all reviews about our favorite place
outputf = open('bizReviews', 'w')
reviewsf = open('/Users/sigberto/Documents/yelpCSProject/yelp_academic_dataset_review.json')

for line in reviewsf:
    review = json.loads(line)
    # reviews.append(review)
    if review["business_id"] == 'McikHxxEqZ2X0joaRNKlaw':
        outputf.write(review['text'].lower().encode("UTF-8"))
        # userIds.add(review["user_id"])
reviewsf.close()
outputf.close()

# userToCount = {}
# for review in reviews:
#     if review['user_id'] == 'nEYPahVwXGD2Pjvgkm7QqQ':
#         print review
    # userId = review['user_id']
    # if userId in userIds:
    #     if userId in userToCount.keys():
    #         userToCount[userId] = userToCount[userId] + 1
    #     else:
    #         userToCount[userId] = 0

# activeUsers = set()
# for key in userToCount.keys():
#     if userToCount[key]>10:
#         activeUsers.add(key)
#
# print activeUsers


#set([u'nEYPahVwXGD2Pjvgkm7QqQ', u'bSkco4ZdB7REFtDsJUVrDA', u't5mr9snU8tI7hcjMSCYxLQ', u'IbvOxKSps_K5wa3a2_jc-Q', u'9cCTmiJ7hz35rHIdr8n9kA', u'scIDar9VGDMcTOHbem37pg', u'FyCBkNXwoI_6X6apbslg4g', u'i0aToSFRKd6PjF3dMY8JLQ', u'u7rJ_CFbp4IYeT39fNfVDQ', u'VSfPN4tGH4Tjd3JUu8yUew', u'wtQsINapBJLhtfIe2xecpw', u'2rSeth60_CuWN3ZJ4k41lg', u'MSuyK2p8G9hEqyWf5IgnYQ', u'fwsJGulnozT2U6FefsLiFw', u'ytgLwKzD6B4af5vW56RpJg', u'Yvn7DHj7o7sSCH2rNH74Mw', u'kRa-sZPizt2EqWYVFZ6DGw', u'2VqU7uoDeiE8OUL14Ce5Ig', u'G7KQF0U0p25nf07dObPeVg', u'WMTm9HHRA3EewoxTX1Gleg', u'B5WkNWDxZ-baWoQc6DBNHA', u'8fApIAMHn2MZJFUiCQto5Q', u'DLu1Bum8EXNE62xno-v0VA', u'SlKJNLm1SQdgaaKtqD--1w', u'bvu13GyOUwhEjPum2xjiqQ', u'fFoRIzbwpMiv3BvfdtdlXQ', u'LhgQq1x4n9ardg1PFo8vgg', u'jaDTgMqh4DzoLJrMrW3JcA', u'sS2sSWqZMnQMRoWx-Mdzgg', u'op2Gve4sAMQ4qEzq2Tad0g', u'N_cH3QA_eXXmnzz-nzEeKA', u'QmykNasFdjyaQ3K8oME3fQ', u'W9ZG5q-QIblXYeHgaPS5Uw', u'9Ify25DK87s5_u2EhK0_Rg', u'p3MF38R3Htt3ZWyPJMTFgQ', u'5UCdxI_krbnv7_7Q3Ur_eQ', u'_e5drmTJBSV0yiFBrTimtg', u'JPPhyFE-UE453zA6K0TVgw'])
