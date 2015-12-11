import json, string, re
from util import Review
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from nltk.corpus import stopwords


city = 'Pittsburgh'

# get the maxBizs number of businesses in a city with the highest review count
bizRead = open('../yelp_academic_dataset_business.json')
bizIds = set()
nBizs = 0
for line in bizRead:
    jsonBiz = json.loads(line)
    if 'Restaurants' in jsonBiz['categories'] and (city == None or city == jsonBiz['city']):
        bizIds.add(jsonBiz['business_id'])
        nBizs += 1
bizRead.close()
print "Read {} restaurants from {}".format(nBizs, city)



reviewsRead = open('../yelp_academic_dataset_review.json')
reviews = []
# bizIdToReviewCount = dict.fromkeys(bizIds, 0)
nReviews = 0
for line in reviewsRead:
    jsonReview = json.loads(line)
    if jsonReview['business_id'] in bizIds:
        jsonReview = json.loads(line)
        reviews.append(Review(jsonReview))
        nReviews += 1
    # bizToReviewCount[review.bizId] += 1
reviewsRead.close()
print "Read {} reviews from {}".format(nReviews, city)


remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
remove_punctuation_map.pop(ord('\''), None)
stopwordsEng = stopwords.words('english')
removeStopWordsMap = dict((stopWord, None) for stopWord in stopwordsEng)

for review in reviews:
    text = review.getText()
    text = text.translate(remove_punctuation_map).lower()
    tokens = re.findall(r"[\w\u0027]+", text)
    tokens = [word for word in tokens if word not in stopwordsEng]
    review.setText(tokens)

reviewCorpus = []
n = 0
for review in reviews:
    # user = review.getUser()
    # biz = review.getBiz()
    # sentence = LabeledSentence(words=review.getText(), tags=[user.getId(), biz.getId()])
    sentence = LabeledSentence(words=review.getText(), tags=['SENT_%s' % n])
    reviewCorpus.append(sentence)
    n += 1

print 'Finished preprocessing'

model = Doc2Vec(reviewCorpus)
model.save('doc2VecModel')
