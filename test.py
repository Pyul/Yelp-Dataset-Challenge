import sys
import pickle, string, nltk, re
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from nltk.corpus import stopwords
#
#
# # reviewCorpus = pickle.load(open('reviewCorpus'))
# # reviewCorpus = reviewCorpus[:10]
#
# reviewCorpus = ['TO! have not. I\'ll do it', "*haha. nyet"]
# reviewCorpus = [unicode(s, "utf-8") for s in reviewCorpus]
#
# processedCorpus = []
# remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
# remove_punctuation_map.pop(ord('\''), None)
# stopwordsEng = stopwords.words('english')
# removeStopWordsMap = dict((stopWord, None) for stopWord in stopwordsEng)
# for doc in reviewCorpus:
#     newDoc = doc.translate(remove_punctuation_map)
#     newDoc = newDoc.lower()
#     tokens = re.findall(r"[\w\u0027]+", newDoc)
#     tokens = [word for word in tokens if word not in stopwords.words('english')]
#     newDoc = ' '.join(tokens)
#     processedCorpus.append(newDoc)
# reviewCorpus = processedCorpus
#
# for doc in processedCorpus:
#     print doc

# rawCorpus = pickle.load(open('reviewCorpus'))
# rec = pickle.load(open('pickledRecommender'))
# users = rec.users
# reviews = rec.reviews
# bizs = rec.bizs
# rawCorpus = rawCorpus[:10]
#
# processedCorpus = []
# remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
# remove_punctuation_map.pop(ord('\''), None)
# stopwordsEng = stopwords.words('english')
# removeStopWordsMap = dict((stopWord, None) for stopWord in stopwordsEng)
#
# for review in reviews:
#     text = review.getText()
#     text = text.translate(remove_punctuation_map).lower()
#     tokens = re.findall(r"[\w\u0027]+", text)
#     tokens = [word for word in tokens if word not in stopwords.words('english')]
#     review.setText(tokens)
#
# # for iterating over raw corpus list
# # for doc in rawCorpus:
# #     newDoc = doc.translate(remove_punctuation_map)
# #     newDoc = newDoc.lower()
# #     tokens = re.findall(r"[\w\u0027]+", newDoc)
# #     tokens = [word for word in tokens if word not in stopwords.words('english')]
# #     newDoc = ' '.join(tokens)
# #     processedCorpus.append(newDoc)
#
# reviewCorpus = []
# n = 0
# for review in reviews:
#     # user = review.getUser()
#     # biz = review.getBiz()
#     # sentence = LabeledSentence(words=review.getText(), tags=[user.getId(), biz.getId()])
#     sentence = LabeledSentence(words=review.getText(), tags=['SENT_%s' % n])
#     reviewCorpus.append(sentence)
#     n += 1
#
# for doc in reviewCorpus:
#     print doc
# model = Doc2Vec(reviewCorpus)
# # vectorizedReviewTexts = TfidfVectorizer().fit_transform(reviewCorpus)
#
#
#
# for i in xrange(len(reviews)):
#     text = reviews[i].getText()
#     print text
#     vec = model.infer_vector(text)
#     print vec
#     reviews[i].setVectorizedText(vec)
#     # reviews[i].setVectorizedText(vectorizedReviewTexts[i])
#     reviews[i].setText(None)

import json, string, re
from util import Review
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from nltk.corpus import stopwords
from util import Recommender, Review

# reviewsRead = open('../yelp_academic_dataset_review.json')
# reviews = []
# # bizIdToReviewCount = dict.fromkeys(bizIds, 0)
# n = 0
# for line in reviewsRead:
#     jsonReview = json.loads(line)
#     jsonReview["text"] = jsonReview["text"].lower()
#     reviews.append(Review(jsonReview))
#     # bizToReviewCount[review.bizId] += 1
#     n+=1
#     if n == 10000:
#         break
# reviewsRead.close()
#
# print 'Done reading from JSON'
#
# remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
# remove_punctuation_map.pop(ord('\''), None)
# stopwordsEng = stopwords.words('english')
# removeStopWordsMap = dict((stopWord, None) for stopWord in stopwordsEng)
#
# for review in reviews:
#     text = review.getText()
#     text = text.translate(remove_punctuation_map).lower()
#     tokens = re.findall(r"[\w\u0027]+", text)
#     tokens = [word for word in tokens if word not in stopwords.words('english')]
#     review.setText(tokens)
#
# reviewCorpus = []
# n = 0
# for review in reviews:
#     # user = review.getUser()
#     # biz = review.getBiz()
#     # sentence = LabeledSentence(words=review.getText(), tags=[user.getId(), biz.getId()])
#     sentence = LabeledSentence(words=review.getText(), tags=['SENT_%s' % n])
#     reviewCorpus.append(sentence)
#     n += 1
#
# print 'Done preprocessing for Doc2Vec'
#
# model = Doc2Vec(reviewCorpus)
# model.save('doc2VecModelTest')
rec = pickle.load(open('pickledRecommender'))
reviews = rec.reviews
users = rec.users
bizs = rec.bizs

model = Doc2Vec.load('doc2VecModel')

for review in reviews:


