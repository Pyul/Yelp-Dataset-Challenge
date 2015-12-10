import pickle, string, nltk, re
from nltk.corpus import stopwords


# reviewCorpus = pickle.load(open('reviewCorpus'))
# reviewCorpus = reviewCorpus[:10]

reviewCorpus = ['TO! have not. I\'ll do it', "*haha. nyet"]
reviewCorpus = [unicode(s, "utf-8") for s in reviewCorpus]

processedCorpus = []
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
remove_punctuation_map.pop(ord('\''), None)
stopwordsEng = stopwords.words('english')
removeStopWordsMap = dict((stopWord, None) for stopWord in stopwordsEng)
for doc in reviewCorpus:
    newDoc = doc.translate(remove_punctuation_map)
    newDoc = newDoc.lower()
    tokens = re.findall(r"[\w\u0027]+", newDoc)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    newDoc = ' '.join(tokens)
    processedCorpus.append(newDoc)
reviewCorpus = processedCorpus

for doc in processedCorpus:
    print doc
