import wordsegUtil, sys, string

CORPUS1 = 'sampleUserReviews'
CORPUS2 = 'sampleBizReviews'

def getRealCosts(CORPUS):
    sys.stdout.write('Training language cost functions [corpus: %s]... ' % CORPUS)
    sys.stdout.flush()

    _realUnigramCost, _realBigramCost = wordsegUtil.makeLanguageModels(CORPUS)
    _possibleFills = wordsegUtil.makeInverseRemovalDictionary(CORPUS, 'aeiou')

    print 'Done!'
    print ''
    return _realUnigramCost, _realBigramCost, _possibleFills

userBigramCost, _, _ = getRealCosts(CORPUS1)
bizBigramCost, _, _ = getRealCosts(CORPUS2)

def processString(line):
    line = line.lower()
    line = ' '.join(word.strip(string.punctuation) for word in line.split() if word.strip(string.punctuation))
    return line

def findCost(line):
    line = line.split()
    lenLine = len(line)
    totalCost = 0.0
    for i in xrange(lenLine):
        totalCost += userBigramCost(line[i]) #+ bizBigramCost(line[i])
    return totalCost / (lenLine)


baseline = "I am a FAN of rock bottom. You'll find some serious southern hospitality here! The cajun tacos were much better but executed oddly. It just drags me down! So if you have decided to take a seat at Rock Bottom, take comfort in knowing that even the staff seems to know your pain. I love me a glazed donut, and was happy to see it wasn't so massive that you couldn't finish."
actualReview = "I must admit, I do like the brews.  I also like the white cheddar mash potatoes.  Other than that, meh.  The food is pretty forgettable.  Another \"brewery\" that sadly fails to be anything special in the food department.  With the restaurants upping the ante on 8th Ave, you may want to kick it up a notch Rock Bottom.  But how would anyone know to?  It's a chain.  Whomp whomp!"
baseline = processString(baseline)
actualReview = processString(actualReview)
print findCost(baseline)
print findCost(actualReview)