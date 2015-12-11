import collections

import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.cross_validation import train_test_split
import sklearn as sklearn
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from math import sqrt

def preprocessUsers(users):
    pass

def preprocessBizs(bizs, dictVectCategoriesInput, dictVectAttributesInput):
    train = True if dictVectCategoriesInput is None else False
    categoryDicts = []
    dictVectCatFinal = None
    for biz in bizs:
        categoryDicts.append(dict.fromkeys(biz.getCategories(), 1))
    if train:
        dictVectCatFinal = DictVectorizer(sparse=False)
        featurizedCategories = dictVectCatFinal.fit_transform(categoryDicts)
    else:
        featurizedCategories = dictVectCategoriesInput.transform(categoryDicts)
        dictVectCatFinal = dictVectCategoriesInput
    for i in xrange(len(bizs)):
        bizs[i].setFeaturizedCategories(featurizedCategories[i])

    attributeDicts = []
    for biz in bizs:
        attr = biz.getAttributes()
        keys = attr.keys()
        for key in keys:
            val = attr[key]
            #correct for nested dicts
            if isinstance(val, dict):
                valkeys = val.keys()
                for valkey in valkeys:
                    attr[key + valkey] = val[valkey]
                attr.pop(key)
        attributeDicts.append(attr)
    dictVectAttrFinal = None
    if train:
        dictVectAttrFinal = DictVectorizer(sparse=False)
        featurizedAttributes = dictVectAttrFinal.fit_transform(attributeDicts)
    else:
        featurizedAttributes = dictVectAttributesInput.transform(attributeDicts)
        dictVectAttrFinal = dictVectAttributesInput
    for i in xrange(len(bizs)):
        bizs[i].setFeaturizedAttributes(featurizedAttributes[i])
    return dictVectCatFinal, dictVectAttrFinal

def preprocessCosineSimilarity(UIPairs):
    similarity = np.zeros(len(UIPairs))
    i = 0
    for user, biz in UIPairs:
        similarity[i] = cosine_similarity(user.getVectorizedText(), biz.getVectorizedText())
        i += 1
    return similarity

def preprocessCategorySimilarity(UIPairs):
    similarity = np.zeros(len(UIPairs))
    index = 0
    bannedCategories = set()
    bannedCategories.add(u'Restaurants')
    bannedCategories.add(u'Food')
    for user, biz in UIPairs:
        bizCategories = set(biz.getCategories())
        userCategoryCounts = collections.Counter()
        nCategories = 0
        for userBiz in user.getReviewedBizs():
            for category in userBiz.getCategories():
                if category not in bannedCategories:
                    userCategoryCounts[category] += 1
                    nCategories += 1
        for cat in userCategoryCounts.keys():
            userCategoryCounts[cat] = 1.0*userCategoryCounts[cat]/nCategories
        score = 0
        userCategories = userCategoryCounts.keys()
        for bizCategory in bizCategories:
            if bizCategory in userCategories:
                score += userCategoryCounts[bizCategory]
        similarity[index] = score
        index += 1
    return similarity

def preprocessJacard(UIPairs):
    jacard = np.zeros(len(UIPairs))
    bannedCategories = set()
    bannedCategories.add(u'Restaurants')
    bannedCategories.add(u'Food')
    i = 0
    for user, biz in UIPairs:
        bizCategories = set(biz.getCategories())
        userCategories = set()
        for userBiz in user.getReviewedBizs():
            for category in userBiz.getCategories():
                userCategories.add(category)
        bizCategories = bizCategories.difference(bannedCategories)
        userCategories = userCategories.difference(bannedCategories)
        intersection = bizCategories.intersection(userCategories)
        union = bizCategories.union(userCategories)
        jacard[i] = 1.0*len(intersection)/len(union)
        i += 1
    return jacard

def preprocessBaseline(UIPairs):
    userAverages = np.zeros(len(UIPairs))
    i = 0
    for user, _ in UIPairs:
        userAverages[i] = user.findAverageStars()
        i += 1
    return userAverages

def preprocessUIPairs(UIPairs, users, bizs, dictVectCategoriesInput=None, dictVectAttributesInput=None):
    # inArrayForm = []
    # for vec in UIPairs:
    #     inArrayForm.append(vec.toarray())
    # vectorizedUIPairs = np.vstack(inArrayForm)
    preprocessUsers(users)
    dictVectCategories, dictVectAttributes = preprocessBizs(bizs, dictVectCategoriesInput, dictVectAttributesInput)

    featureVector = np.ones((len(UIPairs), 1))

    user0 = UIPairs[0][0]
    biz0 = UIPairs[0][1]
    lenVectorizedTextArray = max(user0.getVectorizedText().shape) + max(biz0.getVectorizedText().shape)
    lenCategories = max(biz0.getFeaturizedCategories().shape) if dictVectCategories is None else len(dictVectCategories.feature_names_)
    lenAttributes = max(biz0.getFeaturizedAttributes().shape) if dictVectAttributes is None else len(dictVectAttributes.feature_names_)

    vectorizedUITexts = np.zeros((0, lenVectorizedTextArray))
    vectorizedCategories = np.zeros((0, lenCategories))
    vectorizedAttributes = np.zeros((0, lenAttributes))
    # featureVector = np.concatenate((featureVector, vectorizedUITexts))

    #add featurized categories for restaurants
    for _, biz in UIPairs:
        vcs = biz.getFeaturizedCategories()
        vcs = vcs.reshape((1, lenCategories))
        vectorizedCategories = np.append(vectorizedCategories, vcs, axis=0)
    featureVector = np.hstack((featureVector, vectorizedCategories))

    #add featurized attributes for restaurants
    for _, biz in UIPairs:
        vcs = biz.getFeaturizedAttributes()
        vcs = vcs.reshape((1, lenAttributes))
        vectorizedAttributes = np.append(vectorizedAttributes, vcs, axis=0)
    featureVector = np.hstack((featureVector, vectorizedAttributes))

    #add rating averages for user and restaurant
    avgRatingPairs = np.zeros((0, 4))
    for user, biz in UIPairs:
        avgRatingPair = np.array([user.findAverageStars(), user.findStarStdDev(), biz.findAverageStars(), biz.findStarStdDev()])
        avgRatingPair = avgRatingPair.reshape((1, 4))
        avgRatingPairs = np.append(avgRatingPairs, avgRatingPair, axis=0)
    featureVector = np.hstack((featureVector, avgRatingPairs))

    # add features for vectorized texts
    for user, biz in UIPairs:
        # vectorizedUIPair = sp.hstack((user.getVectorizedText(), biz.getVectorizedText()), format='csr')
        vectorizedUIPair = np.hstack((user.getVectorizedText(), biz.getVectorizedText()))
        vectorizedUIPair = vectorizedUIPair.reshape(1, 600)
        # vectorizedUIPair = vectorizedUIPair.toarray()
        vectorizedUITexts = np.append(vectorizedUITexts, vectorizedUIPair, axis=0)
    featureVector = np.hstack((featureVector, vectorizedUITexts))

    return featureVector, dictVectCategories, dictVectAttributes



def load_train_data(X, Y, train_size=0.8):
    # The competition datafiles are in the directory ../input
    # Read competition data files:
    # np.random.shuffle(X)
    
    # Test training and validations sets
    # Note in this case training data has -
    # predictors: starting from second column to end
    # targets: in first column
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, Y, train_size=train_size, random_state=42)
    print(" -- Loaded data.")
    # print("Training set has {0[0]} rows and {0[1]} columns".format(X.shape))

    return (X_train, X_valid,
            y_train, y_valid)

def train(X,Y,n_estimators=20,model='Random Forest',kernel='linear'):
    # define regressor
    #model='Random Forest'
    #model='Linear Regression' 
    X_train, X_valid, y_train, y_valid = load_train_data(X,Y)

    if (model=='Random Forest'):
        regr = RandomForestRegressor(n_estimators=n_estimators)
    elif (model=='Linear Regression'):
        regr = linear_model.LinearRegression()
    elif (model=='SVM'):
        if (kernel=='linear'):
            regr = svm.SVR(kernel='linear',C=1.0)
        elif (kernel=='poly'):
            regr = svm.SVR(kernel='poly',degree=1)

    # Start training
    print(" -- Start training Random Forest Classifier. Number of trees = "+str(n_estimators))
    regr.fit(X_train,y_train)
    y_valid=y_valid.astype(float)
    y_pred=regr.predict(X_valid).astype(float)
    print(" -- Finished training.")

    print("  Calculate CV score")
    score = sqrt(sum((y_pred-y_valid)**2)/len(y_pred))
    return regr, score

def example():
    print(" - Start.")

    np.random.seed(17411)
    #CVscore=[]
    #treenum=range(1,21,2)
    #for n_estimators in treenum:
    #    clf,encoder,score = train(n_estimators)
    #    print("CV Score = "+str(score))
    #    CVscore.append(score)
    #plt.plot(treenum,CVscore)
    #plt.show()

    # load traning set
    # first column = y
    # second to end = x
    Xtrain = np.random.rand(40,3)
    Ytrain = np.random.rand(40)

    # start training
    #regr,score = train(Xtrain,model='SVM')
    regr,score = train(Xtrain,Ytrain,model='Random Forest',n_estimators=1)
    print("CV Score = "+str(score))

    # load test set
    Xtest = np.random.rand(5,3)

    # make prediction
    y_pred = regr.predict(Xtest)
    print(" - Finished.")

if __name__ == '__main__':
    example()
