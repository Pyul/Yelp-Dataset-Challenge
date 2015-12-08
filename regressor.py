import sys

import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.cross_validation import train_test_split
import sklearn as sklearn
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.feature_extraction import DictVectorizer

def preprocessUsers(users):
    pass

def preprocessBizs(bizs):
    categoryDicts = []
    for biz in bizs:
        categoryDicts.append(dict.fromkeys(biz.getCategories(), 1))
    dictVect = DictVectorizer(sparse=False)
    featurizedCategories = dictVect.fit_transform(categoryDicts)
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
    featurizedAttributes = dictVect.fit_transform(attributeDicts)
    for i in xrange(len(bizs)):
        bizs[i].setFeaturizedAttributes(featurizedAttributes[i])


def preprocess(UIPairs, reviewStars, users, bizs):
    # inArrayForm = []
    # for vec in UIPairs:
    #     inArrayForm.append(vec.toarray())
    # vectorizedUIPairs = np.vstack(inArrayForm)
    preprocessUsers(users)
    preprocessBizs(bizs)

    featureVector = np.ones((len(UIPairs), 1))

    user0 = UIPairs[0][0]
    biz0 = UIPairs[0][1]
    lenVectorizedTextArray = max(user0.getVectorizedText().shape) + max(biz0.getVectorizedText().shape)
    lenCategories = max(biz0.getFeaturizedCategories().shape)
    lenAttributes = max(biz0.getFeaturizedAttributes().shape)

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
        vectorizedUIPair = sp.hstack((user.getVectorizedText(), biz.getVectorizedText()), format='csr')
        vectorizedUIPair = vectorizedUIPair.toarray()
        vectorizedUITexts = np.append(vectorizedUITexts, vectorizedUIPair, axis=0)
    featureVector = np.hstack((featureVector, vectorizedUITexts))

    return featureVector, np.array(reviewStars)

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

    return (X_train, X_valid, y_train, y_valid)

def train(X,Y,train_size=0.8,n_estimators=20,model='Random Forest',kernel='linear',degree=3):
    # define regressor
    #model='Random Forest'
    #model='Linear Regression' 
    X_train, X_valid, y_train, y_valid = load_train_data(X,Y,train_size)

    #encode data (handle categorical data)
    #encoder = LabelEncoder()
    #for i in range(0,len(X_train[0,:])):
    #    X_train[:,i] = encoder.fit_transform(X_train[:,i])
    #    X_valid[:,i] = encoder.fit_transform(X_valid[:,i])

    if (model=='Random Forest'):
        regr = RandomForestRegressor(n_estimators=n_estimators)
    elif (model=='Linear Regression'):
        regr = linear_model.LinearRegression()
    elif (model=='SVM'):
        if (kernel=='linear'):
            regr = svm.SVR(kernel='linear')
        elif (kernel=='poly'):
            regr = svm.SVR(kernel='poly',degree=degree)

    # Start training
    print(" -- Start training regression. Number of trees = "+str(n_estimators))
    regr.fit(X_train,y_train)

    # Start validation
    y_train_pred=regr.predict(X_train)
    y_valid_pred=regr.predict(X_valid)
    print(" -- Finished training.")

    print("  Calculate training and test errors")
    trainscore = sum((y_train_pred-y_train)**2)/float(len(y_train))
    testscore = sum((y_valid_pred-y_valid)**2)/float(len(y_valid))
    return regr, trainscore, testscore

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
    regr,trainscore, testscore = train(Xtrain,Ytrain,model='Random Forest')
    print("Training error = "+str(trainscore)+' // Test error = ' + str(testscore))

    # load test set
    Xtest = np.random.rand(5,3)

    # make prediction
    y_pred = regr.predict(Xtest)
    print(" - Finished.")

if __name__ == '__main__':
    example()
