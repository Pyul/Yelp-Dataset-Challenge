import sys

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import sklearn as sklearn
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
#from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from math import sqrt

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
