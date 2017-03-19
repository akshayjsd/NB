#!/usr/bin/python

"""
    Understanding Naive Bayes from Udacity's Intro to ML

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print "Training time: ", round(time()-t0,3), "s"
t1 = time()
pred = clf.predict(features_test)
print "Prediction time: ", round(time()-t1,3), "s"
print(pred)
print(accuracy_score(pred, labels_test))

#no. of Chris training emails: 7936
#no. of Sara training emails: 7884
#Training time:  4.463 s
#[0 0 1 ..., 1 0 0]
#Prediction time:  0.405 s
#Accuracy---0.973265073948




#########################################################
