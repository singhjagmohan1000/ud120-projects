#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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

from sklearn.svm import SVC
import time
import numpy as np

#clf = SVC(kernel="linear")
#clf = SVC(kernel="rbf")  .61
#clf = SVC(C=10,kernel="rbf") .61
#clf = SVC(C=100,kernel="rbf") .61
#clf = SVC(C=1000,kernel="rbf") .821
clf = SVC(C=10000,kernel="rbf")   #.892

#slicing input set for training
#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]
tick = time.time()
clf.fit(features_train,labels_train)
print ("Time in training: ", time.time()-tick)
testtime = time.time()
pred = clf.predict(features_test)
print ("Time in prediction: ", time.time()-testtime)

print (pred)

print (np.count_nonzero(pred))

#print (pred[10],"  ",pred[26], " ",pred[50])


print ("Accuracy: ",clf.score(features_test,labels_test))


#########################################################


