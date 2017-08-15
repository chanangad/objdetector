# Import the required modules
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
import argparse as ap
import glob
import os, time
from config import *
import numpy as np
import cPickle as pickle
from sklearn import cross_validation, metrics

y = pickle.load(open('./features/labels_sift.p', 'rb'))

def getstat(labels, predictions):
	print "Accuracy = ", metrics.accuracy_score(labels, predictions),"Precision = ", metrics.precision_score(labels, predictions, pos_label = None, average = 'macro'),"Recall = " ,metrics.recall_score(labels, predictions, pos_label = None, average = 'macro'),"F1 score = " ,metrics.f1_score(labels, predictions, pos_label = None, average = 'macro')

for i in range(100,1001,100):
	X = pickle.load(open('./features/sift_' + str(i) + '.p', 'rb'))
	clf = OneVsRestClassifier(LinearSVC(random_state = 0))
	# clf = RandomForestClassifier(n_estimators = 200, n_jobs = -1)
	# clf = AdaBoostClassifier(n_estimators = 100)
	predicted = cross_validation.cross_val_predict(clf, X, y, cv=5, n_jobs = -1)

	print "For num_clusters = " + str(i) + ":" 
	getstat(y,predicted)