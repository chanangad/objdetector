import cv2
import pickle
import os
from sklearn.cluster import MiniBatchKMeans
import time
import numpy as np

x = pickle.load(open("./features/features_sift.p","rb"))
y = pickle.load(open("./features/labels_sift.p","rb"))

X = []
for k in x:
	X.extend(k)

X = np.array(X)

num_clusters_list = [100,200,300,400, 500, 600, 700, 800, 900, 1000]

for num_clusters in num_clusters_list:
	mbk = MiniBatchKMeans(init='k-means++', n_clusters=num_clusters, batch_size = 100, n_init = 5)
	mbk.fit(X)
	mbk_means_labels = mbk.labels_
	
	feat_vector = [[0 for i in range(num_clusters)] for j in range(len(y))]
	
	pos=0
	for i in range(len(y)):
		for j in range(len(x[i])):
			feat_vector[i][mbk_means_labels[pos]] += 1
			pos +=1
	
	pickle.dump(np.array(feat_vector),open("./features/sift_"+ str(num_clusters) +".p","wb"))
	print "k-means clustering done for num_clusters = " + str(num_clusters)
