# Import the functions to calculate feature descriptors
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
from skimage.transform import resize
# To read file names
import argparse as ap
import glob
import os, sys, cv2, imutils
import numpy as np
import cPickle as pickle

if __name__ == "__main__":
	dirs = os.listdir('../train')

	fds = []
	labels = []
	sift = cv2.xfeatures2d.SIFT_create()

	for path in dirs:
		filenames = os.listdir('./train/' + path)
		for filename in filenames:
			im_path = './train/' + path + '/' + filename
			im = cv2.imread(im_path)
			gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

			kp, fd = sift.detectAndCompute(gray, None)

			if len(kp) == 0:
				continue

			fd /= (fd.sum(axis=1, keepdims=True) + 0.0000001)
			fd = np.sqrt(fd)
			fds.append(fd)
			labels.append(path)

		print 'Extracted features for ' + path

	fds = np.array(fds)
	print fds.shape
	labels = np.array(labels)

	pickle.dump(fds, open('./features/features_sift.p', 'wb'))
	pickle.dump(labels, open('./features/labels_sift.p', 'wb'))

	print "Completed calculating SIFT features from training images"
