# Import the required modules
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import cv2, sys
import argparse as ap
from nms import nms
from config import *
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
import cPickle as pickle
from skimage.transform import rescale

def sliding_window(image, window_size, step_size):
	for y in xrange(0, image.shape[0], step_size[1]):
		for x in xrange(0, image.shape[1], step_size[0]):
			yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def testimage(im,clf):
	downscale = 1.25
	visualize_det = "store_true"

	# Load the classifier

	# List to store the detections
	detections = []
	# The current scale of the image
	scale = 0
	# Downscale the image and iterate
	for im_scaled in pyramid_gaussian(im, downscale=downscale):
		# This list contains detections at the current scale
		# If the width or height of the scaled image is less than
		# the width or height of the window, then end the iterations.
		if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
			break
		for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
			if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
				continue
			# Calculate the HOG features
			fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
			# print im_window.shape
			fd = np.array(fd).reshape((1,-1))
			# print fd.shape
			# break
			pred = clf.predict(fd)
			if pred == 'Bicycle':
				# print  "Detection:: Location -> ({}, {})".format(x, y)
				# print "Scale ->  {} | Confidence Score {} \n".format(scale,clf.decision_function(fd))
				detections.append((x, y, x + int(min_wdw_sz[0]*(downscale**scale)),y + int(min_wdw_sz[1]*(downscale**scale))))
			# If visualize is set to true, display the working
			# of the sliding window
		# Move the the next scale
		scale+=1

	# Display the results before performing NMS
	clone = im.copy()

	# Perform Non Maxima Suppression
	detections = nms(detections, 0.5)

	# Display the results after performing NMS
	for (x_tl, y_tl, x_br, y_br) in detections:
		# Draw the detections
		cv2.rectangle(clone, (x_tl, y_tl), (x_br,y_br), (0, 0, 0), thickness=2)

	return clone

if __name__ == '__main__':
	filename = sys.argv[1]

	model_path = '../data/models/' + sys.argv[2]

	clf = pickle.load(open(model_path, 'rb'))

	cap = cv2.VideoCapture(filename)
	i = 0

	while True:
		ret, frame = cap.read()

		if ret == False:
			break

		i = (i + 1)%30

		if i != 1:
			continue
		
		gray = rescale(frame, 1.0/3)
		gray = rgb2gray(gray)

		out = testimage(gray,clf)
		cv2.imshow('output', out)
		cv2.waitKey()

	cap.release()
	cv2.destroyAllWindows()
