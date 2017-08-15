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
	'''
	This function returns a patch of the input image `image` of size equal
	to `window_size`. The first image returned top-left co-ordinates (0, 0) 
	and are increment in both x and y directions by the `step_size` supplied.
	So, the input parameters are -
	* `image` - Input Image
	* `window_size` - Size of Sliding Window
	* `step_size` - Incremented Size of Window

	The function returns a tuple -
	(x, y, im_window)
	where
	* x is the top-left x co-ordinate
	* y is the top-left y co-ordinate
	* im_window is the sliding window image
	'''
	for y in xrange(0, image.shape[0], step_size[1]):
		for x in xrange(0, image.shape[1], step_size[0]):
			yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def testimage(im):
	downscale = 1.25
	visualize_det = "store_true"

	# Load the classifier
	clf = pickle.load(open(model_path, 'rb'))

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
			if pred != 'None':
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

	return detections

videofiles =['datasample1.mov', 'input_video_sample1.mov', 'input_video_sample2.mov', 'input_video_sample3.mov','nov92015-1.dav', 'nov92015-2.dav', 'dec21h1330.dav', 'videosample5.mov']

for filename in videofiles:
	cap = cv2.VideoCapture('../videos/' + filename)
	j = pickle.load(open('tempnone.p', 'rb'))
	i = 0
	print j

	while True:
		ret, frame = cap.read()

		if ret == False:
			break

		i = i + 1

		if i % 300 != 1:
			continue

		print filename, i
		
		gray = rescale(frame, 1.0/3)
		gray = rgb2gray(gray)

		detections = testimage(gray)
		
		for rect in detections:
			xtl = 3*rect[0]
			ytl = 3*rect[1]
			xbr = 3*rect[2]
			ybr = 3*rect[3]

			cv2.imwrite('../data/train/None/' + str(j) + '.png', frame[ytl:ybr, xtl:xbr])
			j += 1

	print j

	pickle.dump(j, open('tempnone.p', 'wb'))

	cap.release()

cv2.destroyAllWindows()
