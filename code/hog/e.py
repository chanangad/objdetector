import numpy as np
import imutils
import cv2
import sys
from skimage.feature import hog
from sklearn.externals import joblib
from config import *
import sys,os,re
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
from PIL import Image
#CvFont font = fontQt(''Times'')

def background_subtract(filename):
	cap = cv2.VideoCapture(filename)
	fgbg = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold = 30, detectShadows = True)
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5),(3,3))

	ret, frame = cap.read()
	# frame = imutils.resize(frame, width=500)
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# gray = cv2.GaussiaBlur(gray, (21, 21), 0)
	frame = imutils.resize(frame, width=(len(frame)))
	gray = cv2.blur(frame,(3,3))
	fgmask = fgbg.apply(gray)

	i = 1
	j = 1
	k = 1
	while True:
		# print i
		ret, frame = cap.read()
		#cv2.imshow("Security Feed", frame)
		#print frame
		if ret == False:
			break

			# resize the frame, convert it to grayscale, and blur it
		# frame = imutils.resize(frame, width=500)
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (11, 11), 0)
			frame = imutils.resize(frame, width=(len(frame)))
			gray = cv2.blur(frame,(3,3))
			fgmask = fgbg.apply(gray)
			thresh = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
			thresh = cv2.threshold(thresh, 100, 255, cv2.THRESH_BINARY)[1]

			# dilate the thresholded image to fill in holes, then find contours
			# on thresholded image

			# thresh = cv2.erode(thresh, kernel, iterations=1)
			cv2.imshow('thresh without erosion', thresh)
			(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			# loop over the contours
			for c in cnts:
				# if the contour is too small, ignore it
				if cv2.contourArea(c) < 1200:
					continue

					# compute the bounding box for the contour, draw it on the frame,
				# and update the text
			(x, y, w, h) = cv2.boundingRect(c)
			# print x,y,w,h
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			image = frame[y:y+h,x:x+w]
			cv2.imwrite('image.png', image)
			image = Image.open('image.png').convert('L')
			image = image.resize((100,100),Image.ANTIALIAS)
			data = np.asarray(image.getdata()).reshape((100,100))
			fd = hog(data, orientations=9, pixels_per_cell=(8,8),
				 cells_per_block=(3, 3), visualise=False)
			prediction=clf.predict(fd)
			image = frame[y:y + h/2, x : x + w]
			image = Image.open('image.png').convert('L')
			image = image.resize((100,100),Image.ANTIALIAS)
			data = np.asarray(image.getdata()).reshape((100,100))
			fd = hog(data, orientations=9, pixels_per_cell=(8,8),
				 cells_per_block=(3, 3), visualise=False)
			prediction2=clf.predict(fd)
					if(prediction2 == 4 or prediction2 == 3):
						prediction = prediction2

						if (prediction[0]==4):
							print "Bicycle"
							cv2.putText(frame,'Bicycle',(x,y),cv2.FONT_HERSHEY_PLAIN , 1,(255,255,255),2,cv2.LINE_AA)
						elif (prediction[0]==1):
							print "Car"
							cv2.putText(frame,'Car',(x,y), cv2.FONT_HERSHEY_PLAIN , 1,(255,255,255),2,cv2.LINE_AA)
						elif (prediction[0]==2):
							print "Person"
							cv2.putText(frame,'Person',(x,y),cv2.FONT_HERSHEY_PLAIN , 1,(255,255,255),2,cv2.LINE_AA)
						elif (prediction[0]==3):
							print "MOtorcycle"
							cv2.putText(frame,'MOtorcycle',(x,y),cv2.FONT_HERSHEY_PLAIN , 1,(255,255,255),2,cv2.LINE_AA)
							#elif (prediction==4):
							#	print "Rickshaw"
						else:
							cv2.putText(frame,'Rickshaw',(x,y),cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),2,cv2.LINE_AA)
							print "rickshaw"
							j += 1
							text = "Occupied"

							cv2.imshow("Security Feed", frame)
							#cv2.imshow("threshold", thresh)

							key = cv2.waitKey(1) & 0xFF

							# if the `q` key is pressed, break from the lop
							if key == ord("q"):
								break

								cap.release()
								cv2.destroyAllWindows()

								if __name__ == '__main__':
									if len(sys.argv) == 2:
										filename = sys.argv[1]
										clf = joblib.load(model_path)
										background_subtract(filename)
