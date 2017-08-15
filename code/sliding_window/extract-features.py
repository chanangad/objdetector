# Import the functions to calculate feature descriptors
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
from skimage.transform import resize
# To read file names
import argparse as ap
import glob
import os
from config import *
import numpy as np
import cPickle as pickle

if __name__ == "__main__":
    dirs = os.listdir('../data/train')
    
    fds = []
    labels = []
    
    for path in dirs:
        filenames = os.listdir('../data/train/' + path)
        for filename in filenames:
            im_path = '../data/train/' + path + '/' + filename
            im = imread(im_path, as_grey=True)
            im = resize(im, (min_wdw_sz[0],min_wdw_sz[1]))
            fd = hog(im,  orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            fds.append(fd)
            labels.append(path)
        print 'Extracted features for ' + path

    fds = np.array(fds)
    labels = np.array(labels)

    pickle.dump(fds, open('../data/features/feat.p', 'wb'))
    pickle.dump(labels, open('../data/features/labels.p', 'wb'))

    print "Completed calculating features from training images"
