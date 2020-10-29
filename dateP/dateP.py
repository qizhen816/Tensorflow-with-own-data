#!/usr/bin/python
# -*- coding: utf-8 -*-

# Imports
import cv2
import gzip
import numpy as np
import pickle
import random

class DateP(object):
    def __init__(self, train_file,test_file, total_size):
        self.train_file = train_file
        self.test_file  = test_file
        self.total_size = total_size

    def date_process(self):
        label = []
        all_data = []
        print('\nReading...\n')

        # Read train.txt
        with open(self.train_file) as f:
            for line in f:

                # Separate images and labels
                image_file = line.strip().split(' ')[0]
                label_file = line.strip().split(' ')[1]

                '''
                print 'Image file:',image_file
                print 'Label file:',label_file
                '''

                # Append all labels
                label.append(str(int(label_file)))

                # To get path of images
                path_image = 'dataset/'+image_file

                # Get data location information and original size
                data = data_pro(path_image, 0, 0, 384, 256)

                # Append all data
                all_data.append(data)

        # Read test.txt     
        with open(self.test_file) as f:
            for line in f:

                # Separate images and labels
                image_file = line.strip().split(' ')[0]
                label_file = line.strip().split(' ')[1]

                '''
                print 'Image file:',image_file
                print 'Label file:',label_file
                '''

                # Append all labels
                label.append(str(int(label_file)))

                # To get path of images
                path_image = 'dataset/'+image_file

                # Get data location information and original size
                # and send to data_pro()
                data = data_pro(path_image, 0, 0, 384, 256)

                # Append all data
                all_data.append(data)

        image = np.asarray(all_data, dtype=float)
        labels = np.asarray(label, dtype=int)

        # It divides left operand with the right
        # operand and assign the result to left operand
        image /= 255

        # returns the number of elements in all_data
        lens = len(all_data)

        # Gives a new shape to image without changing its data.
        image = image.reshape(lens, self.total_size)

        # To zip
        toZip = list(zip(image, labels))
        random.shuffle(toZip)
        datas, labelss = map(list, zip(*toZip))

        Data = np.asarray(datas, dtype=float)
        Lable = np.asarray(labelss, dtype=int)
        All = Data, Lable

        return All

def data_pro(src, x1, y1, x2, y2):
    # Read image
    image_ = cv2.imread(src)

    # ROI (region of interest)
    ROI = image_[x1:x2, y1:y2]

    # New size
    size = (28, 28)

    # Resize image
    image = cv2.resize(image_, size)

    # Converts the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    bb = np.zeros((image.shape[0], image.shape[1]), dtype=image.dtype)
    bb[:, :] = gray_image[:, :]
    cc = bb.reshape(1, image.shape[0] * image.shape[1])
    return cc