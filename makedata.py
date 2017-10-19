#!/usr/bin/python
# -*- coding: utf-8 -*-

# Imports
from dateP.dateP import DateP
import cv2
import gzip
import numpy as np
import pickle
import random
import shutil

# train.txt file path, should have 'imagepath label' as each row in .txt
train_file = 'dataset/train.txt'

# text.txt same row	
test_file = 'dataset/test.txt'   	

# train, test and the total size of one image (28*28)
Object = DateP(train_file, test_file, 784)
O1 = Object.date_process()
d = O1

# Generate the pkl.gz file
p1 = pickle.dumps(d, 2)

# Save as .gz
file = gzip.open('dataset.pkl.gz', 'wb')

# Print message
print('writing...\n')

# Write and then close file
file.write(p1)
file.close()

print('Done!\n')