# TensorFlow With Own Data

CNN classification with TensorFlow, make .pkl files from pictures and read it.

# Overview

This project can help you make .pkl files from pictures and use TensorFlow to classify.

# Dependencies

- cv2
- gzip
- math
- matplotlib
- numpy
- pickle
- random
- shutil
- tensorflow
- time

You can install missing dependencies with [pip](https://pip.pypa.io/en/stable/ "pip"). And install TensorFlow via [TensorFlow](https://www.tensorflow.org/install/ "TensorFlow") link.

# Required directory structure needed to create your own dataset:

-- database
|	|	-- test (Folder containing the pictures that will be used for testing).
|	|	-- train (Folder containing the pictures that will be used for training).
|	test.txt (.txt file that contains the testing labels).
|	train.txt (.txt file that contains the training labels).

dataset.zip contains pictures and .txt files which contains picture's path and labels.

# The directory structure for our project:

We created a dateP module to keep our code organized, and inside the dateP module we created dateP.py which provides a method for reading images and labels, then write them both in the .pkl file.

We created a new file named makedata.py that opens and saves our dataset in .pkl format.

-- database
|	|	-- dataset (Our database as explained above).
|	|	-- dateP (Our module as explained above).
|	makedata.py (Run it).

# Usage

1. Install the dependencies;
2. Open terminal and run makedata.py file.

# Other information:

The dataset.pkl.gz file is already the converted file for use in TensorFlow.

The unpack.py file is used to unpack .pkl files and to plot some pictures on the screen.

The examples folder contains other .py files showing ways to use it on a CNN via TensorFlow.

# License

Feel free to use this source code to modify or deploy to your own project.