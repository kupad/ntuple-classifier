"""
mnist_ntuple.py
Author: Phil Dreizen
requires libraries: numpy, numba

Uses the N-tuple Classifier on MNIST handwritten digit data for classification. We print out a confusion matrix for the test data.
The accuracy tends to be ~91%

The N-tuple classifier works by using subclassifiers (called modules). 
When training: 
each module will sample the same random pixel positions from each image, 
determining which of the pixels are "active" (have a value above a threshold).
A table is used to keep track of the number of times each label was encountered for the active sampled pixels.

When classifying an image:
for each module, we sample the same pixel positions, 
and we sum up the number of times we've seen each label for the active sample pixels. 
The label with the highest count will be the label we choose.


Get the dataset here: http://yann.lecun.com/exdb/mnist/

USAGE:
    #help:
    $ python3 mnist_ntuple.py -h

    #example usage:
    $ python3 mnist_ntuple.py data/train-images-idx3-ubyte data/train-labels-idx1-ubyte data/t10k-images-idx3-ubyte data/t10k-labels-idx1-ubyte

TODO: 
    * The classifier is currently written for the MNIST data set. But it can easily be generalized to any data that can be quantized into 0/1, either by providing
        it a quantize function, or requiring that the data provided is already quantized
    * Before selecting random pixels to sample, we can examine the training data for pixel positions that are rarely "active" and exclude them from the modules
    * I'm currently using a lot of memory to store the count information data. It might be worth switching to a dictionary. 
        I don't think it will be much slower, but might save a lot of memory.
    * Print out examples of images that were categorized incorrectly?
"""

import argparse 

#numpy, and especially numba's JIT compilation, significantly speed up the code. 
#(ie: On my computer, the code is 300x faster with JIT compiling)
import numpy as np
from numba import jit

from mnist_reader import MNISTReader
from utils import eprint

#number of labels (0-9)
L = 10

M = 500 #num modules. Each module will sample pixels in the image data to predict the label
N = 10  #num pixels sampled per module

"""
JIT: 
Using numba's JIT makes the code much much faster
But, numba's jit compiling cannot be applied to methods in classes. So the functions optimized with it are here, outside the classifying class
"""

@jit(nopython=True)
def _update_count_tbl(img, label, idxs, count_tbl):
    """
    update table counts

    img: an np.array of pixel data. each pixel contains a value of 0-255
    label: what number this image represents
    idxs: the indexes of the pixels we'll be sampling. There are M modules each sampling N pixels
    count_tbl: an np.array. it's a giant in-memory hash table for storing the count data. (module#,active pixel hash, label) -> count 
   
    For each module,
        we store which sample pixels are active in a bitvector integer variable, 
        which we use, along with the module number, as a hash key into the table, 
        and increment the number of times we've seen the given label by 1
    """
    for m in range(M):
        h = 0 #bitvector of active pixels
        for n in range(N):
            h += (1 if img[idxs[m][n]] >= 128 else 0) << n
        count_tbl[m][h][label] += 1


@jit(nopython=True)
def _classify(img, idxs, count_tbl):
    """
    classify an image (is it 0,1...9?)
    
    img: an np.array of pixel data. each pixel contains 0-255
    idxs: the indexes of the pixels we'll be sampling. There are M modules each sampling N pixels
    count_tbl: an np.array. it's a giant in-memory hash table for storing the count data. (module#,active pixel hash, label) -> count 
    returns: label with most "votes"
    
    Each module will "vote" for the labels it thinks the image belongs to, contributing higher mumbers to the labels it matched on more often.
    
    For each module, 
        we store which sample pixels are active in a bitvector,
        which we use as a hash key into the table,
        we iterate over the 10 labels. the values stored are considered "votes" for the given label

    The label with the most votes at the end will be the label returned

    """
    votes = np.zeros(L)
    for m in range(M):
        h = 0
        for n in range(N):
            h += (1 if img[idxs[m][n]] >= 128 else 0) << n 
        
        for l in range(L):
            votes[l] += count_tbl[m][h][l]
    maxlabel = np.argmax(votes)
    return maxlabel

class NTupleClassifier:
    """
    Use the ntuple method for classifying handwritten image data
    """

    def __init__(self):
        self.idxs = None #np.array of M modules that sample N pixels each
        self.count_tbl = None #np.array that acts as an in-memory hash table that stores the times a given module seen each label
    
    def train(self, train_x, train_y):
        """
        train_x: the image data we read in from MNIST. Each row represents an image, which is represented as a flat array of pixels
        train_y: the labels of each image we read in

        * initialize self.idxs by choosing M*N random pixels we will be sampling
        * initalize self.count_tbl by zeroing out a big 3 dimensional np.array to store the label counts for each module

        training the classifier means updating the table with our counts, which we'll later use to make predictions
        """
        eprint("training...")
       
        #the number of pixels in each image
        npixels = train_x.shape[1]
        
        #generate our random pixel indexes
        self.idxs = np.random.randint(0, high=npixels, size=(M,N))
        
        #zero out an array for counts
        self.count_tbl = np.zeros((M,1<<N,L), int)

        #for each image:
        #   - check to see if, given our current table, we would properly classify the image.
        #   - if we would not have classified the image correctly, update the table
        # 
        #Checking if we would have classified things correctly acts as a normalization. If we didn't do it, our predictions would be skewed
        #toward the labels that happened to show up in our training data more often.
        for i in range(train_x.shape[0]):
            img = train_x[i]
            actual = train_y[i] #the actual 
            prediction = _classify(img, self.idxs, self.count_tbl)
            if actual != prediction:
                _update_count_tbl(img,actual, self.idxs, self.count_tbl)
       
    def test(self, test_x, test_y):
        """
        test_x: test image data we read in from MNIST
        test_y: test label data

        For each image in the test set, we'll attempt to classify it, and compare how we did with the actual label of the image.
        We'll then print out a confusion matrix and our accuracy.
        """
        eprint("testing...")
    
        #classify each image
        #populate the the confusion matrix,
        cm = np.zeros( (L,L), dtype=int)
        for i in range(test_x.shape[0]):
            img = test_x[i]
            actual = test_y[i]
            prediction = _classify(img, self.idxs, self.count_tbl);
            cm[actual][prediction] += 1
       
        #print the confusion matrix and accuracy
        print("Confusion Matrix:")
        row_format = "{:>8}" * (L + 1)
        print(row_format.format("", *range(L)))
        for a in range(L):
            print(row_format.format(a, *[cm[a][p] for p in range(L)]))

        accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)

        print(f"accuracy: {accuracy:.4f}")
    
def main():
    #np.random.seed(42) #for testing

    parser = argparse.ArgumentParser()
    parser.add_argument("train_images_file", help="file with the training data")
    parser.add_argument("train_labels_file", help="file with the training labels")
    parser.add_argument("test_images_file", help="file with the testing data")
    parser.add_argument("test_labels_file", help="file with the testing labels")
    args = parser.parse_args()

    eprint("reading in training data...")
    reader = MNISTReader()
    train_x = reader.read_img(args.train_images_file)
    train_y = reader.read_labels(args.train_labels_file)
    model = NTupleClassifier()
    model.train(train_x, train_y)

    eprint("reading in test data...")
    test_x = reader.read_img(args.test_images_file)
    test_y = reader.read_labels(args.test_labels_file)
    model.test(test_x, test_y)
    
if __name__ == '__main__':
    main()

