# mnist_ntuple.py
#
# Phil Dreizen
# 
# requires libraries: numpy, numba
#
# python3 mnist_ntuple.py train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
#
#
# get the dataset here: http://yann.lecun.com/exdb/mnist/

import argparse 
import numpy as np
from numba import jit

from mnist_reader import MNISTReader
from utils import eprint

#number of labels (0-9)
L = 10

M = 500 #num modules
N = 10  #num pixels sampled per module

@jit(nopython=True)
def _classify(img, idx, tbl):
    votes = np.zeros(L)
    for m in range(M):
        h = 0
        for n in range(N):
            h += (1 if img[idx[m][n]] >= 128 else 0) << n
        
        for l in range(L):
            votes[l] += tbl[m][h][l]
    maxlabel = np.argmax(votes)
    return maxlabel

@jit(nopython=True)
def _update_tbl(img, label, idx, tbl):
    for m in range(M):
        h = 0
        for n in range(N):
            h += (1 if img[idx[m][n]] >= 128 else 0) << n
        tbl[m][h][label] += 1

class NTupleClassifier:

    def __init__(self):
        self.idx = None
        self.tbl = None
    
    def train(self, train_x, train_y):
        eprint("training...")
        
        npixels = train_x.shape[1]
        
        #generate our random pixel indexes
        self.idx = np.random.randint(0, high=npixels, size=(M,N))
        
        #zero out an array for counts
        self.tbl = np.zeros((M,1<<N,L), int)

        for i in range(train_x.shape[0]):
            img = train_x[i]
            actual = train_y[i]
            prediction = _classify(img, self.idx, self.tbl)
            if actual != prediction:
                _update_tbl(img,actual, self.idx, self.tbl)
       
    def test(self, test_x, test_y):
        eprint("testing...")
     
        #calculate the confusion matrix
        cm = np.zeros( (L,L), dtype=int)
        for i in range(test_x.shape[0]):
            img = test_x[i]
            actual = test_y[i]
            prediction = _classify(img, self.idx, self.tbl);
            cm[actual][prediction] += 1
        
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

