# N-tuple Classifier

Here I use the N-tuple Classifier to classify handwritten images of numbers (0-9) using the MNIST handwritten digit data. The program will train the classifier using the training data, test the performance using the test data, and then summarize performance with a confusion matrix and will print out the accuracy. 

## Requirements

To run, you will need to install the [numpy](https://numpy.org/) and [numba](https://numba.pydata.org/) packages. This is for performance.

You'll also need the [MNIST data](http://yann.lecun.com/exdb/mnist/), a copy of which I've checked into this repository.

## To Execute

The main program is  `mnist_ntuple.py`

```
$ python3 mnist_ntuple.py data/train-images-idx3-ubyte data/train-labels-idx1-ubyte data/t10k-images-idx3-ubyte data/t10k-labels-idx1-ubyte 
```

## Description

The N-tuple classifier is pretty neat. It's relatively simple, potentially quite fast, and even with a naive implementation has an accuracy of around 91% 
on the MNIST data. But it isn't very popular, and is not typically discussed in introductory data science material.

The N-tuple classifier works by using subclassifiers (called modules). 
When training: 
each module will sample a small number of pixel positions from each image, determining which of the pixels are "active" (have a value above a threshold).
A table is used to keep track of the number of times each label was encountered for the active sampled pixels.

When classifying an image:
for each module, we sample the same pixel positions, 
and we sum up the number of times we've seen each label given the active sample pixels. 
The label with the highest count will be the label we choose.

For more information on n-tuple classifiers:
* [Bledsoe and Browning.  Pattern recognition and reading by machine.](https://dl.acm.org/doi/10.1145/1460299.1460326) - The original paper
* http://www.theparticle.com/cs/bc/dsci/ntuple.pdf - Which this python implementation is based on
* https://haralick.org/ML/ntuple_classifier_10_2_2020.pdf

USAGE:
```
$ python3 mnist_ntuple.py -h
usage: mnist_ntuple.py [-h] train_images_file train_labels_file test_images_file test_labels_file

positional arguments:
  train_images_file  file with the training data
  train_labels_file  file with the training labels
  test_images_file   file with the testing data
  test_labels_file   file with the testing labels

optional arguments:
  -h, --help         show this help message and exit
```

Here's an example run:
```
$ python3 mnist_ntuple.py data/train-images-idx3-ubyte data/train-labels-idx1-ubyte data/t10k-images-idx3-ubyte data/t10k-labels-idx1-ubyte 
Confusion Matrix:
               0       1       2       3       4       5       6       7       8       9
       0     920       0       6       3       0      41       6       0       3       1
       1       0    1080       4       7       1      14       2       1      24       2
       2       7       3     894      28      10      15      10      12      46       7
       3       0       0       5     921       1      47       0       8      15      13
       4       1       0       7       0     844       6       4       1      13     106
       5       2       2       1      31       4     822       3       2      10      15
       6       8       2      10       0      19      63     845       3       7       1
       7       0       2      22       5       2       1       0     916      19      61
       8       0       0       5      30       5      34       0       3     873      24
       9       4       2       4      15      12       5       1       8      13     945
accuracy: 0.9060
```
