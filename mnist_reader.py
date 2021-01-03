"""
MNISTReader: Utility for reading MNIST data
Author: Phil Dreizen

The MNIST database of handwritten digits is available here: http://yann.lecun.com/exdb/mnist/

"The data is stored in a very simple file format designed for storing vectors and multidimensional matrices" which is described on the webpage.

There are two kinds of files:
    - image data: containing pixel information for each handwritten image
    - label data: letting us know what number the image represents [0-9]

TODO: 
    * improved error handling of the data files; right now we're assuming we are working with proper MNIST data files
"""

import numpy as np
from utils import eprint

class MNISTReader:
    """
    Utility for reading MNIST image and training data files
    """
    
    def read_img(self, fp):
        """
        read image data file

        The format looks like this:
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000803(2051) magic number
        0004     32 bit integer  10000            number of images
        0008     32 bit integer  28               number of rows
        0012     32 bit integer  28               number of columns
        0016     unsigned byte   ??               pixel
        0017     unsigned byte   ??               pixel
        ........
        xxxx     unsigned byte   ??               pixel

        * All the integers in the files are stored in the MSB first (high endian) format
        * Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
        """
        fh = open(fp,'rb')

        #we don't really need the magic number, iff we trust the data is in the right format, which I am
        magic_num = bytes(fh.read(4)) 

        #we read in the number of images, and the dimensions of the images
        nimg = int.from_bytes(fh.read(4), byteorder='big')
        rows = int.from_bytes(fh.read(4), byteorder='big')
        cols = int.from_bytes(fh.read(4), byteorder='big')
        #eprint('nimg', nimg, 'rows', rows, 'cols', cols)

        #read in the image data and return
        arr1d = np.fromfile(fh, dtype=np.uint8)
        images = np.reshape(arr1d,(nimg,rows*cols))
        fh.close()
        return images 

    def read_labels(self, fp):
        """
        read label data file

        The format looks like this:
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000801(2049) magic number (MSB first)
        0004     32 bit integer  10000            number of items
        0008     unsigned byte   ??               label
        0009     unsigned byte   ??               label
        ........
        xxxx     unsigned byte   ??               label
        The labels values are 0 to 9.

        We simply use numpy to read in the data, skipping the MSB and the number of items (trusting it will match the image data)
        """
        labels = np.fromfile(fp, dtype=np.uint8, offset=8) #read in the labels, skipping the first 8 bytes
        return labels 

