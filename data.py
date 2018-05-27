# -*- coding: utf-8 -*-
# @Author: Saurabh Agarwal
# @Date:   2018-05-24 10:39:32
# @Last Modified by:   Saurabh Agarwal
# @Last Modified time: 2018-05-24 15:27:48
from skimage import io
import os
import glob
import numpy as np

BATCH_SIZE = 10

def data_in_batches():
	XPath = os.path.join(".","data/X")
	YPath = os.path.join(".","data/Y")

	XFileName = glob.glob(os.path.join(XPath,"*.tiff"))
	YFileName = glob.glob(os.path.join(YPath,"*.tiff"))

	x = []
	y = []
	for file in XFileName:
		# print file
		x.extend(io.imread(file))
		# print len(x)
		# print shape(x)
	x = np.asarray(x)
	x = np.reshape(x, [x.shape[0],1,x.shape[1], x.shape[2]])
	print(x.shape)

	for file in YFileName:
		# print file
		y.extend(io.imread(file))
		# print len(x)
		# print shape(x)
	y = np.asarray(y)
	y = np.reshape(y, [y.shape[0],1,y.shape[1], y.shape[2]])
	print(y.shape)

	indices = np.random.permutation(x.shape[0])
	print(indices)

	x = x[indices]
	y = y[indices]

	count = 0
	for i in x.shape[0]/BATCH_SIZE:
		yield (x[count:count+BATCH_SIZE], y[count:count+BATCH_SIZE])
		count+=BATCH_SIZE