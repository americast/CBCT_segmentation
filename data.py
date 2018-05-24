# -*- coding: utf-8 -*-
# @Author: Saurabh Agarwal
# @Date:   2018-05-24 10:39:32
# @Last Modified by:   Saurabh Agarwal
# @Last Modified time: 2018-05-24 15:27:48
from skimage import io
import os
import glob
import numpy as np

XPath = os.path.join(".","CBCT/Data/X")
YPath = os.path.join(".","CBCT/Data/Y")

XFileName = glob.glob(os.path.join(XPath,"*.tiff"))
YFileName = glob.glob(os.path.join(YPath,"*.tiff"))

x = []
y = []
for file in XFileName[:10]:
	x.append(io.imread(file))

x = np.asarray(x)
print(x.shape)

for file in YFileName[:10]:
	y.append(io.imread(file))

y = np.asarray(y)

print(x.shape, y.shape)