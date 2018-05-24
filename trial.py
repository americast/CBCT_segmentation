# -*- coding: utf-8 -*-
# @Author: Saurabh Agarwal
# @Date:   2018-05-14 03:17:00
# @Last Modified by:   Saurabh Agarwal
# @Last Modified time: 2018-05-24 10:32:51
import tensorflow as tf
import numpy as np
import unet
from PIL import Image
import numpy as np

minValue = np.array([1.0000],dtype=np.float32)
maxValue = np.array([9.0],dtype=np.float32)

# Correct Range = (3,6)
# Output Function = (input-low)*(high-input)
# For values in between given range
inputValue = [[1,2,3],[4,5,6],[7,8,9]]
outputValue = [[4,4,4],[4,5,5],[5,5,5]]

x = tf.placeholder(tf.float32, shape=[3,3])
y = tf.placeholder(tf.float32,shape=[3,3])


zeros = tf.zeros(shape=[3,3])
ones = tf.ones(shape=[3,3])

lowThresh = tf.Variable(minValue)
highThresh = tf.Variable(maxValue)


firstLayer = tf.maximum(x,lowThresh)
secondLayer = tf.minimum(firstLayer, highThresh)
# firstLayer = tf.maximum(x-lowThresh,zeros)
# secondLayer = tf.maximum(highThresh-x,zeros)

# Need to convert it into binary form
# predict = tf.multiply(firstLayer, secondLayer)
training = tf.placeholder(tf.bool)
predict = unet.make_unet(x, training)

cost = tf.reduce_mean(tf.squared_difference(predict,y))
learningrate = 0.01
train = tf.train.AdamOptimizer(learningrate).minimize(cost)

init = tf.global_variables_initializer()
numepochs = 1000

with tf.Session() as sess:
	sess.run(init)
	for i in xrange(numepochs):
		_, lowValue, highValue, c = sess.run([train, lowThresh, highThresh, cost], feed_dict={
			x: inputValue,
			y: outputValue,
			training: True
			})
		print(lowValue[0], highValue[0],c)

	lowValue, highValue, pred, c = sess.run([lowThresh,highThresh, predict, cost], feed_dict={
		x:inputValue,
		y:outputValue,
		training: True
		})
	print "Range -> ({},{})".format(lowValue[0], highValue[0])
	print "Predicted Output: {}".format(pred)
	print "Cost: {}".format(c)

