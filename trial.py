# -*- coding: utf-8 -*-
# @Author: Saurabh Agarwal
# @Date:   2018-05-14 03:17:00
# @Last Modified by:   Saurabh Agarwal
# @Last Modified time: 2018-05-14 04:10:40
import tensorflow as tf
import numpy as np

minValue = np.array([4],dtype=np.float32)
maxValue = np.array([5],dtype=np.float32)

# Correct Range = (3,6)
# Output Function = (input-low)*(high-input)
# For values in between given range
inputValue = [[1,2,3],[4,5,6],[7,8,9]]
outputValue = [[0,0,0],[2,2,0],[0,0,0]]

x = tf.placeholder(tf.float32, shape=[3,3])
y = tf.placeholder(tf.float32,shape=[3,3])


zeros = tf.zeros(shape=[3,3])
ones = tf.ones(shape=[3,3])

lowThresh = tf.Variable(minValue)
highThresh = tf.Variable(maxValue)

firstLayer = tf.maximum(x-lowThresh,zeros)
secondLayer = tf.maximum(highThresh-x,zeros)

# Need to convert it into binary form
predict = tf.multiply(firstLayer, secondLayer)


cost = tf.reduce_mean(tf.squared_difference(predict,y))
learningrate = 0.01
train = tf.train.AdamOptimizer(learningrate).minimize(cost)

init = tf.global_variables_initializer()
numepochs = 1000

with tf.Session() as sess:
	sess.run(init)
	for i in xrange(numepochs):
		_ = sess.run([train], feed_dict={
			x: inputValue,
			y: outputValue
			})

	lowValue, highValue, pred = sess.run([lowThresh,highThresh, predict], feed_dict={
		x:inputValue,
		y:outputValue
		})
	print "Range -> ({},{})".format(lowValue[0], highValue[0])
	print "Predicted Output: {}".format(pred)

