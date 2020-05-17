# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:17:25 2020

@author: ChefLiutao
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot = True)


'''
X -> conv1 -> relu -> pooling1 -> conv2 - relu -> pooling2 -> fc1 -> 
relu -> dropout -> fc2 -> softmax -> y_
'''
X = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
X_image = tf.reshape(X,[-1,28,28,1])

# X -> conv1 -> relu -> pooling1
W1_conv = tf.Variable(tf.truncated_normal(shape = [5,5,1,32],stddev = 0.1))
b1_conv = tf.Variable(tf.constant(0.1,shape = [32]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(X_image,W1_conv,strides = [1,1,1,1],
                                  padding = 'SAME') + b1_conv)
h_pooling1 = tf.nn.max_pool(h_conv1,ksize = [1,2,2,1],strides = [1,2,2,1],
                            padding = 'SAME')

# conv2 -> relu -> pooling2
W2_conv = tf.Variable(tf.truncated_normal(shape = [5,5,32,64],stddev = 0.1))
b2_conv = tf.Variable(tf.constant(0.1,shape = [64]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pooling1,W2_conv,strides = [1,1,1,1],
                                  padding = 'SAME') + b2_conv)
h_pooling2 = tf.nn.max_pool(h_conv2,ksize = [1,2,2,1],strides = [1,2,2,1],
                            padding = 'SAME')

# fc1 -> relu -> dropout -> fc2 -> softmax -> y_
W3_fc1 = tf.Variable(tf.truncated_normal(shape = [7*7*64,1000],stddev = 0.1))
b3_fc1 = tf.Variable(tf.constant(0.1,shape = [1000]))
h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(h_pooling2,[-1,7*7*64]),W3_fc1) + b3_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

W4_fc2 = tf.Variable(tf.truncated_normal(shape = [1000,10],stddev = 0.1))
b4_fc2 = tf.Variable(tf.constant(0.1,shape = [10]))
logits = tf.matmul(h_fc1_drop,W4_fc2) + b4_fc2
y_ = tf.nn.softmax(logits)

#loss, optimizer
cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.multiply(y,tf.log(y_)),
                                              axis = 1))
train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#testing
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,axis = 1),
                                       tf.argmax(y_,axis = 1)),tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(10):
        for step in range(55000 // 128):
            batch_x,batch_y = mnist.train.next_batch(128)
            if step % 100 == 0:
                accu = accuracy.eval(feed_dict = {X:batch_x,y:batch_y,keep_prob:1.0})
                print('Accuracy:',accu)
            sess.run(train_op,feed_dict = {X:batch_x,y:batch_y,keep_prob:0.5})
    print('Testing accuracy:',accuracy.eval(feed_dict = {X:mnist.test.images,
                                                         y:mnist.test.labels,
                                                         keep_prob:1.0}))
            



