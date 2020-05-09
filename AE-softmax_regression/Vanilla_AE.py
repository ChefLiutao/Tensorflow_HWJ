# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:48:03 2020

@author: Chef_LT
"""

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data

class Vanilla_AE():
    '''
    '''
    def __init__(self,n_input,n_hidden,learning_rate):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.X = tf.placeholder(dtype = tf.float32,shape = [None,n_input])
        self.hidden = fully_connected(self.X,self.n_hidden,activation_fn = None)
        self.output = fully_connected(self.hidden,self.n_input,activation_fn = None)
        self.loss = tf.reduce_mean(tf.square(self.X - self.output))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def batch_train(self,batch):
        self.sess.run(self.train_op,feed_dict = {self.X:batch})
    
    def feature_extract(self,data):
        output = self.sess.run(self.hidden,feed_dict = {self.X:data})
        return output

if __name__ == '__main__':
    #载入数据
    mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)
    #参数
    n_input = mnist.train.images.shape[1]
    n_hidden = 150
    epochs = 10
    batch_size = 100
    steps_per_epoch = mnist.train.images.shape[0] // batch_size
    #实例化
    vanilla_ae = Vanilla_AE(n_input,n_hidden,0.01)
    #训练
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            vanilla_ae.batch_train(batch_x)
    #feature extraction
    extracted_data = vanilla_ae.feature_extract(mnist.train.images)
