# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:48:03 2020

@author: Chef_LT
"""

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

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
        codings = self.sess.run(self.hidden,feed_dict = {self.X:data})
        return codings
    
    def decode_display(self,input):
        '''
        显示重构数据
        '''
        output = self.sess.run(self.output,feed_dict = {self.X:input})
        return output
