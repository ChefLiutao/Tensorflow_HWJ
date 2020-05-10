# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:20:51 2020

@author: Chef_LT
"""
import tensorflow as tf

class Softmax_regression():
    '''
    '''
    def __init__(self,n_input,n_output,learning_rate):
        self.lr = learning_rate
        self.W = tf.Variable(tf.zeros([n_input,n_output]),dtype = tf.float32)
        self.b = tf.Variable(tf.zeros([1,n_output]),dtype = tf.float32)
        self.X = tf.placeholder(dtype = tf.float32,shape = [None,n_input])
        self.y = tf.placeholder(dtype = tf.float32,shape = [None,n_output])
        self.sess = tf.Session()
        self.logits = tf.add(tf.matmul(self.X,self.W),self.b)
        self.y_ = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(tf.reduce_sum(-tf.multiply(tf.log(self.y_),self.y),axis = 1))
        self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
        self.pred = tf.equal(tf.argmax(self.y_,axis = 1),tf.argmax(self.y,axis = 1))
        self.accu = tf.reduce_mean(tf.cast(self.pred,tf.float32))
        
        self.sess.run(tf.global_variables_initializer())
    
    def batch_train(self,input,label):
        self.sess.run(self.train_op,feed_dict = {self.X:input,self.y:label})
    
    def evaluation(self,input,label):
        return self.sess.run(self.accu,feed_dict = {self.X:input,self.y:label})

