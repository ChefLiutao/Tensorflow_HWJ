# -*- coding: utf-8 -*-
"""
Created on Fri May 15 19:30:54 2020

@author: ChefLiutao
"""
import tensorflow as tf

class MLP():
    '''
    '''
    def __init__(self,n_input,n_hidden,n_output,learning_rate,dropout = 0.7):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lr = learning_rate
        self.train_keep_prob = 1 - dropout
        self.keep_prob = tf.placeholder(dtype = tf.float32)
        self.X = tf.placeholder(shape = [None,self.n_input],
                                dtype = tf.float32)
        self.y = tf.placeholder(shape = [None,self.n_output],
                                dtype = tf.float32)
        self.W1 = tf.Variable(tf.truncated_normal([self.n_input,self.n_hidden],
                                                  stddev = 0.1))
        self.b1 = tf.Variable(tf.zeros(shape = [1,self.n_hidden]))
        self.W2 = tf.Variable(tf.zeros(shape = [self.n_hidden,self.n_output]))
        self.b2 = tf.Variable(tf.zeros(shape = [1,self.n_output]))
        
        self.hidden = tf.nn.relu(tf.add(tf.matmul(self.X,self.W1),self.b1))
        self.hidden_drop = tf.nn.dropout(self.hidden,self.keep_prob)
        self.logits = tf.add(tf.matmul(self.hidden_drop,self.W2),self.b2)
        self.output = tf.nn.softmax(self.logits)
        self.pred = tf.argmax(self.output,axis = 1)
                
        self.loss = tf.reduce_mean(-tf.reduce_sum(tf.multiply(
                self.y,tf.log(self.output + 1e-10)),axis = 1))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        self.accu = tf.reduce_mean(tf.cast(tf.equal(
                self.pred,tf.argmax(self.y,axis = 1)),tf.float32))
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def batch_train(self,input_X,label_y):
        train_op,loss = self.sess.run((self.train_op,self.loss),
                                      feed_dict = {self.X:input_X,
                                        self.y:label_y,
                                        self.keep_prob:self.train_keep_prob})
        return loss
    
    def predict(self,input_X):
        pred = self.sess.run(self.pred,
                             feed_dict = {self.X:input_X,self.keep_prob:1.0})
        return pred
    
    def evaluate(self,input_X,label_y):
        accuracy = self.sess.run(self.accu,feed_dict = {self.X:input_X,
                                                        self.y:label_y,
                                                        self.keep_prob:1.0})
        return accuracy
        
