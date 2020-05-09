# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:20:51 2020

@author: Chef_LT
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Softmax_regression():
    '''
    '''
    def __init__(self,learning_rate):
        self.lr = learning_rate
        self.W = tf.Variable(tf.zeros([784,10]),dtype = tf.float32)
        self.b = tf.Variable(tf.zeros([1,10]),dtype = tf.float32)
        self.X = tf.placeholder(dtype = tf.float32,shape = [None,784])
        self.y = tf.placeholder(dtype = tf.float32,shape = [None,10])
        self.sess = tf.Session()
        self.logits = tf.add(tf.matmul(self.X,self.W),self.b)
        self.y_ = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(tf.reduce_sum(-tf.multiply(tf.log(self.y_),self.y),axis = 1))
        self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
        self.pred = tf.equal(tf.argmax(self.y_,axis = 1),tf.argmax(self.y,axis = 1))
        self.accu = tf.reduce_mean(tf.cast(self.pred,tf.float32))
        
        self.sess.run(tf.global_variables_initializer())
    
    def train(self,input,label):
        self.sess.run(self.train_op,feed_dict = {self.X:input,self.y:label})
    
    def evaluation(self,input,label):
        return self.sess.run(self.accu,feed_dict = {self.X:input,self.y:label})


if __name__ == '__main__':
    #加载mnist数据
    mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)
    #实例化
    softmax_regression = Softmax_regression(0.1)
    #训练
    for step in range(5000):
        batch_x,batch_y = mnist.train.next_batch(100)
        softmax_regression.train(batch_x,batch_y)
    #测试
    accu = softmax_regression.evaluation(mnist.test.images,mnist.test.labels)
    print('Testing acuracy:',accu)
    
    