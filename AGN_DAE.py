# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:42:29 2020

@author: ChefLiutao
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def xavier_initializer(n_input,n_output):
    '''
    Xavier initializer
    '''
    low = -tf.sqrt(6/(n_input + n_output))
    high = tf.sqrt(6/(n_input + n_output))
    return tf.random_uniform(shape = [n_input,n_output],
                             minval = low,
                             maxval = high,
                             dtype = tf.float32)

class Add_gauss_noise_DAE():
    def __init__(self,n_input,n_hidden,learning_rate,scale = 0.1):
        '''
        Args:
            n_input:输入节点数
            n_hidden:隐层节点数
            learning_rate:DAE学习率
            scale:高斯噪声系数
        '''
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.scale = scale
        self.X = tf.placeholder(shape = [None,self.n_input],dtype = tf.float32)
        self.W1 = tf.Variable(xavier_initializer(self.n_input,
                                                 self.n_hidden))
        self.b1 = tf.Variable(tf.zeros(shape = [1,self.n_hidden],
                                       dtype = tf.float32))
        self.W2 = tf.Variable(tf.zeros(shape = [self.n_hidden,self.n_input],
                                       dtype = tf.float32))
        self.b2 = tf.Variable(tf.zeros(shape = [1,self.n_input],
                                       dtype = tf.float32))
        self.hidden = tf.nn.softplus(tf.matmul(tf.add(
                self.X,self.scale*tf.random_normal(shape = [self.n_input,])),
                                self.W1) + self.b1)
        self.output = tf.matmul(self.hidden,self.W2) + self.b2
        self.loss = tf.reduce_mean(tf.square(self.X - self.output))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def batch_train(self,input_batch):
        opt,loss =  self.sess.run((self.train_op,self.loss),
                                  feed_dict = {self.X:input_batch})
        return loss
    
    def encoder(self,input_data):
        '''
        Obtain the codings in hidden layer
        '''
        return self.sess.run(self.hidden,feed_dict = {self.X:input_data})
    
    def decoder(self,hidden_data):
        '''
        Obtain reconstracted data from hidden data
        '''
        return self.sess.run(self.output,
                             feed_dict = {self.hidden:hidden_data})
    
    def reconstruct(self,input_data):
        '''
        Obtain reconstracted data from input data
        '''
        return self.sess.run(self.output,
                             feed_dict = {self.X:input_data})
    
    def get_weights(self):
        return self.W1
    
    def get_bias(self):
        return self.b1
    
if __name__ == '__main__':
    mnist = tf.examples.tutorials.mnist.input_data.read_data_sets('MNIST_data/',one_hot = True)
    dae = Add_gauss_noise_DAE(784,300,0.005,0.1)

    BATCH_SIZE = 128
    EPOCHS = 10
    STEPS_PER_EPOCH = mnist.train.images.shape[0] // batch_size

    for epoch in range(EPOCHS):
        for step in range(STEPS_PER_EPOCH):
            batch_x,batch_y = mnist.train.next_batch(BATCH_SIZE)
            loss = dae.batch_train(batch_x)
        print('Epoch %d:'%epoch,loss)
    
    origin_image = mnist.test.images[0].reshape([28,28])
    noise_image = (mnist.test.images[0] + 0.1*np.random.normal(
            loc = 0,scale = 1,size = [784,])).reshape([28,28])
    reconstracted_image = dae.reconstruct(origin_image.reshape([1,784])).reshape(28,28)
    
    plt.imshow(origin_image)
    plt.imshow(noise_image)
    plt.imshow(reconstracted_image)
