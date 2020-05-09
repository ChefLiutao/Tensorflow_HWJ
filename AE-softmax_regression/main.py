# -*- coding: utf-8 -*-
"""
Created on Sat May  9 21:16:51 2020

@author: Chef_LT
"""
from tensorflow.examples.tutorials.mnist import input_data
from Vanilla_AE import Vanilla_AE
from softmax_regression import Softmax_regression

#load data
mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)
#print(mnist.train.images.shape)

#parameters
n_input = mnist.train.images.shape[1]
n_hidden = 300
lr_ae = 0.01
lr = 0.08                #lr of softmax regression
epochs_ae = 10
epochs = 10             #epoch of softmax regression
batch_size = 100
steps_per_epoch = mnist.train.images.shape[0] // batch_size

#Features extraction via AE
ae = Vanilla_AE(n_input,n_hidden,lr_ae)
for epoch in range(epochs_ae):
    for step in range(steps_per_epoch):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        ae.batch_train(batch_x)
ex_train_images = ae.feature_extract(mnist.train.images)

#Training of softmax regression
softmax_reg = Softmax_regression(n_hidden,10,lr)
for epoch in range(epochs):
    for step in range(steps_per_epoch):
        batch_x = ex_train_images[(step*batch_size):((step+1)*batch_size-1)]
        batch_y = mnist.train.labels[(step*batch_size):((step+1)*batch_size-1)]
        softmax_reg.batch_train(batch_x,batch_y)

#Testing
ex_test_images = ae.feature_extract(mnist.test.images)
accuracy = softmax_reg.evaluation(ex_test_images,mnist.test.labels)
print('Testing accuracy',accuracy)
