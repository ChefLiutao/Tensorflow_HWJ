# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:42:05 2020

@author: ChefLiutao
"""

from MLP import MLP
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)

n_input,n_output = mnist.train.images.shape[1],mnist.train.labels.shape[1]
n_hidden = 300
batch_size = 64
learning_rate = 0.001
mlp = MLP(n_input,n_hidden,n_output,learning_rate)

for epoch in range(10):
    for step in range(mnist.train.images.shape[0] // batch_size):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        loss = mlp.batch_train(batch_x,batch_y)
    print('Epoch %d:'%epoch,loss)

print(mlp.evaluate(mnist.train.images,mnist.train.labels))
print(mlp.evaluate(mnist.test.images,mnist.test.labels))
print(mlp.evaluate(mnist.validation.images,mnist.validation.labels))