'''
A linear regression learning algorithm example using pytorch library.
Author: Terrance DeVries
Project: https://github.com/TDeVries/Deep-Learning-Rosetta-Stone
'''

from __future__ import print_function

import numpy as np
from sklearn.datasets import make_regression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

##################
### Parameters ###
##################
learning_rate = 0.01
nb_epoch = 10

######################
### Create dataset ###
######################
train_X, train_Y = make_regression(n_features=1, noise=5.0, random_state=0)
train_Y = np.reshape(train_Y, (100, 1))
train_X, train_Y = torch.FloatTensor(train_X), torch.FloatTensor(train_Y)

#######################
### Construct model ### 
#######################
class LinearRegression(nn.Module):
	def __init__(self):
		super(LinearRegression, self).__init__()
		self.w = nn.Linear(1,1)

	def forward(self, x):
		x = self.w(x)
		return x

linear_regression = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(linear_regression.parameters(), lr = learning_rate)

###################
### Train model ###
###################
for epoch in range(nb_epoch):
	for (inputs, labels) in zip(train_X, train_Y):

		# Wrap tensors so that they can be differentiated
		inputs, labels = Variable(inputs), Variable(labels)
		# Add a batch dimension (Pytorch only does minibatches)
		inputs = inputs.unsqueeze(0)

		# Zero the parameter gradients
		optimizer.zero_grad()

		# Forward pass through network
		outputs = linear_regression(inputs)

		# Calculate the loss
		loss = criterion(outputs, labels)

		# Backpropogate the loss through the network 
		loss.backward()

		# Update the network weights
		optimizer.step()

	#Calculate the loss over the entire training set
	outputs = linear_regression(Variable(train_X))
	loss = criterion(outputs, Variable(train_Y))
	loss = loss.data[0]

	print("Epoch:", '%d' % (epoch+1), "cost =", "{:.4f}".format(loss))

weight, bias =  linear_regression.w.parameters()
weight = weight.data[0,0]
bias = bias.data[0]

print("Optimization Finished!")
print("W=", weight, "b=", bias)


