# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:11:51 2020

@author: ADubey4
"""
import os
import shutil
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn
from torch.autograd import Variable 
import torch.onnx
import onnx

x1 =  np.random.random(1000)
x2 = np.random.random(1000)
y = 2*x1 + 5*x2

data = pd.DataFrame(np.array(list(zip(x1, x2, y))), columns=["x1", "x2", "y"])
normed_train_data, normed_test_data, train_labels, test_labels = train_test_split(data[["x1","x2"]], data[["y"]], test_size = 0.2, random_state = 0)

x_train = torch.torch.FloatTensor(normed_train_data.values)
y_train = torch.torch.FloatTensor(train_labels.values)

class LinearRegression(nn.Module):
    def __init__(self, in_size=2, out_size=1):
        super().__init__()
        self.lin = nn.Linear(in_features = in_size, out_features = out_size)
    def forward(self, X):
        pred = self.lin(X)
        return(pred)

input_dim = normed_train_data.shape[1]
output_dim = train_labels.shape[1] 

model = LinearRegression(input_dim,output_dim)
criterion = nn.MSELoss()# Mean Squared Loss
l_rate = 0.01
optimiser = torch.optim.SGD(model.parameters(), lr = l_rate) #Stochastic Gradient Descent
epochs = 5000
total_loss = 0
for epoch in range(epochs):
    inputs = x_train
    labels = y_train
    optimiser.zero_grad()
    outputs = model.forward(inputs)
    loss = criterion(outputs, labels)
    loss.backward()# back props
    optimiser.step()# update the parameters
    print('epoch {}, loss {}'.format(epoch,loss.data))

predicted = model.forward(torch.torch.FloatTensor(normed_test_data.values)).data.numpy()
print(model.state_dict())

if os.path.exists("./model"):
    os.system('rmdir /S /Q "{}"'.format('./model'))
os.mkdir("./model")

dummy_input = Variable(torch.rand(2, device='cpu'))
torch.onnx.export(model, dummy_input, os.path.join("./model", "model.onnx"), verbose=True)
# torch.onnx.export(model, dummy_input, "model.onnx", input_names["input_1"], output_names=["output_1"], verbose=True)
# model = onnx.load("model.onnx")
# onnx.checker.check_model(model)
# onnx.helper.printable_graph(model.graph)

input_shape_str = str([1,normed_train_data.shape[1]]).replace(' ','')
output_dir = "./model"

cmd = 'python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\mo_onnx.py" \
      --input_model "./model/model.onnx" --output_dir '+output_dir+' --input_shape '+input_shape_str+' --data_type FP32 --log_level DEBUG'
# import os
os.system(cmd)