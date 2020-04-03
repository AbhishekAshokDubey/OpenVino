# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:01:14 2020

@author: ADubey4
"""
import onnx
model = onnx.load(".\model\model.onnx")
onnx.checker.check_model(model)
onnx.helper.printable_graph(model.graph) # to check in python console

input_node = model.graph.input[0]
print(input_node.type.tensor_type)
