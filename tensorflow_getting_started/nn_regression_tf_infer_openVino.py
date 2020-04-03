# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 02:26:35 2020
@author: ADubey4

Note:       If you have not set the paths permanetely in environment.
            Then run this file on command promopt, preceded by below two commands
code Line:  cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
code Line:  setupvars.bat
            then without closing the same command prompt, run this file
code Line:  python nn_regression_tf_infer_openVino.py
"""

import os
import sys
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin, IECore

plugin_dir = None
model_xml = r'C:\Users\Adubey4\Desktop\Agora\tensorflow\model\frozen_model.xml'
model_bin = r'C:\Users\Adubey4\Desktop\Agora\tensorflow\model\frozen_model.bin'

# plugin = IEPlugin("CPU", plugin_dirs=plugin_dir)
ie = IECore()
# versions = ie.get_versions("CPU")
# Read IR
net = IENetwork(model=model_xml, weights=model_bin)
# check net.inputs.keys(), net.outputs
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
# exec_net = plugin.load(network=net)
exec_net = ie.load_network(network=net, device_name="CPU")
del net

# Run inference
x = np.random.random((1,2))
res = exec_net.infer(inputs={input_blob: x})
print(x)
print("Actual: ",2*x[0][0] + 5*x[0][1])
print("Predicted: ",res)