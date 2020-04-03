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
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
tf.keras.backend.set_floatx('float32')

x1 =  np.random.random(1000)
x2 = np.random.random(1000)
y = 2*x1 + 5*x2

data = pd.DataFrame(np.array(list(zip(x1, x2, y))), columns=["x1", "x2", "y"])
normed_train_data, normed_test_data, train_labels, test_labels = train_test_split(data[["x1","x2"]], data[["y"]], test_size = 0.2, random_state = 0)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(normed_train_data.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

# example_batch = normed_train_data[:10]
# example_result = model.predict(example_batch)
# example_result

EPOCHS = 1000
history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0)

test_predictions = model.predict(normed_test_data).flatten()
out = list(zip(test_labels.values, test_predictions))
if os.path.exists("./model"):
    os.system('rmdir /S /Q "{}"'.format('./model'))
os.mkdir("./model")
model.save("./model/regression_tf.h5")

from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model

save_pb_dir = './model'
model_fname = './model/regression_tf.h5'
def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0) 

model = load_model(model_fname)
session = tf.keras.backend.get_session()

INPUT_NODE = [t.op.name for t in model.inputs]
OUTPUT_NODE = [t.op.name for t in model.outputs]
print(INPUT_NODE, OUTPUT_NODE)
frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)

input_shape_str = str([1,normed_train_data.shape[1]]).replace(' ','')
output_dir = "./model"

cmd = 'python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\mo_tf.py"\
     --input_model "./model/frozen_model.pb" --output_dir '+output_dir+' --input_shape '+input_shape_str+' --data_type FP32 --log_level DEBUG'
# import os
os.system(cmd)