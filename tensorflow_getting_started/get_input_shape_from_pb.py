# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:01:14 2020

@author: ADubey4
"""

import tensorflow as tf
from tensorflow.python.platform import gfile
tf_PB_file_path = r'C:\Users\Adubey4\Desktop\Agora\tensorflow_getting_started\model\frozen_model.pb'
with tf.Session() as sess:
    print("load graph")
    with gfile.FastGFile(tf_PB_file_path,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    print(graph_def.node[0])
    # tf.import_graph_def(graph_def, name='')
    # graph_nodes=[n for n in graph_def.node]
    # names = []
    # for t in graph_nodes:
    #     names.append(t.name)
    #     print(t)
    # print(names)
    # print("Input Layer below")
    # print(graph_nodes[0])
