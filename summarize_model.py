#! /usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow.saved_model import tag_constants


frozen_model = "/home/david/Downloads/tfmodels/caenet/mobilenetv2_cae_autoenc_1792to256_frozen_model-relu3.pb"
print("model: ", frozen_model)

with tf.gfile.GFile(frozen_model, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as g:
    tf.import_graph_def(graph_def, name="")

    print("operations:")
    for op in g.get_operations():
        print("{0} {1}".format(op.name, op.type))
print("Done.")
