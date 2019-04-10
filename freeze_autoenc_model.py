#! /usr/bin/env python3
import os
import tensorflow as tf
from tensorflow.saved_model import tag_constants

autoenc_model_dir = "/home/david/Downloads/tfmodels/deepaec/"
frozen_model = os.path.join(autoenc_model_dir, "autoenc_frozen_model.pb")

print("load model: ", autoenc_model_dir)
with tf.Graph().as_default() as graph:
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], autoenc_model_dir)

        graphdef = graph.as_graph_def()
        print("extract autoenc subgraph")
        subgraphdef = tf.graph_util.extract_sub_graph(graphdef, ["output256"])
        print("convert variables to constants")
        subgraphdef2 = tf.graph_util.convert_variables_to_constants(sess, subgraphdef, ['output256'])

        print("write frozen graph")
        with tf.gfile.GFile(frozen_model, "wb") as f:
            f.write(subgraphdef2.SerializeToString())


print("Done.")
        

