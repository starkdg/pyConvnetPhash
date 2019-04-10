#! /usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow.saved_model import tag_constants

image_input_width = 224
image_input_height = 224
image_input_channels = 3
classification_model = "/home/david/Downloads/tfmodels/mobilenetv2/mobilenet_v2_1.4_224_frozen.pb"
aec_model = "/home/david/Downloads/tfmodels/caenet/mobilenetv2_cae_autoenc_1792to256_frozen_model-10.pb"
frozen_model = "/home/david/Downloads/tfmodels/caenet/mobilenetv2_cae_autoenc_1792to256_combined_frozen_model.py"
classif_input = "input:0"
classif_feature = "MobilenetV2/Logits/AvgPool:0"
autoenc_input = "input:0"
autoenc_output = "output256:0"


print("read model: ", classification_model)
with tf.gfile.GFile(classification_model, "rb") as f:
    classif_graph_def = tf.GraphDef()
    classif_graph_def.ParseFromString(f.read())

print("read model: ", aec_model)
with tf.gfile.GFile(aec_model, "rb") as f:
    autoenc_graph_def = tf.GraphDef()
    autoenc_graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as g:
    print("import classification model graphdef")
    input_tensor, feature_tensor = tf.import_graph_def(classif_graph_def,
                        return_elements=[classif_input, classif_feature],
                        name="classifier")

    feature2_tensor = tf.squeeze(feature_tensor,
                                 [1, 2],
                                 name="classifier/feature_vector")

    print("input: ", input_tensor)
    print("feature: ", feature_tensor.shape)
    print("feature: ", feature2_tensor.shape)

    print("import autoenc model graphdef")
    autoenc_input_tensor, autoenc_output_tensor = tf.import_graph_def(autoenc_graph_def,
                                                        input_map={autoenc_input: feature2_tensor},
                                                        return_elements=[autoenc_input, autoenc_output],
                                                        name="aec_encoder")

    print("aec input: ", autoenc_input_tensor.shape)
    print("aec output: ", autoenc_output_tensor.shape)

    with tf.variable_scope("preprocess"):
        image_file = tf.placeholder(tf.string, name="input")
        height = tf.constant(image_input_height, dtype=tf.int32, name="height")
        width = tf.constant(image_input_width, dtype=tf.int32, name="width")
        img_data = tf.read_file(image_file, name="readfile")
        img = tf.image.decode_image(img_data, channels=3, name="decode")
        imgfl = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img_expanded = tf.expand_dims(imgfl, 0)
        img_resized = tf.image.resize_bicubic(img_expanded, (height, width))
        preprocessed_img = tf.identity(img_resized, name="output")

    for op in g.get_operations():
        print("{0} {1}".format(op.name, op.type))

    output_graph_def = g.as_graph_def()
    with tf.gfile.GFile(frozen_model, "wb") as f:
        print("write model: ", frozen_model)
        f.write(output_graph_def.SerializeToString())

print("Done.")
