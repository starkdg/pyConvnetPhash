#! /usr/bin/env python3

import os
import os.path
import warnings
from skimage import img_as_ubyte
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf

images_directory = "/ext/mirflickr25k"
tfrecords_subdir = "tfrecords"

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def process_image_dir(img_dir):
    count = 0
    file_limit = 1000
    tfrecords_filename = "mirflicker1m_images_{0}.tfrecord"
    tfrecords_path = os.path.join(images_directory, tfrecords_subdir, tfrecords_filename.format(count))
    count = count + 1
    tfrecords_writer = tf.python_io.TFRecordWriter(tfrecords_path)
    print("tfrecords file: ", tfrecords_path)

    no_files = 0
    for entry in os.scandir(img_dir):
        if entry.name.startswith('.') or entry.name.startswith('..'):
            continue
        elif entry.is_file() and entry.name.endswith('.jpg'):
            try:
                img = imread(entry.path)
                img = img_as_ubyte(img)
                if (img.ndim >= 3) and (img.shape[2] >= 3):
                    height = img.shape[0]
                    width = img.shape[1]
                    img_raw = img.tobytes()
                    example = tf.train.Example(
                        features=tf.train.Features(feature={
                            'height': int64_feature(height),
                            'width': int64_feature(width),
                            'image_raw': bytes_feature(img_raw)}))
                    tfrecords_writer.write(example.SerializeToString())
                    no_files = no_files + 1
            except Exception as ex:
                print("skipping: {0} {1}".format(entry.path, str(ex)))

        if (no_files % file_limit == 0):
            try:
                tfrecords_writer.close()
                tfrecords_path = os.path.join(images_directory, tfrecords_subdir, tfrecords_filename.format(count))
                count = count + 1
                tfrecords_writer = tf.python_io.TFRecordWriter(tfrecords_path)
                print("tfrecords_file: ", tfrecords_path)
            except Exception as ex:
                print("Unable to write file: {0} {1}".format(tfrecords_path, str(ex)))
                break

    tfrecords_writer.close()


print("Process images in: ", images_directory)
process_image_dir(images_directory)
print("Done.")
