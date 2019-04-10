import os
import numpy as np
import tensorflow as tf


class ConvPhashAutoEnc:
    """ Convolutional Perceptual Hash
    """

    def __init__(self, model_file):
        """ init method
        args
        model_file  -- file for inception/autoenc model
        """
        with tf.gfile.GFile(model_file, 'rb') as f:
            self.classif_graph_def = tf.GraphDef()
            self.classif_graph_def.ParseFromString(f.read())

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(self.classif_graph_def, name="aec")

        self.file_in = self.graph.get_tensor_by_name('aec/preprocess/input:0')
        self.img_out = self.graph.get_tensor_by_name('aec/preprocess/output:0')
        self.in_tensor = self.graph.get_tensor_by_name('aec/classifier/input:0')
        self.feature_tensor = self.graph.get_tensor_by_name('aec/classifier/feature_vector:0')
        # self.out    = self.graph.get_tensor_by_name('aec/aec_encoder/output:0')
        # self.out1024 = self.graph.get_tensor_by_name('aec/aec_encoder/output1024:0')
        # self.out512 = self.graph.get_tensor_by_name('aec/aec_encoder/output512:0')
        self.out256 = self.graph.get_tensor_by_name('aec/aec_encoder/output256:0')
        # self.out128 = self.graph.get_tensor_by_name('aec/aec_encoder/output128:0')
        # self.out64 = self.graph.get_tensor_by_name('aec/aec_encoder/output64:0')
        # self.out32 = self.graph.get_tensor_by_name('aec/aec_encoder/output32:0')
        self.sess = tf.Session(graph=self.graph)

    def process_image_files(self, img_dir, limit=50):
        """ process images from files in a directory
        args
        img_dir -- directory of jpeg images
        return     generator of ndarray's (shape: limit x h x w x 3)
        """
        arrlist = []
        count = 0
        for entry in os.scandir(img_dir):
            if entry.is_file() and entry.name.endswith('.jpg'):
                img_data = self.sess.run(self.img_out,
                                         feed_dict={self.file_in: entry.path})
                arrlist.append(img_data)
                count = count + 1
            if count >= limit:
                count = 0
                yield np.concatenate(arrlist)
                arrlist.clear()
        if count > 0:
            yield np.concatenate(arrlist)

    def image_features(self, img_dir, limit=50):
        """ Get feature vectors for images from module
        args
        images -- ndarray of images obtained from process_image_files()
                  size: [no_images x 299 x 299 x 3]
        return
        features -- ndarray (size no_images x feature_length)
        """

        features_list = []
        images_gen = self.process_image_files(img_dir, limit=limit)
        with self.graph.as_default():
            for np_images in images_gen:
                features = self.sess.run(self.feature_tensor,
                                         feed_dict={self.in_tensor: np_images})
                features_list.append(features)

        concatfeatures = np.concatenate(features_list, axis=0)
        return concatfeatures

    '''
    def image_condensed1024(self, img_dir, limit=50):
        """ Compute condensed image perceptual hashes from features.
        args
        features -- ndarray of features (size: [no_images x 1 x 1 x 2048])
        return -- ndarray of final hashes (size: no_images x 256)
        """
        features_list = []
        images_gen = self.process_image_files(img_dir, limit=limit)
        with self.graph.as_default():
            for np_images in images_gen:
                features = self.sess.run(self.out1024,
                                         feed_dict={self.in_tensor: np_images})
                features_list.append(features)

        concatfeatures = np.concatenate(features_list, axis=0)
        return concatfeatures

    def image_condensed512(self, img_dir, limit=50):
        """ Compute condensed image perceptual hashes from features.
        args
        features -- ndarray of features (size: [no_images x 1 x 1 x 2048])
        return -- ndarray of final hashes (size: no_images x 256)
        """
        features_list = []
        images_gen = self.process_image_files(img_dir, limit=limit)
        with self.graph.as_default():
            for np_images in images_gen:
                features = self.sess.run(self.out512,
                                         feed_dict={self.in_tensor: np_images})
                features_list.append(features)

        concatfeatures = np.concatenate(features_list, axis=0)
        return concatfeatures
    '''
    def image_condensed256(self, img_dir, limit=50):
        """ Compute condensed image perceptual hashes from features.
        args
        features -- ndarray of features (size: [no_images x 1 x 1 x 2048])
        return -- ndarray of final hashes (size: no_images x 256)
        """
        features_list = []
        images_gen = self.process_image_files(img_dir, limit=limit)
        with self.graph.as_default():
            for np_images in images_gen:
                features = self.sess.run(self.out256,
                                         feed_dict={self.in_tensor: np_images})
                features_list.append(features)

        concatfeatures = np.concatenate(features_list, axis=0)
        return concatfeatures

    '''
    def image_condensed128(self, img_dir, limit=50):
        """ Compute condensed image perceptual hashes from features.
        args
        features -- ndarray of features (size: [no_images x 1 x 1 x 2048])
        return -- ndarray of final hashes (size: no_images x 256)
        """
        features_list = []
        images_gen = self.process_image_files(img_dir, limit=limit)
        with self.graph.as_default():
            for np_images in images_gen:
                features = self.sess.run(self.out128,
                                         feed_dict={self.in_tensor: np_images})
                features_list.append(features)

        concatfeatures = np.concatenate(features_list, axis=0)
        return concatfeatures

    def image_condensed64(self, img_dir, limit=50):
        """ Compute condensed image perceptual hashes from features.
        args
        features -- ndarray of features (size: [no_images x 1 x 1 x 2048])
        return -- ndarray of final hashes (size: no_images x 256)
        """
        features_list = []
        images_gen = self.process_image_files(img_dir, limit=limit)
        with self.graph.as_default():
            for np_images in images_gen:
                features = self.sess.run(self.out64, feed_dict={self.in_tensor: np_images})
                features_list.append(features)

        concatfeatures = np.concatenate(features_list, axis=0)
        return concatfeatures

    def image_condensed32(self, img_dir, limit=50):
        """ Compute condensed image perceptual hashes from features.
        args
        features -- ndarray of features (size: [no_images x 1 x 1 x 2048])
        return -- ndarray of final hashes (size: no_images x 256)
        """
        features_list = []
        images_gen = self.process_image_files(img_dir, limit=limit)
        with self.graph.as_default():
            for np_images in images_gen:
                features = self.sess.run(self.out32, feed_dict={self.in_tensor: np_images})
                features_list.append(features)

        concatfeatures = np.concatenate(features_list, axis=0)
        return concatfeatures
    '''
    def l1_distance(self, x, y, axis=1):
        diff = np.sum(np.abs((x.astype(float) - y.astype(float))),
                      axis=axis, keepdims=False)
        return diff

    def l2_distance(self, x, y, axis=1):
        diff = np.sqrt(np.sum(np.square(x.astype(float)-y.astype(float)),
                              axis=axis, keepdims=False))
        return diff
