# pyConvnetPhash

This repository contains experiments related to this blog post:
[On Extracting Descriptors](https://starkdg.github.io/posts/concise-image-descriptor)

The basic idea is to extract a descriptor from images by taking advantage of some convolutional neural
net models trained to do image classification tasks.  These models output a feature vector from one of
the hidden layers. The object here is to see how far we can reduce the dimensionality of the feature
space by training additional autoencoder layers that might functionaly piggy-back on top of these models. 

Here's a map to the files. Locations for files are located at the top of the scripts which will need
to be changed.

## Utility Scripts

- process_images_into_tfrecords.py - pack image files, jpegs into tensorflow's .tfrecord format (the training/validation/test sets)

- pre_process_image_files.py - Alter original files into various test categories: blur, noise, crop, etc. 

- freeze_autoenc_model.py - Put all the variables for a model in constants. Put everything for model in one .pb file.

- summarize_model.py - print out layers of model.

- merge_models.py - Combine classification model with autoencoder. Output into one big .pb file.


## Train Autoencoder Models

There are several python notebooks for training the autoencoder models. Trained on google colab to
take advantage of the GPU.  These scripts need a training/validation/test set of image files in google gdrive.

- train_pca_with_svd.ipynb Train a linear pca model using SVD (singular value decomposition) to learn the weights.

- train_pcanet.ipynb Train a linear pca autoencoder model on the whole training set.

- train_caenet.ipynb Train a 1-level contractive autoencoder

- train_cae_layers.ipynb Train each layer of a contractive autoencoder

- train_deep_cae.ipynb train the full multi-level contractive  autoencoder

- contracture_curve.ipynb - plot the rate of contraction for the feature space of an autoencoder

## Comparison Tests

- convphashaec.py - class file to hold tensorflow hub and autoencoder data.  

- cmpconvphashaec.py - script to run test comparing files in one directory to their modified counterparts in a corresponding directory.
				       Displays histogram plots.  Depends on convphashaec.py



