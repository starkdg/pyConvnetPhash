#! /usr/bin/env python3
from convphashaec import ConvPhashAutoEnc
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import numpy as np

""" Script to compare images in two directories, one being
    a slightly modified version of the others. The image files
    must have the same name so that similar images are compared.
"""

tf.logging.set_verbosity(tf.logging.WARN)

parser = argparse.ArgumentParser(prog="cmpconvphashaec", description="Compare Image files.")
parser.add_argument('--dir1', required=True,
                    help="Directory containing original images")
parser.add_argument('--dir2', required=True,
                    help="Directory containing modified images")

model_file = "/home/david/Downloads/tfmodels/caenet/mobilenetv2_cae_autoenc_1792to256_combined_frozen_model.py"

args = parser.parse_args()
print("model: ", model_file)
cp = ConvPhashAutoEnc(model_file)

print("process images in ", args.dir1)
# features1 = cp.image_features(args.dir1)
# features1 = cp.image_condensed(args.dir1)
# features1 = cp.image_condensed1024(args.dir1)
# features1 = cp.image_condensed512(args.dir1)
features1 = cp.image_condensed256(args.dir1)
# features1 = cp.image_condensed128(args.dir1)
# features1 = cp.image_condensed64(args.dir1)
# features1 = cp.image_condensed32(args.dir1)

print("features: ", features1.shape)
print("mean: ", np.mean(features1.ravel()))
print("std: ", np.std(features1.ravel()))
print("min: ", np.amin(features1.ravel()))
print("max: ", np.amax(features1.ravel()))

print("process images in ", args.dir2)
# features2 = cp.image_features(args.dir2)
# features2 = cp.image_condensed(args.dir2)
# features2 = cp.image_condensed1024(args.dir2)
# features2 = cp.image_condensed512(args.dir2)
features2 = cp.image_condensed256(args.dir2)
# features2 = cp.image_condensed128(args.dir2)
# features2 = cp.image_condensed64(args.dir2)
# features2 = cp.image_condensed32(args.dir2)

print("features: ", features2.shape)
print("mean: ", np.mean(features2.ravel()))
print("std: ", np.std(features2.ravel()))
print("min: ", np.amin(features2.ravel()))
print("max: ", np.amax(features2.ravel()))


print("Similarity measures.")
inter_distances = cp.l2_distance(features1, features2, axis=0)
# inter_distances = cp.hamming_distance(features1, features2)

inter_mean = np.mean(inter_distances)
inter_median = np.median(inter_distances)
inter_stddev = np.std(inter_distances)
inter_max = np.amax(inter_distances)
inter_min = np.amin(inter_distances)

print("  mean ", inter_mean)
print("  median ", inter_median)
print("  std dev ", inter_stddev)
print("  max ", inter_max)
print("  min ", inter_min)


print("Dissimilarity measures.")
m, n = features1.shape
intra_distances = []
for i in range(0, 4):
    x = features1[i, :]
    for j in range(i+1, m):
        y = features1[j, :]
        hdist = cp.l2_distance(x, y, axis=0)
        # hdist = cp.hamming_distance(x, y, axis=0)
        intra_distances.append(hdist)

intra_distances = np.array(intra_distances)
intra_mean = np.mean(intra_distances)
intra_median = np.median(intra_distances)
intra_stddev = np.std(intra_distances)
intra_max = np.amax(intra_distances)
intra_min = np.amin(intra_distances)

print("  mean ", intra_mean)
print("  median ", intra_median)
print("  std dev ", intra_stddev)
print("  max ", intra_max)
print("  min ", intra_min)

# calculate percentage of mismarked similarity distances
threshold1 = intra_mean - 2*intra_stddev
threshold2 = intra_mean - intra_stddev
mismarked_distances = inter_distances[inter_distances > threshold1]
mismarked_distances2 = inter_distances[inter_distances > threshold2]
print("threshold1: ", threshold1)
print("threshold2: ", threshold2)
print("pct above threshold1: ", np.prod(mismarked_distances.shape)/np.prod(inter_distances.shape))
print("pct above threshold2: ", np.prod(mismarked_distances2.shape)/np.prod(inter_distances.shape))


nbins = 100
plt.figure(1)
plt.hist([inter_distances, intra_distances], bins=nbins,
         color=['cyan', 'lime'],
         density=True,
         histtype='barstacked')
plt.xlabel("L2 Distance")
plt.ylabel("counts")
plt.title("Distances Between Similar/Dissimilar Image Features")
plt.show()


plt.figure(2)
plt.hist(features1.ravel(), density=True, bins=nbins)
plt.xlabel("values")
plt.ylabel("counts")
plt.title("Histogram of Feature Values")
plt.show()

print("Done.")
