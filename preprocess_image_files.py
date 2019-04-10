#! /usr/bin/env python3
import argparse
import os
import skimage
import numpy as np
import math
import matplotlib
import PIL
from skimage import io, filters, transform, util, draw, exposure
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser("Preprocess image files.")
parser.add_argument("--dir",
                    required=True,
                    help="directory of images to process")
args = parser.parse_args()
img_dir = args.dir

print("process images in ", img_dir)
os.chdir(path=img_dir)

orig_dir = "original"
blur_dir = "blurred"
compr_dir = "compressed"
cropped_dir = "cropped"
decimated_dir = "decimated"
upscale_dir = "upscale"
downscale_dir = "downscale"
noise_dir = "noise"
occluded_dir = "occluded"
rotated_dir = "rotated"
horizflip_dir = "hflip"
vertflip_dir = "vflip"
brighten_dir = "brighten"
darken_dir = "darken"
histeq_dir = "histeq"
shear_dir = "shear"

try:
    print("Create Directories")
    os.mkdir(orig_dir)
    os.mkdir(blur_dir)
    os.mkdir(compr_dir)
    os.mkdir(cropped_dir)
    os.mkdir(decimated_dir)
    os.mkdir(noise_dir)
    os.mkdir(occluded_dir)
    os.mkdir(rotated_dir)
    os.mkdir(upscale_dir)
    os.mkdir(downscale_dir)
    os.mkdir(horizflip_dir)
    os.mkdir(vertflip_dir)
    os.mkdir(brighten_dir)
    os.mkdir(darken_dir)
    os.mkdir(histeq_dir)
    os.mkdir(shear_dir)
except FileExistsError:
    print("Some Directories already exist.")


def blur_image_and_save(file, img, sigma):
    path = os.path.join(img_dir, blur_dir, file)
    blurred_img = filters.gaussian(img, sigma=sigma)
    io.imsave(path, blurred_img, plugin="pil", quality=100)


def compress_image_and_save(file, img, quality=50):
    path = os.path.join(img_dir, compr_dir, file)
    io.imsave(path, img, plugin="pil", quality=quality)


def crop_image_and_save(file, img, pct_margin=0.05):
    path = os.path.join(img_dir, cropped_dir, file)
    h, w, d = img.shape
    w_margin = int(w*pct_margin)
    h_margin = int(h*pct_margin)
    cropped = util.crop(img, ((h_margin, h_margin), (w_margin, w_margin),(0,0)), copy=True)
    io.imsave(path, cropped, plugin="pil", quality=100)


def decimate_image_and_save(file, img, factors):
    path = os.path.join(img_dir, decimated_dir, file)
    decimated = transform.downscale_local_mean(img, factors)
    decimated = exposure.rescale_intensity(decimated, in_range='image', out_range=(0.0,1.0))
    decimated = util.img_as_ubyte(decimated)
    io.imsave(path, decimated, plugin="pil", quality=100)


def scale_image_and_save(file, img, dir_name, scale_factor):
    path = os.path.join(img_dir, dir_name, file)
    scaled = transform.rescale(img, scale_factor, anti_aliasing_sigma=1.0)
    io.imsave(path, scaled, plugin="pil", quality=100)


def noise_image_and_save(file, img, gaussian_var=0.005):
    path = os.path.join(img_dir, noise_dir, file)
    noisy = util.random_noise(img, mode='gaussian', mean=0, var=gaussian_var)
    io.imsave(path, noisy, plugin="pil", quality=100)


def occlude_image_and_save(file, img, font_size=40):
    path = os.path.join(img_dir, occluded_dir, file)
    h, w, d = img.shape
    y = int(0.75*h)
    x = int(0.10*w)
    myfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    imgpil = Image.fromarray(img)
    draw_ctx = ImageDraw.Draw(imgpil)
    draw_ctx.text((x,y), file, fill=(255, 255, 0), font=myfont)
    occluded_img = np.asarray(imgpil)
    io.imsave(path, occluded_img, plugin="pil", quality=100)


def rotate_image_and_save(file, img, degrees=4.0):
    path = os.path.join(img_dir, rotated_dir, file)
    rotd_img = transform.rotate(img, degrees, resize=False)
    h, w, d = rotd_img.shape
    h_margin = int(0.05 * h)
    w_margin = int(0.05 * w)
    cropped_img = util.crop(rotd_img, ((h_margin, h_margin), (w_margin, w_margin), (0, 0)),  copy=True)
    io.imsave(path, cropped_img, plugin="pil", quality=100)

    
def hflip_image_and_save(file, img):
    path = os.path.join(img_dir, horizflip_dir, file)
    flipped = np.fliplr(img)
    io.imsave(path, flipped, plugin="pil", quality=100)


def vflip_image_and_save(file, img):
    path = os.path.join(img_dir, vertflip_dir, file)
    flipped = np.flipud(img)
    io.imsave(path, flipped, plugin="pil", quality=100)


def gamma_correction_and_save(file, img, dir_name, gamma=1):
    path = os.path.join(img_dir, dir_name, file)
    imgfl = util.img_as_float(img)
    img2 = exposure.adjust_gamma(imgfl, gamma=gamma)
    img2 = util.img_as_ubyte(img2)
    io.imsave(path, img2, plugin="pil", quality=100)


def histogram_equalization_and_save(file, img):
    path = os.path.join(img_dir, histeq_dir, file)
    img_eq = exposure.equalize_hist(img)
    img2 = util.img_as_ubyte(img_eq)
    io.imsave(path, img2, plugin="pil", quality=100)


def shear_image_and_save(file, img, shear, translation):
    path = os.path.join(img_dir, shear_dir, file)
    tform = transform.AffineTransform(shear=shear, translation=translation)
    img_warped = transform.warp(img, tform)
    img2 = util.img_as_ubyte(img_warped)
    io.imsave(path, img2, plugin="pil", quality=100)


def save_original(file, img):
    path = os.path.join(img_dir, orig_dir, file)
    io.imsave(path, img, plugin="pil", quality=100)


count = 0
for entry in os.scandir(args.dir):
    if entry.is_file() and entry.name.endswith(".jpg"):
        img = io.imread(entry.name, plugin='pil')
        if len(img.shape) == 3 and img.shape[2] >= 3:
            print("({0}) : {1}".format(count, entry.name))
            save_original(entry.name, img)
            blur_image_and_save(entry.name, img, sigma=1.25)
            compress_image_and_save(entry.name, img, quality=35)
            crop_image_and_save(entry.name, img, pct_margin=0.10)
            decimate_image_and_save(entry.name, img, (2, 2, 1))
            noise_image_and_save(entry.name, img, gaussian_var=0.005)
            occlude_image_and_save(entry.name, img, font_size=60)
            rotate_image_and_save(entry.name, img, degrees=5.0)
            scale_image_and_save(entry.name, img, downscale_dir, scale_factor=0.70)
            scale_image_and_save(entry.name, img, upscale_dir, scale_factor=1.30)
            hflip_image_and_save(entry.name, img)
            vflip_image_and_save(entry.name, img)
            gamma_correction_and_save(entry.name, img, darken_dir, gamma=2.2)
            gamma_correction_and_save(entry.name, img, brighten_dir, gamma=0.4)
            histogram_equalization_and_save(entry.name, img)
            shear_image_and_save(entry.name, img, math.pi/8, (20, 10))
        os.remove(entry.name)
        count = count + 1

print("{0} files".format(count))
print("Done.")
