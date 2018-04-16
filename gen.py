#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.



"""
Generate training and test images.

"""


__all__ = (
    'generate_ims',
)


import itertools
import math
import os
import random
import sys

import cv2
import numpy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import common

FONT_DIR = "./fonts"
SMALL_FONT_HEIGHT = 40
FONT_HEIGHT = 80  # Pixel size to which the chars are resized
PLATE_HEIGHT = 165
PLATE_WIDTH = 330

OUTPUT_SHAPE = (64, 128)


def make_char_ims(font_path, chars, output_height):
    chars += " "
    font_size = output_height * 4

    font = ImageFont.truetype(font_path, font_size)

    height = max(font.getsize(c)[1] for c in chars)
    for c in chars:
        width = font.getsize(c)[0]
        im = Image.new("RGBA", (width, height), (0, 0, 0))

        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (255, 255, 255), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        yield c, numpy.array(im)[:, :, 0].astype(numpy.float32) / 255.


def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M


def pick_colors():
    first = True
    while first or plate_color - text_color < 0.3:
        text_color = random.random()
        plate_color = random.random()
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color


def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False

    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = numpy.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                              numpy.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= numpy.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (numpy.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if numpy.any(trans < -0.5) or numpy.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = numpy.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds


def generate_region():
    index = 1
    while 1:
        index = random.randrange(common.REGION_NUM)
        # NOTE(wtakase): Currently 3-chars not supported
        if index != 0 and index != 4 and index != 8:
            break
    start = common.REGIONS_SLICE[index]["start"]
    end = common.REGIONS_SLICE[index]["end"]
    return common.REGIONS[start:end]


def generate_class_number():
    return "{}{}{}".format(
        random.choice(common.DIGITS),
        random.choice(common.DIGITS),
        random.choice(common.DIGITS))


def generate_hiragana():
    return "{}".format(
        random.choice(common.HIRAGANAS))


def generate_number_code():
    #digit_num = random.randrange(1, 20)
    digit_num = random.randrange(4, 20)
    if digit_num == 1:
        return "{}{} {}{}".format(
            random.choice(common.DOT),
            random.choice(common.DOT),
            random.choice(common.DOT),
            random.choice(common.DIGITS))
    elif digit_num == 2:
        return "{}{} {}{}".format(
            random.choice(common.DOT),
            random.choice(common.DOT),
            random.choice(common.DIGITS),
            random.choice(common.DIGITS))
    elif digit_num == 3:
        return "{}{} {}{}".format(
            random.choice(common.DOT),
            random.choice(common.DIGITS),
            random.choice(common.DIGITS),
            random.choice(common.DIGITS))
    else:
        return "{}{}{}{}{}".format(
            random.choice(common.DIGITS),
            random.choice(common.DIGITS),
            random.choice(common.HYPHEN),
            random.choice(common.DIGITS),
            random.choice(common.DIGITS))


def rounded_rect(shape, radius):
    out = numpy.ones(shape)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)

    return out


def generate_plate(font_height, char_ims, plate_height, plate_width):
    h_padding = 20 * random.uniform(0.95, 1.05)
    v_padding = 15 * random.uniform(0.95, 1.05)
    region_spacing = 6 * random.uniform(0.95, 1.05)
    class_number_spacing = 3 * random.uniform(0.95, 1.05)
    lower_spacing = 9 * random.uniform(0.95, 1.05)
    radius = 1 + int(font_height * 0.1 * random.random())

    region = generate_region()
    class_number = generate_class_number()
    hiragana = generate_hiragana()
    number_code = generate_number_code()

    code = region + class_number + hiragana + number_code

    out_shape = (int(135 + v_padding * 2),
                 int(315 + h_padding * 2))

    text_color, plate_color = pick_colors()
    
    text_mask = numpy.zeros(out_shape)
    
    x = 85 * random.uniform(0.95, 1.05)
    y = v_padding
    for r in region:
        char_im = char_ims["region,%s" % common.FZ_FONT][r]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + region_spacing
    for c in class_number:
        char_im = char_ims["class_number,%s" % common.TRM_FONT][c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + class_number_spacing
    x = h_padding
    y += 75
    for h in hiragana:
        char_im = char_ims["hiragana,%s" % common.FZ_FONT][h]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + lower_spacing
    y -= 20
    for n in number_code:
        char_im = char_ims["number_code,%s" % common.TRM_FONT][n]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + lower_spacing

    plate = (numpy.ones(out_shape) * plate_color * (1. - text_mask) +
             numpy.ones(out_shape) * text_color * text_mask)

    return plate, rounded_rect(out_shape, radius), code


def generate_bg(num_bg_images):
    found = False
    while not found:
        fname = "bgs/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
        bg = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) / 255.
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
            bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True

    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

    return bg


def generate_im(char_ims, num_bg_images):
    bg = generate_bg(num_bg_images)

    plate, plate_mask, code = generate_plate(FONT_HEIGHT, char_ims,
                                             PLATE_HEIGHT, PLATE_WIDTH)
    
    M, out_of_bounds = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            min_scale=0.6,
                            max_scale=0.875,
                            rotation_variation=1.0,
                            scale_variation=1.5,
                            translation_variation=1.2)
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))

    out = plate * plate_mask + bg * (1.0 - plate_mask)

    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    out += numpy.random.normal(scale=0.05, size=out.shape)
    out = numpy.clip(out, 0., 1.)

    return out, code, not out_of_bounds


def load_fonts(folder_path):
    font_char_ims = {}
    fonts = [common.TRM_FONT, common.FZ_FONT]
    font_char_ims["region,%s" % common.FZ_FONT] = dict(
        make_char_ims(os.path.join(folder_path,
                                   common.FZ_FONT),
                      common.REGIONS,
                      SMALL_FONT_HEIGHT))
    font_char_ims["class_number,%s" % common.TRM_FONT] = dict(
        make_char_ims(os.path.join(folder_path,
                                   common.TRM_FONT),
                      common.DIGITS,
                      SMALL_FONT_HEIGHT))
    font_char_ims["hiragana,%s" % common.FZ_FONT] = dict(
        make_char_ims(os.path.join(folder_path,
                                   common.FZ_FONT),
                      common.HIRAGANAS,
                      SMALL_FONT_HEIGHT))
    font_char_ims["number_code,%s" % common.TRM_FONT] = dict(
        make_char_ims(os.path.join(folder_path,
                                   common.TRM_FONT),
                      common.TRM_CHARS,
                      FONT_HEIGHT))
    return fonts, font_char_ims


def generate_ims():
    """
    Generate number plate images.

    :return:
        Iterable of number plate images.

    """
    variation = 1.0
    fonts, font_char_ims = load_fonts(FONT_DIR)
    num_bg_images = len(os.listdir("bgs"))
    while True:
        yield generate_im(font_char_ims, num_bg_images)


if __name__ == "__main__":
    if not os.path.exists("test"):
        os.mkdir("test")
    im_gen = itertools.islice(generate_ims(), int(sys.argv[1]))
    for img_idx, (im, c, p) in enumerate(im_gen):
        fname = "test/{:08d}_{}_{}.png".format(img_idx, c,
                                               "1" if p else "0")
        print(fname)
        cv2.imwrite(fname, im * 255.)

