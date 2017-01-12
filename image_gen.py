"""
Author:     Henry Wolf
Twitter:    @chaoticneural

Date:       2017.01.12

Notes:      This file includes the functions for generating and loading basic images with text on them.
"""

from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import pandas as pd


def gen_image(txt, width=128, height=32, suffix='.tiff'):
    """
    This function takes a string and creates and image.

    :param txt: A string containing a word that will be used to generate an image.
    :param width:   Image width
    :param height:  Image height
    :param suffix:  Image filename suffix
    :return:    None
    """
    im = Image.new("F", (width, height), "#000")  # create the 32-bit float image
    draw = ImageDraw.Draw(im)  # create an object that can be used to draw in the given image
    font = ImageFont.truetype("fonts/Inconsolata-Regular.ttf", 24)  # set the font and its size
    w, h = draw.textsize(txt, font=font)  # get the size of the text
    draw.text(((width-w)/2, (height-h)/2), txt, font=font, fill="#fff")  # draw the text, centered
    image_path = 'op_images/'  # set the output location
    filename = txt + suffix  # append the suffix to the filename
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    filename = os.path.join(image_path, filename)  # join the location and filename for saving
    im.save(filename)  # save the image to a file
    return None


def gen_images(in_dict):
    """
    Loops through a dictionary and calls gen_image() for each word.

    :param in_dict: Dictionary of words for which images should be generated.
    :return:        None
    """
    for i in in_dict:
        gen_image(i)
    return None


def load_image(in_filename):
    """
    This function reads an image as an array and returns it flattened and scaled.

    :param in_filename: The filename of the image.
    :return:            Flattened, scaled array.
    """
    img = Image.open(in_filename)
    img.load()
    data = np.asarray(img, dtype="float32")
    flat_data = data.ravel() / 255  # scales to 0-1
    return flat_data


def load_images(in_dict, width=128, height=32, suffix='.tiff'):
    """
    Loops through a dictionary and calls load_image() for each word.

    :param in_dict: Dictionary of word images for which array should be loaded
    :param width:   Image width
    :param height:  Image height
    :param suffix:  Image filename suffix
    :return:        Array of image data
    """
    word_array = np.zeros((len(in_dict), width * height))  # placeholder for image arrays
    word_count = 0
    for i in in_dict:
        word_array[word_count] = load_image('op_images/' + i + suffix)
        word_count += 1  # iterates through word_array
    return word_array
