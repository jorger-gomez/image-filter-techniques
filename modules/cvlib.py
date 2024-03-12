""" HW5 - GitHub version control
This scripts is a Python module that implements the following functions:
    read_image: Reads an image.
    rotate_image: Rotates an image.
    translate_image: Translates an image.
    flip_image: Flips an image.
    visualise_image: Visualises an image.

Author: Jorge Rodrigo Gomez Mayo
Contact: jorger.gomez@udem.edu
Organization: Universidad de Monterrey
First created on Monday 11 March 2024
"""
#Import Std libraries
import cv2
import numpy as np
from typing import Optional

def read_image(filename: str) -> Optional[np.ndarray]:
    """
    Reads an image from a specified file path.

    Args:
        filename (str): Path to the input image.

    Returns:
        Optional[np.ndarray]: The read image as a NumPy array, or None if the image could not be read.
    """
    img = cv2.imread(filename)
    if img is None:
        print(f"ERROR! - Image {filename} could not be read!")
        return None
    return img

def rotate_image(img: np.ndarray, angle: float = 180, scale: float = 1.0) -> np.ndarray:
    """
    Rotates an image by a given angle around its center.

    Args:
        img (np.ndarray): Input image.
        angle (float): Rotation angle in degrees. Default is 180 degrees.
        scale (float): Scale factor. Default is 1.

    Returns:
        np.ndarray: The rotated image.
    """
    center = (img.shape[1] / 2, img.shape[0] / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return rotated_img

def translate_image(img: np.ndarray, x: float = 50, y: float = 50) -> np.ndarray:
    """
    Translates an image by a certain amount in x and y directions.

    Args:
        img (np.ndarray): Input image.
        x (float): Translation along the x-axis. Default is 50.
        y (float): Translation along the y-axis. Default is 50.

    Returns:
        np.ndarray: The translated image.
    """
    M = np.float32([[1, 0, x], [0, 1, y]])
    translated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return translated_img

def flip_image(img: np.ndarray, flip_code: int = 1) -> np.ndarray:
    """
    Flips an image either horizontally, vertically, or both.

    Args:
        img (np.ndarray): Input image.
        flip_code (int): Specifies how to flip the image: 0 for flipping around the x-axis (vertical flip),
                        1 for y-axis (horizontal flip), and -1 for both axes (both flips). Default is 1 (horizontal flip).

    Returns:
        np.ndarray: The flipped image.
    """
    return cv2.flip(img, flip_code)

def resize_image(img: np.ndarray, factor: float = 0.47) -> np.ndarray:
    """
    Resizes an image by a given factor.

    Args:
        img (np.ndarray): Input image.
        factor (float): The factor by which to resize the image. Default is 0.47.

    Returns:
        np.ndarray: The resized image.
    """
    width = int(img.shape[1] * factor)
    height = int(img.shape[0] * factor)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def visualise_image(img: np.ndarray, title: str = "Image", resize_factor: float = 0.47) -> None:
    """
    Displays an image with an optional resizing factor.

    Args:
        img (np.ndarray): Input image to display.
        title (str): Window title.
        resize_factor (float): Factor to resize the image by. Default is 0.47.

    Note:
        Press any key to close the displayed window.
    """
    resized_img = resize_image(img, resize_factor)
    cv2.imshow(title, resized_img)
    cv2.waitKey(0)