""" HW5 - GitHub version control
This scripts performs image filtering by implementing 
the following image filters using OpenCV: median, average, and
Gaussian filters.

Author: Jorge Rodrigo Gomez Mayo
Contact: jorger.gomez@udem.edu
Organization: Universidad de Monterrey
First created on Monday 11 March 2024
"""
#Import Std libraries
import argparse
import cv2
from typing import Optional

#Import developed libraries
import modules.cvlib as cl

def validate_kernel_size(kernel_size: int) -> bool:
    """
    Checks if the kernel size is a positive odd integer.

    Args:
        kernel_size (int): The kernel size to check.

    Returns:
        bool: True if the kernel size is a valid positive odd integer, False otherwise.
    """
    return kernel_size > 0 and kernel_size % 2 == 1

def apply_median_filter(img: cv2.Mat, kernel_size: int = 11) -> Optional[cv2.Mat]:
    """
    Applies a median filter to the image with a default kernel size of 11.

    Args:
        img (cv2.Mat): Input image.
        kernel_size (int): Size of the kernel. Must be a positive odd integer. Default is 11.

    Returns:
        Optional[cv2.Mat]: The filtered image or None if input is invalid or kernel size is not valid.
    """
    if img is None or not validate_kernel_size(kernel_size):
        return None
    return cv2.medianBlur(img, kernel_size)

def apply_average_filter(img: cv2.Mat, kernel_size: int = 11) -> Optional[cv2.Mat]:
    """
    Applies an average (blur) filter to the image with a default kernel size of 11.

    Args:
        img (cv2.Mat): Input image.
        kernel_size (int): Size of the kernel (width and height). Must be a positive odd integer. Default is 11.

    Returns:
        Optional[cv2.Mat]: The filtered image or None if input is invalid or kernel size is not valid.
    """
    if img is None or not validate_kernel_size(kernel_size):
        return None
    return cv2.blur(img, (kernel_size, kernel_size))

def apply_gaussian_filter(img: cv2.Mat, kernel_size: int = 11, sigma_x: float = 0) -> Optional[cv2.Mat]:
    """
    Applies a Gaussian filter to the image with a default kernel size of 11 and default sigma_x of 0.

    Args:
        img (cv2.Mat): Input image.
        kernel_size (int): Size of the kernel. Must be a positive odd integer. Default is 11.
        sigma_x (float): Standard deviation in the X direction. If 0, it is calculated from the kernel size. Default is 0.

    Returns:
        Optional[cv2.Mat]: The filtered image or None if input is invalid or kernel size is not valid.
    """
    if img is None or not validate_kernel_size(kernel_size):
        return None
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma_x)

def parse_user_data():
    """
    Parse the command-line arguments provided by the user.
    """
    parser = argparse.ArgumentParser(prog='HW5 - Image Filtering',
                                    description='Performs image filtering by implementing median, average, and Gaussian filters using OpenCV.',
                                    epilog='JRGM - 2024')
    parser.add_argument('-i', '--input_image', type=str, required=True,
                        help='Path to the input image')
    parser.add_argument('-m', '--median_kernel_size', type=int, default=11,
                        help='Kernel size for the median filter. Must be a positive odd integer. Default is 11.')
    parser.add_argument('-a', '--average_kernel_size', type=int, default=11,
                        help='Kernel size for the average filter. Must be a positive odd integer. Default is 11.')
    parser.add_argument('-g', '--gaussian_kernel_size', type=int, default=11,
                        help='Kernel size for the Gaussian filter. Must be a positive odd integer. Default is 11.')
    parser.add_argument('-s', '--sigma_x', type=float, default=0,
                        help='Standard deviation in the X direction for the Gaussian filter. Default is 0.')

    args = parser.parse_args()
    return args

def main():
    args = parse_user_data()

    # Load the image
    img = cl.read_image(args.input_image)
    if img is None:
        print("Failed to load image.")
        return

    # Apply filters
    median_filtered = apply_median_filter(img, args.median_kernel_size)
    average_filtered = apply_average_filter(img, args.average_kernel_size)
    gaussian_filtered = apply_gaussian_filter(img, args.gaussian_kernel_size, args.sigma_x)

    # Visualise the original and filtered images
    cl.visualise_image(img, "Original Image")
    if median_filtered is not None:
        cl.visualise_image(median_filtered, "Median Filtered Image")
    if average_filtered is not None:
        cl.visualise_image(average_filtered, "Average Filtered Image")
    if gaussian_filtered is not None:
        cl.visualise_image(gaussian_filtered, "Gaussian Filtered Image")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()