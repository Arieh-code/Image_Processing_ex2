import math
import numpy as np
import cv2


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    k_len = len(k_size)
    in_signal = np.pad(in_signal, (k_len - 1, k_len - 1), 'constant')
    sig_len = len(in_signal)
    signal_conv = np.zeros(sig_len - k_len + 1)
    for i in range(len(signal_conv)):
        signal_conv[i] = (in_signal[i:i + k_len] * k_size).sum()
    return signal_conv


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """

    kernel = np.flip(kernel)
    img_h, img_w = in_image.shape
    ker_h, ker_w = kernel.shape
    image_padded = np.pad(in_image, (ker_h // 2, ker_w // 2), 'edge')
    output = np.zeros((img_h, img_w))
    for y in range(img_h):
        for x in range(img_w):
            output[y, x] = (image_padded[y:y + ker_h, x:x + ker_w] * kernel).sum()
    return output


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """

    Gx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    Gy = Gx.transpose()

    x_der = conv2D(in_image, Gx)
    y_der = conv2D(in_image, Gy)

    directions = np.rad2deg(np.arctan2(y_der, x_der))
    # directions[directions < 0] += 180

    magnitude = np.sqrt(np.square(x_der) + np.square(y_der))
    # magnitude = magnitude * 255.0 / magnitude.max()

    return directions, magnitude


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    assert (k_size % 2 == 1)
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    return conv2D(in_image, create_gaussian(k_size, sigma))


def create_gaussian(size, sigma):
    mid = size // 2
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x, y = i - mid, j - mid
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return kernel


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    assert (k_size % 2 == 1)
    sigma = int(round(0.3 * ((k_size - 1) * 0.5 - 1) + 0.8))
    kernel = cv2.getGaussianKernel(k_size, sigma)
    return cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    return


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges usint "ZeroCrossingLOG" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    return


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    return


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    return
