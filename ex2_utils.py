import math
import numpy as np
import cv2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """

    return 315074963


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


# laplacian kernal
# One of the two
laplacian_ker = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    img = conv2D(img, laplacian_ker)
    zero_crossing = np.zeros(img.shape)
    for i in range(img.shape[0] - (laplacian_ker.shape[0] - 1)):
        for j in range(img.shape[1] - (laplacian_ker.shape[1] - 1)):
            if img[i][j] == 0:
                if (img[i][j - 1] < 0 and img[i][j + 1] > 0) or \
                        (img[i][j - 1] < 0 and img[i][j + 1] < 0) or \
                        (img[i - 1][j] < 0 and img[i + 1][j] > 0) or \
                        (img[i - 1][j] > 0 and img[i + 1][j] < 0):  # All his neighbors
                    zero_crossing[i][j] = 255
            if img[i][j] < 0:
                if (img[i][j - 1] > 0) or (img[i][j + 1] > 0) or (img[i - 1][j] > 0) or (img[i + 1][j] > 0):
                    zero_crossing[i][j] = 255
    return zero_crossing


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges usint "ZeroCrossingLOG" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    img = cv2.GaussianBlur(img, (5, 5), 0)
    return edgeDetectionZeroCrossingSimple(img)


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

    cannyImage = cv2.Canny((img * 255).astype(np.uint8), 100, 200)

    rows, cols = cannyImage.shape
    edges = []
    points = []
    circlesResult = []
    helpCircles = {}
    ths = 0.47  # at least 0.47% of the pixels of a circle must be detected
    steps = 100

    for r in range(min_radius, max_radius + 1):
        for s in range(steps):
            angle = 2 * math.pi * s / steps
            x = int(r * math.cos(angle))
            y = int(r * math.sin(angle))
            points.append((x, y, r))

    for i in range(rows):
        for j in range(cols):
            if cannyImage[i, j] == 255:
                edges.append((i, j))

    for e1, e2 in edges:
        for d1, d2, r in points:
            a = e2 - d2
            b = e1 - d1
            s = helpCircles.get((a, b, r))
            if s is None:
                s = 0
            helpCircles[(a, b, r)] = s + 1

    sortedCircles = sorted(helpCircles.items(), key=lambda i: -i[1])
    for circle, s in sortedCircles:
        x, y, r = circle
        if s / steps >= ths and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circlesResult):
            print(s / steps, x, y, r)
            circlesResult.append((x, y, r))

    return circlesResult


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    img2 = np.zeros(in_image.shape)
    gaussKer = get_gauss_kernel(k_size, sigma_color)
    sizeX, sizeY = in_image.shape
    for i in range(k_size // 2, sizeX - k_size // 2):
        for j in range(k_size // 2, sizeY - k_size // 2):
            imgS = get_slice(in_image, i, j, k_size)
            imgI = imgS - imgS[k_size // 2, k_size // 2]
            imgIG = vec_gaussian(imgI, sigma_space)
            weights = np.multiply(gaussKer, imgIG)
            vals = np.multiply(imgS, weights)
            val = np.sum(vals) / np.sum(weights)
            img2[i, j] = val
    cv_bilateral = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)
    return cv_bilateral, img2


def get_gauss_kernel(kernel_size: int, spatial_variance: float) -> np.ndarray:
    # Creates a gaussian kernel of given dimension.
    arr = np.zeros((kernel_size, kernel_size))
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            arr[i, j] = math.sqrt(
                abs(i - kernel_size // 2) ** 2 + abs(j - kernel_size // 2) ** 2
            )
    return vec_gaussian(arr, spatial_variance)


def vec_gaussian(img: np.ndarray, variance: float) -> np.ndarray:
    # For applying gaussian function for each element in matrix.
    sigma = math.sqrt(variance)
    cons = 1 / (sigma * math.sqrt(2 * math.pi))
    return cons * np.exp(-((img / sigma) ** 2) * 0.5)


def get_slice(img: np.ndarray, x: int, y: int, kernel_size: int) -> np.ndarray:
    half = kernel_size // 2
    return img[x - half: x + half + 1, y - half: y + half + 1]
