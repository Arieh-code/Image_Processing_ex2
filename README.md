# Image-Processing Ex2


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Content</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#files-to-submit">Files To Submit</a></li>
    <li><a href="#function-details">Function Details</a></li>
    <li><a href="#languages-and-tools">Languages and Tools</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

----------------

<!-- ABOUT THE PROJECT -->
# About The Project
*_Image-Processing Ex2:_*

The purpose of this exercise is to help you understand the concept of the convolution and edge
detection by performing simple manipulations on images.

* Implement convolution on 1D and 2D arrays.
* Performing image derivative and blurring.
* Edge detecting. 
* Hough circle.
* Bilitarel filtering.


``` Version Python 3.10.4```

``` Pycharm```

## Files To Submit

* ex2_main - This file runs all the code
* ex2_utils - This file has all the functions  
* Ex2 - This is the assignment pdf 
* images 
* Readme

---------------------

## Function Details

The discription of each function is the same discription we are given in the assingment from the [pdf](https://github.com/Arieh-code/Image_Processing_ex2/blob/master/Ex2__Convolution_Edge_Detection.pdf)
 
Write two functions that implement convolution of 1 1D discrete signal and 2D discrete signal.

The two function should have the following interface:

```python
def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
```

```python
def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
```

Write a function that computes the magnitude and the direction of an image gradient. You should derive
the image in each direction separately (rows and column) using simple convolution with [1, 0, −1]T and
[1, 0, −1] to get the two image derivatives. Next, use these derivative images to compute the magnitude
and direction matrix and also the x and y derivatives.


The function should have the following interfaces:
```python
def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
```


You should write two functions that performs image blurring using convolution between the image f and
a Gaussian kernel g. 

The functions should have the following interface:

```python
def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
```

```python
def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
```

You should implement edgeDetectionZeroCrossingLOG OR edgeDetectionZeroCrossingSimple

I is the intensity image and edgeImage is binary image (zero/one) with ones in the places the function
identifies edges. Each function implements edge detections accroding to a different method. edgeImage1
is the edge image of your own implementation and edgeImage2 is the edge image returned by pythons’s
edge function with appropriate parameters.

The function should have the following interface:

```python
def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: opencv solution, my implementation
    """
```

```python
def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges usint "ZeroCrossingLOG" method
    :param img: Input image
    :return: opencv solution, my implementation
    """
```

You should implement the Hough circles transform.


```python
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
```

You should implement the Bilateral filter, compare your implementation with OpenCV implementation
cv.bilateralFilter() 

```python
def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
```

---------------------

## Languages and Tools



  <div align="center">
  
 <code><img height="40" width="40" src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png"></code> 
 <code><img height="40" width="80" src="https://matplotlib.org/_static/logo2_compressed.svg"/></code>
 <code><img height="40" width="40" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/PyCharm_Icon.svg/1024px-PyCharm_Icon.svg.png"/></code>
 <code><img height="40" height="40" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/git/git.png"></code>
 <code><img height="40" height="40" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/terminal/terminal.png"></code>
  </div>


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Python](https://www.python.org/)
* [Matplotlib](https://matplotlib.org/)
* [Git](https://git-scm.com/)
* [Pycharm](https://www.jetbrains.com/pycharm/)
* [Git-scm](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)


