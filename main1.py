import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import time

from harris_corner_detector.pycuda_single_kernel import pycuda_single_kernel


if __name__ == "__main__":
    # configuration
    file_name = 'images/test.png'
    k = 0.05
    thresh = 13266136331

    # load image
    image = scipy.misc.imread(file_name).astype(np.float32)

    # time the original python algorithm
    corners, time_took = pycuda_single_kernel(image[:, :, 0], k, thresh)

    print('time with cuda [s]:', time_took)
    print(corners.max())

