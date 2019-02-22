import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import time

from harris_corner_detector.pycuda_single_kernel import pycuda_single_kernel

if __name__ == "__main__":
    # configuration
    file_name = 'images/test2.png'
    k = 0.05
    thresh = 5000000000

    # load image
    image = scipy.misc.imread(file_name).astype(np.float32)

    # time the original python algorithm

    corners, timed = pycuda_single_kernel(image[:, :, 0], k, thresh)

    print('time with the original python algorithm [s]:', timed)
    print('Max corner response:', max(corners, key=lambda x: x[2]))

    # plot the image and a blue dot at the corners
    plt.imshow(image[:, :, 1])
    plt.scatter([x[0] for x in corners], [x[1] for x in corners])

    plt.show()
