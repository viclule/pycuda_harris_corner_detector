import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import time

from harris_corner_detector.python_based import hcd_python_based


if __name__ == "__main__":
    # configuration
    file_name = 'images/test2.png'
    k = 0.05
    thresh = 5266136331

    # load image
    image = scipy.misc.imread(file_name).astype(np.float32)

    # time the original python algorithm
    start = time.time()
    corners = hcd_python_based(image[:, :, 0], k, thresh)
    end = time.time()

    print('time with the original python algorithm [s]:', end - start)
    print('Max corner response:', max(corners, key=lambda x: x[2]))

    # plot the image and a blue dot at the corners
    plt.imshow(image[:, :, 1])
    plt.scatter([x[0] for x in corners], [x[1] for x in corners])

    plt.show()
