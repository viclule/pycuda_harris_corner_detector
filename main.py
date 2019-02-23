import sys
import os
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

from harris_corner_detector.python_based import hcd_python_based
from harris_corner_detector.pycuda_single_kernel import pycuda_single_kernel
from harris_corner_detector.pycuda_multi_kernel import pycuda_multi_kernel


if __name__ == "__main__":
    # option 'py', 'cuda' or 'mcuda'
    if len(sys.argv) != 2:
        print("Please provide an option. 'py', 'cuda' or 'mcuda'.")
        quit()
    option = sys.argv[1]

    executions = 1000
    if option == 'py':
        executions = 10

    # configuration
    wk_dir = os.path.abspath(os.path.dirname('__file__'))
    file_name = os.path.join(wk_dir, r'images/test.png')
    # only during profiling
    # file_name = \
    #   r'C:\_prog\git_wf\pycuda_harris_corner_detector\images\test.png'

    k = 0.05
    thresh = 5000000000

    # load image
    image = scipy.misc.imread(file_name).astype(np.float32)

    # run the algorithm
    if option == 'py':
        corners, timed = hcd_python_based(image[:, :, 0],
                                          k, thresh, executions)
    elif option == 'cuda':
        corners, timed = pycuda_single_kernel(image[:, :, 0],
                                              k, thresh, executions)
    elif option == 'mcuda':
        corners, timed = pycuda_multi_kernel(image[:, :, 0],
                                             k, thresh, executions)
    else:
        print("Please provide an option. 'py', 'cuda' or 'mcuda'.")
        quit()

    # print some results
    print('Average time with', option, 'algorithm [s]:', timed)
    print('Max corner response:', max(corners, key=lambda x: x[2]))

    # plot the image and a blue dot at the corners
    plt.imshow(image[:, :, 1], cmap='gray')
    plt.scatter([x[0] for x in corners], [x[1] for x in corners])
    plt.show()
