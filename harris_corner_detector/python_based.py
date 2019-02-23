import numpy as np
import time


def hcd_python_based(img, k, thresh, executions):
    """
    Finds and returns list of corners
        :param img: grayscale image
        :param k: Harris corner constant. Usually 0.04 - 0.06
        :param thresh: The threshold above which a corner is counted
        :param executions: Number of times to be executed
        :return: corner_list: List with corners
        :return: average_execution_time: Average execution time in seconds
    """
    # Find x and y derivatives
    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    height = img.shape[0]
    width = img.shape[1]
    corner_list = []
    offset = 2

    print('Executing', executions, 'times... it might take up to a minute...')

    start = time.time()  # starting here for a fair comparison
    # Loop through image and find our corners
    for _ in range(executions):
        for y in range(offset, height-offset):
            for x in range(offset, width-offset):
                # Calculate sum of squares
                windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
                windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
                windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
                Sxx = windowIxx.sum()
                Sxy = windowIxy.sum()
                Syy = windowIyy.sum()
                # Find determinant and trace, use to get corner response
                det = (Sxx * Syy) - (Sxy**2)
                trace = Sxx + Syy
                r = det - k*(trace**2)
                # If corner response is over threshold, color the point
                # and add to corner list
                if r > thresh:
                    corner_list.append([x, y, r])
    end = time.time()
    average_execution_time = (end - start) / executions
    return corner_list, average_execution_time
