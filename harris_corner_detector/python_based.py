import numpy as np


def hcd_python_based(img, k, thresh):
    """
    Finds and returns list of corners
        :param img: grayscale image
        :param k: Harris corner constant. Usually 0.04 - 0.06
        :param thresh: The threshold above which a corner is counted
        :return:
    """
    # Find x and y derivatives
    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    height = img.shape[0]
    width = img.shape[1]
    cornerList = []
    offset = 2
    # Loop through image and find our corners
    print("Finding Corners...")
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
            # If corner response is over threshold, color the point and add to
            # corner list
            if r > thresh:
                cornerList.append([x, y, r])
    return cornerList