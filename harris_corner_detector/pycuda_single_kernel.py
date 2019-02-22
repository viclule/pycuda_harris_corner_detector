import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time


def pycuda_single_kernel(img, k, thresh):
    """
    Finds and returns list of corners
        :param img: grayscale image
        :param k: Harris corner constant. Usually 0.04 - 0.06
        :param thresh: The threshold above which a corner is counted
        :return:
    """
    # only for 256 by 512 images
    assert img.shape[0] == 256  # height
    assert img.shape[1] == 512  # width

    vector_size = img.shape[0] * img.shape[1]
    corner_list = []
    offset = 2
    # to fit still in a 32-bit integer
    thresh = int(thresh/10)

    # function
    mod = SourceModule("""
    #include<stdio.h>
    #define INDEX(a, b) a*256+b

    __global__ void corners(
        float *dest,
        float *ixx,
        float *ixy,
        float *iyy,
        int offset,
        float k,
        int threshold) {

        unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));

        unsigned int a = idx/256;
        unsigned int b = idx%256;
        float sxx = 0;
        float sxy = 0;
        float syy = 0;
        float det = 0;
        float trace = 0;
        float r = 0;

        if ((a >= offset) & (a <= (512-offset - 1)) &
            (b >= offset) & (b <= (256-offset - 1))) {
            for (int bi = b - offset; bi < b + offset + 1; ++bi) {
                for (int ai = a - offset; ai < a + offset + 1; ++ai) {
                    sxx = sxx + ixx[INDEX(ai, bi)];
                    sxy = sxy + ixy[INDEX(ai, bi)];
                    syy = syy + iyy[INDEX(ai, bi)];
                }
            }

            det = sxx*syy - sxy*sxy;
            trace = sxx + syy;
            r = det - k*(trace*trace);
            if ((r/10) > threshold)
                dest[INDEX(a, b)] = r;
        }
    }
    """)

    # Find x and y derivatives
    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2

    ixx = Ixx.reshape(vector_size, order='F')
    ixy = Ixy.reshape(vector_size, order='F')
    iyy = Iyy.reshape(vector_size, order='F')
    dest_r = np.zeros_like(ixx)

    pycuda_corners = mod.get_function("corners")
    start = time.time()
    pycuda_corners(drv.Out(dest_r),
                   drv.In(ixx),
                   drv.In(ixy),
                   drv.In(iyy),
                   np.uint32(offset),
                   np.float32(k),
                   np.uint32(thresh),
                   block=(1024, 1, 1),
                   grid=(128, 1, 1))
    end = time.time()

    # extract the corners
    r = np.reshape(dest_r, (256, 512), order='F')
    corners = np.where(r > 0)
    for i, j in zip(corners[0], corners[1]):
        corner_list.append([j, i, r[i, j]])

    return corner_list, end - start
