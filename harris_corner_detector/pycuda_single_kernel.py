from string import Template
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np


def pycuda_single_kernel(img, k, thresh, executions):
    """
    Finds and returns list of corners
        :param img: grayscale image
        :param k: Harris corner constant. Usually 0.04 - 0.06
        :param thresh: The threshold above which a corner is counted
        :param executions: Number of times to be executed
        :return: corner_list: List with corners
        :return: average_execution_time: Average execution time in seconds
    """
    # only for 256 by 512 images
    assert img.shape[0] == 256  # height
    assert img.shape[1] == 512  # width
    height = img.shape[0]
    width = img.shape[1]

    vector_size = img.shape[0] * img.shape[1]
    corner_list = []
    offset = 2
    # to fit still in a 32-bit integer
    thresh = int(thresh/10)

    # function template
    func_mod_template = Template("""
    #include<stdio.h>
    #define INDEX(a, b) a*${HEIGHT}+b

    __global__ void corners(
        float *dest,
        float *ixx,
        float *ixy,
        float *iyy,
        int offset,
        float k,
        int threshold) {

        unsigned int idx = threadIdx.x + threadIdx.y*blockDim.y +
                            (blockIdx.x*(blockDim.x*blockDim.y));

        unsigned int a = idx/${HEIGHT};
        unsigned int b = idx%${HEIGHT};

        float sxx = 0;
        float sxy = 0;
        float syy = 0;
        float det = 0;
        float trace = 0;
        float r = 0;

        if ((a >= offset) & (a <= (${WIDTH}-offset - 1)) &
            (b >= offset) & (b <= (${HEIGHT}-offset - 1))) {
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

    func_mod = SourceModule(func_mod_template.substitute(HEIGHT=height,
                                                         WIDTH=width))
    pycuda_corners = func_mod.get_function("corners")

    # Find x and y derivatives
    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2

    ixx = Ixx.reshape(vector_size, order='F')
    ixy = Ixy.reshape(vector_size, order='F')
    iyy = Iyy.reshape(vector_size, order='F')
    dest_r = np.zeros_like(ixx)

    # start timer
    start = drv.Event()
    end = drv.Event()
    start.record()

    for _ in range(executions):
        pycuda_corners(drv.Out(dest_r),
                    drv.In(ixx),
                    drv.In(ixy),
                    drv.In(iyy),
                    np.uint32(offset),
                    np.float32(k),
                    np.uint32(thresh),
                    block=(32, 32, 1),  # max 1024, typical choice is 32x32
                    grid=(128, 1, 1))  # 1024 * 128 = 256 * 512
    # stop timer
    end.record()
    end.synchronize()

    # calculate used time
    average_execution_time = (start.time_till(end) * 1e-3) / executions

    # extract the corners
    r = np.reshape(dest_r, (256, 512), order='F')
    corners = np.where(r > 0)
    for i, j in zip(corners[0], corners[1]):
        corner_list.append([j, i, r[i, j]])

    return corner_list, average_execution_time
