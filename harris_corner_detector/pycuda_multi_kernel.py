from string import Template
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np


def pycuda_multi_kernel(img, k_harris, thresh, executions):
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

    # Find x and y derivatives
    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2

    ixx = Ixx.reshape(vector_size, order='F')
    ixy = Ixy.reshape(vector_size, order='F')
    iyy = Iyy.reshape(vector_size, order='F')
    dest_r = np.zeros_like(ixx)

    # the image is divided in four parts and processed in 4 diff kernels
    n = 4  # Number of slices (and concurrent operations) used.

    k_height = height
    k_width = int(width / n)

    func_mod = SourceModule(func_mod_template.substitute(HEIGHT=k_height,
                                                         WIDTH=k_width))
    pycuda_corners = func_mod.get_function("corners")

    ###### Start concurrency configuration #######

    # Allocate memory on the host.
    d_ixx, d_ixy, d_iyy, d_dest_r = [], [], [], []
    slice_size = int(vector_size/n)
    for k in range(n):
        # Allocate memory on device.
        d_ixx.append(drv.mem_alloc(ixx[0:slice_size].nbytes))
        d_ixy.append(drv.mem_alloc(ixy[0:slice_size].nbytes))
        d_iyy.append(drv.mem_alloc(iyy[0:slice_size].nbytes))
        d_dest_r.append(drv.mem_alloc(dest_r[0:slice_size].nbytes))

    # Create the streams and events needed.
    stream = []
    event = []
    event_dtoh = []
    marker_names = ['kernel_begin', 'kernel_end']
    for k in range(n):
        stream.append(drv.Stream())
        event.append(dict([(marker_names[l], drv.Event()) \
                    for l in range(len(marker_names))]))
        event_dtoh.append(drv.Event())

    # Use this event as a reference point.
    ref = drv.Event()
    finish = drv.Event()
    ref.record()

    #### Important ######
    # The size of the slices must be larger (+ offset) to calculate
    # r at the limits of each section of the image.
    # This version does not calculate an r values for the limits of the 
    # different image sections.
    #### Important ######

    for _ in range(executions):
        # Transfer to device.
        for k in range(n):
            drv.memcpy_htod_async(d_ixx[k], ixx[slice_size*k:slice_size*(k+1)],
                                  stream=stream[k])
            drv.memcpy_htod_async(d_ixy[k], ixy[slice_size*k:slice_size*(k+1)],
                                  stream=stream[k])
            drv.memcpy_htod_async(d_iyy[k], iyy[slice_size*k:slice_size*(k+1)],
                                  stream=stream[k])
            drv.memcpy_htod_async(d_dest_r[k],
                                  dest_r[slice_size*k:slice_size*(k+1)],
                                  stream=stream[k])

        # Run kernels
        for k in range(n):
            event[k]['kernel_begin'].record(stream[k])
            pycuda_corners(d_dest_r[k],
                           d_ixx[k],
                           d_ixy[k],
                           d_iyy[k],
                           np.uint32(offset),
                           np.float32(k_harris),
                           np.uint32(thresh),
                           # max 1024 threds, 32x32 is a regular choice
                           block=(32, 32, 1),
                           grid=(int(128/n), 1, 1),
                           stream=stream[k])
        for k in range(n):
            event[k]['kernel_end'].record(stream[k])

        # Transfer data back to host.
        for k in range(n):
            drv.memcpy_dtoh_async(dest_r[slice_size*k:slice_size*(k+1)],
                                  d_dest_r[k],
                                  stream=stream[k])
            # event that it completed the transfer
            event_dtoh[k].record(stream[k])
            stream[k].synchronize()

    # finish
    finish.record()
    finish.synchronize()

    ###### Output results #####

    print('Timing info of stream launches in seconds')
    for k in range(n):
        print('Stream', k)
        for l in range(len(marker_names)):
            print(marker_names[l], ':',
                  ref.time_till(event[k][marker_names[l]]) * 1e-3)

    # extract the corners
    r = np.reshape(dest_r, (256, 512), order='F')
    corners = np.where(r > 0)

    for i, j in zip(corners[0], corners[1]):
        corner_list.append([j, i, r[i, j]])

    average_execution_time = (ref.time_till(finish) * 1e-3) / executions

    # for profiling
    # pycuda.autoinit.context.detach()

    return corner_list, average_execution_time
