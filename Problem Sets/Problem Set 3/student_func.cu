/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

// CUDA doesn't have atomicMax for non-integers.
// This function taken from stack overflow!
__device__ static float atomicMaxFloat(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// CUDA doesn't have atomicMin for non-integers.
// This function taken from stack overflow!
__device__ static float atomicMinFloat(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void kernel_minMaxReduce(float* const ll_min, float* const ll_max, float* min_logLum, float* max_logLum, const size_t numRows, const size_t numCols) {
    int absolute_x = blockIdx.x * blockDim.x + threadIdx.x;
    int absolute_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (absolute_x >= numCols || absolute_y >= numRows) {
        return;
    }

    int index = absolute_y * numCols + absolute_x;

    // Block-wide reduction of elements to min and max.
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int loc = blockDim.x * blockDim.y / 2; loc > 0; loc >>= 1) {
        float rhs_min = 0.0f, rhs_max=0.0f;     

        // Find the absolute location of the rhs.
        int rhs_x_in_block = (threadIdx.x + loc) % blockDim.x;
        int rhs_y_in_block = threadIdx.y + ((threadIdx.x + loc) / blockDim.x);
        int absolute_rhs_x = blockIdx.x * blockDim.x + rhs_x_in_block;
        int absolute_rhs_y = blockIdx.y * blockDim.y + rhs_y_in_block;
        int rhs_index = absolute_rhs_y * numCols + absolute_rhs_x;        

        if (tid < loc /* && rhs_index < numRows * numCols */) {
            rhs_min = ll_min[rhs_index];
            rhs_max = ll_max[rhs_index];
        }

        __syncthreads();

        if (tid < loc) {
            ll_min[index] = min(ll_min[index], rhs_min);
            ll_max[index] = max(ll_max[index], rhs_max);
        }

        __syncthreads();
    }

    // Then use an atomic to reduce the global min and max values.
    if (tid == 0) {           
        atomicMinFloat(min_logLum, ll_min[index]);
        atomicMaxFloat(max_logLum, ll_max[index]);        
    }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

    const int THREAD_X = 128;
    const int THREAD_Y = 8;
    const dim3 blocks(THREAD_X, THREAD_Y, 1);
    const dim3 grid(numCols / THREAD_X + 1, numRows / THREAD_Y + 1, 1);

    float* d_min, * d_max;    
    checkCudaErrors(cudaMalloc(&d_min, sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_max, sizeof(float)));
    checkCudaErrors(cudaMemset(d_min, 0, sizeof(float)));
    checkCudaErrors(cudaMemset(d_max, 0, sizeof(float)));

    float* d_ll_min, * d_ll_max;    
    checkCudaErrors(cudaMalloc(&d_ll_min, sizeof(float) * numRows * numCols));
    checkCudaErrors(cudaMalloc(&d_ll_max, sizeof(float) * numRows * numCols));
    checkCudaErrors(cudaMemcpy(d_ll_min, d_logLuminance, sizeof(float) * numRows * numCols, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_ll_max, d_logLuminance, sizeof(float) * numRows * numCols, cudaMemcpyDeviceToDevice));

    // Find the min and max log luminance values.    
    kernel_minMaxReduce << <blocks, grid>> > (d_ll_min, d_ll_max, d_min, d_max, numRows, numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost));    

    printf("Device found: logLumMin=%4.8f; logLumMax=%4.8f\n", min_logLum, max_logLum);

    checkCudaErrors(cudaFree(d_min));
    checkCudaErrors(cudaFree(d_max));
    checkCudaErrors(cudaFree(d_ll_min));
    checkCudaErrors(cudaFree(d_ll_max));
}
