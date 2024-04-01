// Homework 1
// Color to Greyscale Conversion

//A common way to represent color images is known as RGBA - the color
//is specified by how much Red, Grean and Blue is in it.
//The 'A' stands for Alpha and is used for transparency, it will be
//ignored in this homework.

//Each channel Red, Blue, Green and Alpha is represented by one byte.
//Since we are using one byte for each color there are 256 different
//possible values for each color.  This means we use 4 bytes per pixel.

//Greyscale images are represented by a single intensity value per pixel
//which is one byte in size.

//To convert an image from color to grayscale one simple method is to
//set the intensity to the average of the RGB channels.  But we will
//use a more sophisticated method that takes into account how the eye 
//perceives color and weights the channels unequally.

//The eye responds most strongly to green followed by red and then blue.
//The NTSC (National Television System Committee) recommends the following
//formula for color to greyscale conversion:

//I = .299f * R + .587f * G + .114f * B

//Notice the trailing f's on the numbers which indicate that they are 
//single precision floating point constants and not double precision
//constants.

//You should fill in the kernel as well as set the block and grid sizes
//so that the entire image is processed.

#include "utils.h"

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  //TODO
  //Fill in the kernel to convert from color to greyscale
  //the mapping from components of a uchar4 to RGBA is:
  // .x -> R ; .y -> G ; .z -> B ; .w -> A
  //
  //The output (greyImage) at each pixel should be the result of
  //applying the formula: output = .299f * R + .587f * G + .114f * B;
  //Note: We will be ignoring the alpha channel for this conversion

  //First create a mapping from the 2D block and grid locations
  //to an absolute 2D location in the image, then use that to
  //calculate a 1D offset

    // Instead of doing what the assignment asks, and converting it into a 2D index, 
    // I convert it directly into a 1-D index. This is easier for me to understand.
    // 
    // Imagine the entire image laid out in row-major format. 
    // So one row after the other is laid out linearly in memory.
    // 
    // Similarly imagine the blocks and grids laid out linearly. 
    // Each block consists of blockDim.x * blockDim.y threads.
    // Each grid consists of gridDim.x * gridDim.y blocks.
    // Now we take these cuda variables and map them into an "element_offset",
    // which is the element number, linearly determined.
    // 
    // This same "element number" is basically the offset in the row-major layout
    // of the input (color) and output (grayscale) images.

    int block_offset = threadIdx.y * blockDim.x + threadIdx.x;
    int grid_offset = blockIdx.y * gridDim.x + blockIdx.x;
    int block_size = blockDim.x * blockDim.y;
    int element_offset = block_offset + grid_offset * block_size;

    if (element_offset < numRows * numCols) {
        const uchar4* const d_in = &rgbaImage[element_offset];
        unsigned char* const d_out = &greyImage[element_offset];

        float grey_val = 0.299f * d_in->x + 0.587f * d_in->y + 0.114f * d_in->z;
        unsigned char value = (unsigned char)grey_val;
        *d_out = value;
    }

}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  //You must fill in the correct sizes for the blockSize and gridSize
  //currently only one block with one thread is being launched
  const dim3 blockSize(128, 8, 1);  
  const dim3 gridSize(numRows / 128 + 1, numCols / 8 + 1, 1);  

  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}
