//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

// Note: This function has been validated against the host reference.
// Computes a histogram of the number of elements with 0 at each bitpos.
__global__ void countZerosByPositions(unsigned int* const d_in, int numElems, unsigned int* d_histo, unsigned int num_bits) {
    extern __shared__ unsigned int histo[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numElems) {
        return;
    }

    // Initialize the shared mem histo.
    if (tid == 0) {
        for (int i = 0; i < num_bits; i++) {
            histo[i] = 0;
        }
    }
    __syncthreads();

    // Count 0 valued bit at each bit_pos.
    for (unsigned int mask = 0x1, bitpos = 0; mask != 0; mask <<= 1, bitpos++) {
        unsigned int my_val = d_in[gid];
        int inc_by = ((my_val & mask) == 0x0) ? 1 : 0;
        atomicAdd(&histo[bitpos], inc_by);
    }
    __syncthreads();

    // Do the global histogram computation.
    if (tid == 0) {
        for (int bitpos = 0; bitpos < num_bits; bitpos++) {
            atomicAdd(&d_histo[bitpos], histo[bitpos]);
        }
    }
}

// Note: This function has been validated against the host reference.
// pred_cmp: The val to cmp.
// pred_mask: Determines which bit position value gets compared to.
__global__ void predicateKernel(unsigned int pred_cmp, unsigned int pred_mask, unsigned int *const d_in, unsigned int * d_preds, int numElems) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < numElems) {
        d_preds[gid] = (((d_in[gid] & pred_mask) == pred_cmp) ? 1 : 0);
    }
}

// Note: Exclusive scan implementation has been verified against the host implementation!
//       This includes the local + global prefix_scan + add_scalar at the end.
// 
// Performs an exclusive prefix sum only at the block level.
// d_scan: The main input to be scanned in-place.
// d_interim: The last value for each block's prefix_sum. This will be prefix summed later outside this function.
__global__ void exclusivePrefixSum(unsigned int *const d_scan, int numElems, unsigned int* const d_interim) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numElems) {
        return;
    }
    
    // First perform the inclusive prefix sum.
    for (int step_size = 1; step_size < blockDim.x; step_size <<= 1) {
        unsigned int val_lhs = 0, val_rhs = 0;
        unsigned int block_end = (blockIdx.x + 1) * blockDim.x;
        if (gid + step_size < block_end) { 
            val_lhs = d_scan[gid];
            val_rhs = d_scan[gid + step_size]; 
        }
        __syncthreads();
        if (gid + step_size < block_end) { 
            d_scan[gid + step_size] = val_lhs + val_rhs; 
        }
        __syncthreads();
    }

    int tid = threadIdx.x;
    // Now write to d_interim if needed.
    if (tid == 0 && d_interim != nullptr) {
        int read_index = gid + blockDim.x - 1;
        read_index = min(read_index, numElems - 1);  // clamp for the last block!
        int write_index = blockIdx.x;
        d_interim[write_index] = d_scan[read_index];        
    }
    __syncthreads();

    // Now make it an exclusive prefix sum.
    unsigned int val = 0;    
    if (tid != 0) {
        val = d_scan[gid-1];
    }
    __syncthreads();
    d_scan[gid] = val;
    if (nullptr != d_interim && d_scan[gid] >= numElems) {
        // printf("exclusive prefix sum: [t: %d; b: %d] Unknown scan value: %u, numElems = %d\n", threadIdx.x, blockIdx.x, d_scan[gid], numElems); // FIXME
    }
    if (d_interim == nullptr && d_scan[threadIdx.x] >= numElems*1024) {
        printf("exclusive prefix sum: [t: %d; b: %d] interim: %u, numElems = %d\n", threadIdx.x, blockIdx.x, d_scan[threadIdx.x], numElems);
    }
    __syncthreads();
    
}

// Takes the i-th indexed value from d_interim, and adds it to the i-th block elements.
__global__ void addScalar(unsigned int* const d_scan, int numElems, unsigned int* const d_interim) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numElems) {
        return;
    }
    d_scan[gid] = d_scan[gid] + d_interim[blockIdx.x];
    if (d_scan[gid] >= numElems) {
        // printf("addScalar: [t: %d; b: %d] Unknown scan value: %u, interim=%u; numElems = %d\n", threadIdx.x, blockIdx.x, d_scan[gid], d_interim[blockIdx.x], numElems); // FIXME
    }
}

// Note this function has been tested.
// d_histo: The histogram of count of zeroes; bitpos wise.
// bitpos: The bit position we are dealing with in this iteration.
// d_in_val and d_in_pos are the inputs.
// d_out_val and d_out_pos are the outputs.
// If d_preds at index is true, then it scatters the element at that location to the location indicated in d_scan's index.
// d_histo, bitpos, and use_offset are used to offset the location (needed for 1's).
// Both val and pos are moved identically.
__global__ void scatterWithOffsetIfPred(const unsigned int * const d_histo, int bitpos, bool use_offset,
                                        const unsigned int* const d_preds, const unsigned int* const d_scan, 
                                        const unsigned int* const d_in_vals, const unsigned int* const d_in_pos,
                                        unsigned int* const d_out_vals, unsigned int* const d_out_pos,
                                        int numElems) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numElems) {
        return;
    }
    if (d_preds[gid] == 0) {
        return;
    }
    if (d_preds[gid] != 1) { // FIXME
        printf("[t: %d; b: %d] Unknown pred value: %u\n", threadIdx.x, blockIdx.x, d_preds[gid]);
    }
    if (d_scan[gid] >= numElems) {  // FIXME
        // printf("scatter: [t: %d; b: %d] Unknown scan value: %u\n", threadIdx.x, blockIdx.x, d_scan[gid]); // FIXME
    }
    unsigned int move_to_index = d_scan[gid];
    if (use_offset) {
        move_to_index += d_histo[bitpos];
    }
    if (move_to_index < numElems) {
        // FIXME: Why is d_scan[gid] negative sometimes?
        // printf("[t: %d; b: %d] move_to_index = %u (%u + %u offset) (use_offset=%d), numElems = %d\n", threadIdx.x, blockIdx.x, move_to_index, d_scan[gid], d_histo[bitpos], use_offset, numElems);  // FIXME
        d_out_vals[move_to_index] = d_in_vals[gid];
        d_out_pos[move_to_index] = d_in_pos[gid];
    }
    else {
        // printf("[t: %d; b: %d] move_to_index = %u (%u + %u offset) (use_offset=%d), numElems = %d\n", threadIdx.x, blockIdx.x, move_to_index, d_scan[gid], d_histo[bitpos], use_offset, numElems);  // FIXME
    }
    // __syncthreads(); // WHY?
}

void test_histogram(unsigned int* const d_inputVals, const size_t numElems) {
    
    unsigned int* h_vals = static_cast<unsigned int*>(malloc(sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMemcpy(h_vals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost)); 

    unsigned int histogram[32] = { 0 };

    for (int i = 0; i < numElems; i++) {
        unsigned int val = h_vals[i];
        for (unsigned int mask = 0x1, bitpos=0; mask != 0; mask <<=1, bitpos++) {
            if ((val & mask) == 0x0) {
                histogram[bitpos]++;
            }
        }        
    }

    const int NUM_THREADS = 1024;
    const int NUM_BLOCKS = numElems / NUM_THREADS + 1;
    dim3 block(NUM_THREADS);
    dim3 grid(NUM_BLOCKS);
     
    // Histogram counts the number of zeros in each bit position.
    unsigned int num_bits = sizeof(unsigned int) * 8;
    unsigned int* d_histo;
    checkCudaErrors(cudaMalloc(&d_histo, sizeof(unsigned int) * num_bits));
    checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * num_bits));
    countZerosByPositions << <block, grid, sizeof(unsigned int)* num_bits >> > (d_inputVals, numElems, d_histo, num_bits);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    unsigned int* h_histogram = static_cast<unsigned int*>(malloc(sizeof(unsigned int) * num_bits));
    checkCudaErrors(cudaMemcpy(h_histogram, d_histo, sizeof(unsigned int) * num_bits, cudaMemcpyDeviceToHost));

    bool failure = false;
    for (int i = 0; i < num_bits; i++) {
        if (h_histogram[i] != histogram[i]) {
            printf("Test: test_histogram: Mismatch for bucket %d: got %d, want %d\n", i, h_histogram[i], histogram[i]);
            failure = true;
            break;
        }
    }
    printf("Test: test_histogram: Comparing host and device histograms: %s!\n", (failure ? "FAILED":"PASSED"));
    free(h_histogram);
    free(h_vals);
}

void test_exclusiveScan() {

    // Test the exclusive prefix scan.
    const int NUM_BLOCKS = 1024;
    const int NUM_THREADS = 1024;
    const int NUM_ELEMENTS = NUM_BLOCKS * NUM_THREADS;
    unsigned int* h_test_pfsc = static_cast<unsigned int*>(malloc(sizeof(unsigned int*) * NUM_ELEMENTS));
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        h_test_pfsc[i] = 1;
    }
    unsigned int* d_test_pfsc, *d_interim;
    checkCudaErrors(cudaMalloc(&d_test_pfsc, sizeof(unsigned int) * NUM_ELEMENTS));
    checkCudaErrors(cudaMalloc(&d_interim, sizeof(unsigned int) * NUM_BLOCKS));
    checkCudaErrors(cudaMemcpy(d_test_pfsc, h_test_pfsc, sizeof(unsigned int) * NUM_ELEMENTS, cudaMemcpyHostToDevice));

    exclusivePrefixSum << <NUM_BLOCKS, NUM_THREADS >> > (d_test_pfsc, NUM_ELEMENTS, d_interim);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    exclusivePrefixSum << <1, NUM_BLOCKS >> > (d_interim, NUM_BLOCKS, nullptr);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    addScalar << <NUM_BLOCKS, NUM_THREADS >> > (d_test_pfsc, NUM_ELEMENTS, d_interim);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(h_test_pfsc, d_test_pfsc, sizeof(unsigned int) * NUM_ELEMENTS, cudaMemcpyDeviceToHost));
    bool failed = false;
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        if (h_test_pfsc[i] != i) {
            printf("FAILED!! exclusive scan at index: %d; want=%d, got=%d\n", i, i, h_test_pfsc[i]);
            failed = true;
            break;
        }
    }    
    printf("Test: test_exclusiveScan: Exclusive scan implementation on host vs. device for 1M elements: %s!\n", (failed ? "FAILED" : "PASSED"));
    
}

void test_predicateKernel() {
    const int NUM_THREADS = 1024;
    const int NUM_BLOCKS = 1024;
    const int NUM_VALUES = NUM_THREADS * NUM_BLOCKS;
    unsigned int* values = static_cast<unsigned int*>(malloc(sizeof(unsigned int) * NUM_VALUES));
    unsigned int* h_out = static_cast<unsigned int*>(malloc(sizeof(unsigned int) * NUM_VALUES));
    for (int i = 0; i < NUM_VALUES; i++) {
        values[i] = i;
    }
    unsigned int* d_values, * d_out;
    checkCudaErrors(cudaMalloc(&d_out, sizeof(unsigned int) * NUM_VALUES));
    checkCudaErrors(cudaMalloc(&d_values, sizeof(unsigned int) * NUM_VALUES));
    checkCudaErrors(cudaMemcpy(d_values, values, sizeof(unsigned int)* NUM_VALUES, cudaMemcpyHostToDevice));

    // Test bitpos 0, value 0.
    bool failed = false;
    predicateKernel << <NUM_BLOCKS, NUM_THREADS >> > (0x0, 0x1, d_values, d_out, NUM_VALUES);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(unsigned int) * NUM_VALUES, cudaMemcpyDeviceToHost));
    // Verify.    
    for (int i = 0; i < NUM_VALUES; i++) {
        if (i % 2 == 0 && h_out[i] != 1) {
            printf("Mismatch case 1.1! ");
            failed = true;
            break;
        }
        else if (i % 2 == 1 && h_out[i] != 0) {
            printf("Mismatch case 1.2! ");
            failed = true;
            break;
        }
    }

    // Test bitpos 0, value 1.
    predicateKernel << <NUM_BLOCKS, NUM_THREADS >> > (0x1, 0x1, d_values, d_out, NUM_VALUES);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(unsigned int) * NUM_VALUES, cudaMemcpyDeviceToHost));
    // Verify.    
    for (int i = 0; i < NUM_VALUES; i++) {
        if (i % 2 == 0 && h_out[i] != 0) {
            printf("Mismatch case 2.1! ");
            failed = true;
            break;
        }
        else if (i % 2 == 1 && h_out[i] != 1) {
            printf("Mismatch case 2.2! ");
            failed = true;
            break;
        }
    }
    // Test bitpos 1, value 0.
    predicateKernel << <NUM_BLOCKS, NUM_THREADS >> > (0x0, 0x2, d_values, d_out, NUM_VALUES);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(unsigned int) * NUM_VALUES, cudaMemcpyDeviceToHost));
    // Verify.    
    for (int i = 0; i < NUM_VALUES; i++) {
        if ((i / 2) % 2 == 0 && h_out[i] != 1) {
            printf("Mismatch case 3.1! ");
            failed = true;
            break;
        }
        else if ((i / 2) % 2 == 1 && h_out[i] != 0) {
            printf("Mismatch case 3.2! ");
            failed = true;
            break;
        }
    }

    // Test bitpos 1, value 1.
    predicateKernel << <NUM_BLOCKS, NUM_THREADS >> > (0x2, 0x2, d_values, d_out, NUM_VALUES);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(unsigned int) * NUM_VALUES, cudaMemcpyDeviceToHost));
    // Verify.    
    for (int i = 0; i < NUM_VALUES; i++) {
        if ((i/2) % 2 == 0 && h_out[i] != 0) {
            printf("Mismatch case 4.1! ");
            failed = true;
            break;
        }
        else if ((i/2) % 2 == 1 && h_out[i] != 1) {
            printf("Mismatch case 4.2! ");
            failed = true;
            break;
        }
    }

    printf("Test: test_predicateKernel: Collective predicate kernel tests: %s!\n", (failed ? "FAILED" : "PASSED"));

    checkCudaErrors(cudaFree(d_out));
    checkCudaErrors(cudaFree(d_values));
    free(values);
    free(h_out);
}

void test_scatter() {
    const int NUM_THREADS = 1024;
    const int NUM_BLOCKS = 1024;
    const int NUM_VALUES = NUM_THREADS * NUM_BLOCKS;
    unsigned int* h_in     = static_cast<unsigned int*>(malloc(sizeof(unsigned int) * NUM_VALUES));
    unsigned int* h_in_pos = static_cast<unsigned int*>(malloc(sizeof(unsigned int) * NUM_VALUES));
    unsigned int* h_preds  = static_cast<unsigned int*>(malloc(sizeof(unsigned int) * NUM_VALUES));
    unsigned int* h_scan   = static_cast<unsigned int*>(malloc(sizeof(unsigned int) * NUM_VALUES));    
    for (int i = 0; i < NUM_VALUES; i++) {
        h_in[i] = i;
        h_in_pos[i] = i;
        h_preds[i] = 1;
        h_scan[i] = NUM_VALUES - i - 1;
    }

    unsigned int* d_in, *d_in_pos, * d_preds, *d_scan, *d_out, *d_out_pos, * d_histo;

    checkCudaErrors(cudaMalloc(&d_in, sizeof(unsigned int) * NUM_VALUES));
    checkCudaErrors(cudaMalloc(&d_in_pos, sizeof(unsigned int) * NUM_VALUES));
    checkCudaErrors(cudaMalloc(&d_preds, sizeof(unsigned int) * NUM_VALUES));
    checkCudaErrors(cudaMalloc(&d_scan, sizeof(unsigned int) * NUM_VALUES));
    checkCudaErrors(cudaMalloc(&d_out, sizeof(unsigned int) * NUM_VALUES));
    checkCudaErrors(cudaMalloc(&d_out_pos, sizeof(unsigned int) * NUM_VALUES));   
    checkCudaErrors(cudaMalloc(&d_histo, sizeof(unsigned int) * sizeof(unsigned int)));

    checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(unsigned int) * NUM_VALUES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_in_pos, h_in_pos, sizeof(unsigned int) * NUM_VALUES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_preds, h_preds, sizeof(unsigned int) * NUM_VALUES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_scan, h_scan, sizeof(unsigned int) * NUM_VALUES, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(d_out, 0, sizeof(unsigned int) * NUM_VALUES));
    checkCudaErrors(cudaMemset(d_out_pos, 0, sizeof(unsigned int) * NUM_VALUES));
    checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * sizeof(unsigned int)));

    scatterWithOffsetIfPred<<<NUM_BLOCKS, NUM_THREADS>>>(d_histo, 0x0, true, d_preds, d_scan, d_in, d_in_pos, d_out, d_out_pos, NUM_VALUES);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    unsigned int* h_out = static_cast<unsigned int*>(malloc(sizeof(unsigned int) * NUM_VALUES));
    unsigned int* h_out_pos = static_cast<unsigned int*>(malloc(sizeof(unsigned int) * NUM_VALUES));
    checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(unsigned int) * NUM_VALUES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_out_pos, d_out_pos, sizeof(unsigned int) * NUM_VALUES, cudaMemcpyDeviceToHost));
    bool failure = false;
    for (int i = 0; i < NUM_VALUES; i++) {
        if (h_out[i] != NUM_VALUES - i - 1) {
            failure = false;
            printf("Test: scatterWithOffsetIfPred: h_out: Mismatched at index %d: want %d, got %d\n", i, NUM_VALUES-i-1, h_out[i]);
            break;
        }
        if (h_out_pos[i] != NUM_VALUES - i - 1) {
            failure = false;
            printf("Test: scatterWithOffsetIfPred: h_out_pos: Mismatched at index %d: want %d, got %d\n", i, NUM_VALUES - i - 1, h_out_pos[i]);
            break;
        }
    }
    printf("Test: test_scatter: Result: %s!\n", (failure ? "FAILED" : "PASSED"));
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
 
    test_exclusiveScan();
    test_predicateKernel();
    test_scatter();    
    test_histogram(d_inputVals, numElems);
    // return; // FIXME

    const int NUM_THREADS = 1024;    
    const int NUM_BLOCKS = numElems / NUM_THREADS + 1;
    dim3 block(NUM_THREADS);
    dim3 grid(NUM_BLOCKS);    

    // Histogram counts the number of zeros in each bit position.
    unsigned int num_bits = sizeof(unsigned int) * 8;
    unsigned int* d_histo;
    checkCudaErrors(cudaMalloc(&d_histo, sizeof(unsigned int) * num_bits));
    checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * num_bits));
    countZerosByPositions << <block, grid, sizeof(unsigned int) * num_bits >> > (d_inputVals, numElems, d_histo, num_bits);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    unsigned int* h_histo = static_cast<unsigned int *>(malloc(sizeof(unsigned int) * num_bits));
    checkCudaErrors(cudaMemcpy(h_histo, d_histo, sizeof(unsigned int) * num_bits, cudaMemcpyDeviceToHost));
    printf("Device: histo [ \n");
    for (int i = 0; i < num_bits; i++) {
        printf("[%d] = %d; \n", i, h_histo[i]);
    }
    printf("\n");

    unsigned int* d_preds, *d_scan, *d_interim;
    checkCudaErrors(cudaMalloc(&d_preds, sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMalloc(&d_scan, sizeof(unsigned int) * numElems));    
    checkCudaErrors(cudaMalloc(&d_interim, sizeof(unsigned int) * NUM_BLOCKS));

    // swapped indicates whether the input and output are temporarily swapped.
    bool swapped = false;  
    unsigned int* d_pi_val = d_inputVals;
    unsigned int* d_pi_pos = d_inputPos;
    unsigned int* d_po_val = d_outputVals;
    unsigned int* d_po_pos = d_outputPos;

    for (unsigned int mask = 0x1, bitpos=0; mask != 0; mask <<= 1, bitpos++) {
        printf("Main algorithm: Starting step for bitpos %d\n", bitpos);

        printf("Main algorithm : Zero step\n");
        // Perform predicate operations for value 0 at bitpos.
        predicateKernel << <grid, block >> > (0x0, mask, d_pi_val, d_preds, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        // Perform scan: Local prefix sum + global prefix sum + block-wide adds.
        checkCudaErrors(cudaMemcpy(d_scan, d_preds, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemset(d_interim, 0, sizeof(unsigned int) * NUM_BLOCKS));
        exclusivePrefixSum << <grid, block>> > (d_scan, numElems, d_interim);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        exclusivePrefixSum << <1, NUM_BLOCKS>> > (d_interim, NUM_BLOCKS, nullptr);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());        
        addScalar << <grid, block>> > (d_scan, numElems, d_interim);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        // Perform scatter of elements.
        scatterWithOffsetIfPred << <grid, block>> > (d_histo, bitpos, false, d_preds, d_scan, d_pi_val, d_pi_pos, d_po_val, d_po_pos, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        printf("Main algorithm : Ones step\n");
        // Perform predicate operations for value 1 at bitpos.
        predicateKernel << <grid, block >> > (mask, mask, d_pi_val, d_preds, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        // Perform scan: Local prefix sum + global prefix sum + block-wide adds.
        checkCudaErrors(cudaMemcpy(d_scan, d_preds, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemset(d_interim, 0, sizeof(unsigned int) * NUM_BLOCKS));
        exclusivePrefixSum << <grid, block >> > (d_scan, numElems, d_interim);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        exclusivePrefixSum << <1, NUM_BLOCKS >> > (d_interim, NUM_BLOCKS, nullptr);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        addScalar << <grid, block >> > (d_scan, numElems, d_interim);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        // Perform scatter of elements.
        scatterWithOffsetIfPred << <grid, block >> > (d_histo, bitpos, true, d_preds, d_scan, d_pi_val, d_pi_pos, d_po_val, d_po_pos, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        // flip the swap.
        swapped = !swapped;
        if (swapped) {
            d_pi_val = d_outputVals;
            d_pi_pos = d_outputPos;
            d_po_val = d_inputVals;
            d_po_pos = d_inputPos;
        } else {
            d_pi_val = d_inputVals;
            d_pi_pos = d_inputPos;
            d_po_val = d_outputVals;
            d_po_pos = d_outputPos;
        }
    }
    if (!swapped) {  // check for negative since it is flipped at the end of the last iteration at exit!
        checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
    }

    checkCudaErrors(cudaFree(d_preds));
    checkCudaErrors(cudaFree(d_scan));
    checkCudaErrors(cudaFree(d_interim));
    checkCudaErrors(cudaFree(d_histo));

    // Check if sorted.
    unsigned int* h_sorted = static_cast<unsigned int*>(malloc(sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMemcpy(h_sorted, d_outputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
    int count = 0;
    for (int i = 1; i < numElems; i++) {
        if (h_sorted[i-1] > h_sorted[i]) {
            // printf("Device output was not sorted at indices [%d]=%d and [%d]=%d\n", i-1, h_sorted[i-1], i, h_sorted[i]);
            count++;
            // break;
        }
    }
    free(h_sorted);
    printf("Total unsorted on device: %d\n", count);
}
