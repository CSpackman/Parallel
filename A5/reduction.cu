#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#include <time.h>

void init_random(int *input, int n)
{
    for (int i = 0; i < n; i++)
    {
        input[i] = 1;
    }
}

void cpu_reduction_a(int *input, int n){
    for (int stride = 1; stride < n; stride *= 2) {
        for(int j = 0; j < n - stride; j += (stride * 2)){
            input[j] += input[j + stride];
        }
    }
}

void cpu_reduction_b(int *input, int n){
    int prev_stride = n;

    for (int stride = n/2; stride > 0; stride /= 2) {
        for(int j = 0; j < stride; j ++){
            input[j] += input[j + stride];
        }

        if ((prev_stride - stride) % 2 != 0 && stride != 1) {
            input[stride - 1] += input[stride];
        }
        prev_stride = stride;
    }
}

__global__ void reductionA(int *array, int *result, int n)
{
    extern __shared__ int sdata[];
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // load elements from global memory into shared memory, if within bounds
    if (global_index < n) {
        sdata[tid] = array[global_index];
    } else {
        sdata[tid] = 0; // padding out-of-bounds elements with zero
    }
    __syncthreads();

    // perform reduction within the shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); // synchronize to make sure all threads complete the step
    }

    // write the result of each block's reduction to the output array
    if (tid == 0) {
        // output[blockIdx.x] = sdata[0];
        atomicAdd(result, sdata[0]);
    }
}

__global__ void reductionB(int *array)
{
    __shared__ int partialSum[];
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // load elements from global memory into shared memory, if within bounds
    if (global_index < n) {
        partialSum[tid] = array[global_index];
    } else {
        partialSum[tid] = 0; // padding out-of-bounds elements with zero
    }
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            partialSum[tid] += partialSum[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        array[0] = partialSum[0];
    }
}

void cpu_scan(int *input, int *output, int n) {
    output[0] = 0;
    for (int i = 1; i < n; i++) {
        output[i] = input[i - 1] + output[i - 1];      
    }
}


__global__ void segmentScanKernel(int *d_out, const int *d_in, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int offset = 1;

    temp[2 * tid] = d_in[2 * tid];
    temp[2 * tid + 1] = d_in[2 * tid + 1];

    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (tid == 0) {
        temp[n - 1] = 0;
    }

    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    d_out[2 * tid] = temp[2 * tid];
    d_out[2 * tid + 1] = temp[2 * tid + 1];
}

int main()
{

    int *array, *array2;
    int n = 65536;
    array = new int[n];
    array2 = new int [n];

    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    
    struct timespec begin, end;
	double elapsed;

    // cpu_reduction_a test
    init_random(array, n);

    clock_gettime(CLOCK_MONOTONIC, &begin);
    cpu_reduction_a(array, n);
	clock_gettime(CLOCK_MONOTONIC, &end);

	elapsed = end.tv_sec - begin.tv_sec;
	elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
	printf("CPU Reduction A Elapsed Time: %f \n", elapsed);
    std::cout << "CPU Reduction A Result: " << array[0] << std::endl;

    // cpu_reduction_b test
    init_random(array, n);
    clock_gettime(CLOCK_MONOTONIC, &begin);
    cpu_reduction_b(array, n);
	clock_gettime(CLOCK_MONOTONIC, &end);

	elapsed = end.tv_sec - begin.tv_sec;
	elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;  
    printf("CPU Reduction B Elapsed Time: %f \n", elapsed);
    std::cout << "CPU Reduction B Result: " << array[0] << std::endl;

    // REDUCTION A
    int *device_array, *device_out;
    init_random(array, n);
    cudaMalloc((void **)&device_array, n * sizeof(int));
    cudaMalloc((void **)&device_out, numBlocks * sizeof(int));

    cudaMemcpy(device_array, array, n * sizeof(int), cudaMemcpyHostToDevice);
    
    clock_gettime(CLOCK_MONOTONIC, &begin);

    reductionA<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(device_array, device_out, n);
    cudaDeviceSynchronize();
    cudaMemcpy(array, device_array, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(array2, device_out, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

	clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = end.tv_sec - begin.tv_sec;
	elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
    printf("GPU Reduction A Elapsed Time: %f \n", elapsed);
    std::cout << "GPU Reduction A Result: " << array2[0] << std::endl;
    // std::cout << "GPU Reduction A Result: ";
    // for (int i = 0; i < n; ++i) {
    //     std::cout << array[i] << " ";
    // }
    // std::cout << std::endl;

    // REDUCTION B
    init_random(array, n);
    cudaMemcpy(device_array, array, n * sizeof(int), cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC, &begin);

    reductionB<<<1, n>>>(device_array);
    cudaDeviceSynchronize();
    cudaMemcpy(array, device_array, n * sizeof(int), cudaMemcpyDeviceToHost);

	clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = end.tv_sec - begin.tv_sec;
	elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
    printf("GPU Reduction B Elapsed Time: %f \n", elapsed);
    std::cout << "GPU Reduction B Result: " << array[0] << std::endl;

    // cpu_scan test
    init_random(array, n);
    clock_gettime(CLOCK_MONOTONIC, &begin);
    cpu_scan(array, array2, n);
	clock_gettime(CLOCK_MONOTONIC, &end);

	elapsed = end.tv_sec - begin.tv_sec;
	elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;    
    printf("CPU Scan Elapsed Time: %f \n", elapsed);
    std::cout << "CPU Scan Result: " << array2[n-1] << std::endl;

    // std::cout << "CPU Scan Result: ";
    // for (int i = 0; i < n; i++){
    //     std::cout << array2[i] << " ";
    // }
    // std::cout << std::endl;

    // gpu scan test
    int *d_in, *d_out;
    
    init_random(array, n);
    cudaMalloc((void**)&d_in, n * sizeof(int));
    cudaMalloc((void**)&d_out, n * sizeof(int));

    clock_gettime(CLOCK_MONOTONIC, &begin);

    cudaMemcpy(d_in, array, n * sizeof(int), cudaMemcpyHostToDevice);
    segmentScanKernel<<<1, n / 2, n * sizeof(int)>>>(d_out, d_in, n);
    cudaDeviceSynchronize();
    cudaMemcpy(array, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

	clock_gettime(CLOCK_MONOTONIC, &end);
	elapsed = end.tv_sec - begin.tv_sec;
	elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;    
    printf("GPU Scan Elapsed Time: %f \n", elapsed);
    std::cout << "GPU Scan Result: " << array[n-1] << std::endl;

    // std::cout << "GPU Segment scan result: ";
    // for (int i = 0; i < n; ++i) {
    //     std::cout << array[i] << " ";
    // }
    // std::cout << std::endl;
    
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(device_array);

    return 0;
}
