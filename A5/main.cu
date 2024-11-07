#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#include <time.h>

__global__ void reductionA(int *array)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (i < s)
        {
            array[i] += array[i + s];
        }
        __syncthreads();
    }
}

__global__ void reductionB(int *array)
{
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int partialSum[1024];
    partialSum[tid] = array[i];
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

    int *array;
    int *result;
    int size = 1024;
    array = new int[size];

    for (int i = 0; i < size; i++)
    {
        // array[i] = rand() % 255;
        array[i] = 1;
    }

    // REDUCTION A
    int *device_array, *device_result;
    cudaMalloc(&device_array, size * sizeof(int));
    cudaMemcpy(device_array, array, size * sizeof(int), cudaMemcpyHostToDevice);
    reductionA<<<1, size>>>(device_array);
    cudaDeviceSynchronize();
    cudaMemcpy(array, device_array, size * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Reduction A Result: " << array[0] << std::endl;

    for (int i = 0; i < size; i++)
    {
        // array[i] = rand() % 255;
        array[i] = 1;
    }
    cudaMemcpy(device_array, array, size * sizeof(int), cudaMemcpyHostToDevice);
    reductionB<<<1, size>>>(device_array);
    cudaDeviceSynchronize();
    cudaMemcpy(array, device_array, size * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Reduction B Result: " << array[0] << std::endl;

    // SEGMENT SCAN
    for (int i = 0; i < size; i++)
    {
        // array[i] = rand() % 255;
        array[i] = i;
    }

    int *d_in, *d_out;
    cudaMalloc((void**)&d_in, size * sizeof(int));
    cudaMalloc((void**)&d_out, size * sizeof(int));

    cudaMemcpy(d_in, array, size * sizeof(int), cudaMemcpyHostToDevice);

    segmentScanKernel<<<1, size / 2, size * sizeof(int)>>>(d_out, d_in, size);

    cudaDeviceSynchronize();
    cudaMemcpy(array, d_out, size * sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "Segment scan result: ";
    for (int i = 0; i < size; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
    
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(device_array);
    cudaFree(device_result);

    return 0;
}