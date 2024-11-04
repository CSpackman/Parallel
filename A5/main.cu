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
    return 0;
}