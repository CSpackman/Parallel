#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

void init_random(int *input, int n)
{
    for (int i = 0; i < n; i++)
    {
        // input[i] = rand() % 10 + 1;
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
    int n = 1024;
    array = new int[n];

    // cpu_reduction_a test
    init_random(array, n)
    cpu_reduction_a(array, n);
    std::cout << "CPU Reduction A Result: " << array[0] << std::endl;

    // cpu_reduction_b test
    init_random(array, n)
    cpu_reduction_a(array, n);
    std::cout << "CPU Reduction B Result: " << array[0] << std::endl;

    // REDUCTION A
    int *device_array, *device_result;
    init_random(array, n);
    cudaMalloc(&device_array, n * nof(int));
    cudaMemcpy(device_array, array, n * nof(int), cudaMemcpyHostToDevice);
    reductionA<<<1, n>>>(device_array);
    cudaDeviceSynchronize();
    cudaMemcpy(array, device_array, n * nof(int), cudaMemcpyDeviceToHost);
    std::cout << "GPU Reduction A Result: " << array[0] << std::endl;

    // REDUCTINO B
    init_random(array, n);
    cudaMemcpy(device_array, array, n * nof(int), cudaMemcpyHostToDevice);
    reductionB<<<1, n>>>(device_array);
    cudaDeviceSynchronize();
    cudaMemcpy(array, device_array, n * nof(int), cudaMemcpyDeviceToHost);
    std::cout << "GPU Reduction B Result: " << array[0] << std::endl;
    return 0;
}
