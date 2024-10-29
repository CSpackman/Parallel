#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

// CUDA kernel function
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void cpu_convultion(float *img, float *kernel, float *imgf, int Nx, int Ny, int kernal_size)
{
    float center = kernel[(kernal_size / 2)];
    int sum, ii, jj;
    for (int i = center; i < (Ny - center); i++)
    {
        for (int j = center; j < (Nx - center); j++)
        {
            sum = 0;
            for (int ki = 0; ki < kernal_size; ki++)
            {
                for (int kj = 0; kj < kernal_size; kj++)
                {
                    ii = j + kj - center;
                    jj = i + ki - center;
                    sum += img[jj * Nx + ii] * kernel[ki * kernal_size + kj];
                }
            }
            imgf[i * Nx + j] = sum;
        }
    }
}

int *init_grid(int width, int height)
{
    // Plus one to account for ghost corners/rows/cols
    int *grid = new int[(width) * (height)];
    for (unsigned i = 0; i < (width * height); ++i)
    {
        grid[i] = rand() % 2;
    }
    return grid;
}

__global__ void check(int *a, int *b, int width, int height)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < width * height)
    {
        if (a[index] != b[index])
        {
            printf("Mismatch at index %d: CPU value = %d, GPU value = %d\n", index, a[index], b[index]);
            assert(0);
        }
    }
}

int main()
{
    int width = 10;
    int height = 10;
    srand(92507191);
    int *gpu_grid, *cpu_grid;
    gpu_grid = init_grid(width, height);
    cpu_grid = gpu_grid;
    
    unsigned int mem_size_A = sizeof(int) * (width*height);
    int *space_gpu_grid, *space_cpu_grid;
    cudaMalloc((void **)&space_gpu_grid, mem_size_A);
    cudaMalloc((void **)&space_cpu_grid, mem_size_A);

    cudaMemcpy(space_gpu_grid, gpu_grid, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(space_cpu_grid, cpu_grid, mem_size_A, cudaMemcpyHostToDevice);
    check<<<(1, 1), (height, width)>>>(space_gpu_grid, space_cpu_grid, width, height);
    cudaDeviceSynchronize();

    printf("Done\n");
    return 1;
}