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
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int i = row * width + col;
    int
}

int main()
{
    srand(92507191);
    int *gpu_grid, *cpu_grid;
    gpu_grid,cpu_grid = init_grid(10, 10);


    printf("Hello, CUDA!\n");
    return 1;
}