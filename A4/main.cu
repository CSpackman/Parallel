#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>


void cpu_convultion(int *img, int *kernel, int *imgf, int Nx, int Ny, int kernal_size)
{
    int center = kernel[(kernal_size / 2)];
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
// test

__global__ void gpu_convultion(int *img, int *kernel, int *imgf, int Nx, int Ny, int kernal_size){
    // Implement the GPU convolution kernel here

}

int *init_grid(int width, int height)
{
    int *grid = new int[(width) * (height)];
    for (unsigned i = 0; i < (width * height); ++i)
    {
        grid[i] = rand();
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
    //Starting parameters
    int width = 10;
    int height = 10;
    srand(92507191);

    //Init random Grid
    int *gpu_grid, *cpu_grid;
    gpu_grid = init_grid(width, height);
    cpu_grid = gpu_grid;

    int kernal_size = 3;
    int *kernel = new int[kernal_size * kernal_size];

    int *cpu_kernal_out = new int[(width) * (height)];
    
    cpu_convultion(cpu_grid, kernel, cpu_kernal_out, width, height, kernal_size);
    gpu_grid = cpu_kernal_out;
    //Check CPU_GRID and GPU_GRID
    unsigned int mem_size_A = sizeof(int) * (width*height);
    int *space_gpu_grid, *space_cpu_grid;

    cudaMalloc((void **)&space_gpu_grid, mem_size_A);
    cudaMalloc((void **)&space_cpu_grid, mem_size_A);

    cudaMemcpy(space_gpu_grid, gpu_grid, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(space_cpu_grid, cpu_grid, mem_size_A, cudaMemcpyHostToDevice);
    // Number of blocks(vector) and number of threads per block(vector)
    check<<<(1), (height, width)>>>(space_gpu_grid, space_cpu_grid, width, height);
    cudaDeviceSynchronize();




    printf("Done\n");
    return 1;
}