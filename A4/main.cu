#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

void cpu_convultion(int *img, int *kernel, int *imgf, int Nx, int Ny, int kernal_size)
{
    int center = (kernal_size - 1) / 2;
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

__global__ void gpu_convultion(int *img, int *kernel, int *imgf, int Nx, int Ny, int kernal_size)
{
    // Implement the GPU convolution kernel here
    int center = (kernal_size - 1) / 2;
    int iy = blockIdx.x + (kernal_size - 1) / 2;
    int ix = threadIdx.x + (kernal_size - 1) / 2;
    int idx = iy * Nx + ix;
    int K2 = kernal_size * kernal_size;
    extern __shared__ int shared_img[];

    if (threadIdx.x < K2)
    {
        shared_img[threadIdx.x] = kernel[threadIdx.x];
        __syncthreads();
    }
    if (idx < Nx * Ny)
    {
        int sum = 0;
        for (int ki = 0; ki < kernal_size; ki++)
        {
            for (int kj = 0; kj < kernal_size; kj++)
            {
                int ii = ix + kj - center;
                int jj = iy + ki - center;
                sum += img[jj * Nx + ii] * shared_img[ki * kernal_size + kj];
            }
            imgf[idx] = sum;
        }
    }
}

int *init_grid(int width, int height)
{
    int *grid = new int[(width) * (height)];
    for (unsigned i = 0; i < (width * height); ++i)
    {
        grid[i] = rand()%255;
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
    // Starting parameters
    int width = 100;
    int height = 100;
    srand(92567191);

    // Init random Grid
    int *gpu_grid, *cpu_grid;
    gpu_grid = init_grid(width, height);
    cpu_grid = new int[(width) * (height)];
    memcpy(cpu_grid, gpu_grid, ((width * height)*sizeof(int)));
    unsigned int mem_size_A = sizeof(int) * (width * height);
    

    int kernal_size = 3;
    int *kernel = init_grid(kernal_size, kernal_size);

    

    int *cpu_kernal_out = new int[(width) * (height)];

    cpu_convultion(cpu_grid, kernel, cpu_kernal_out, width, height, kernal_size);

    for (int i = 0; i < width*height; i++)
    {
            printf("First CPU output: %d\n",cpu_kernal_out[i]);
    }
    



    int *gpu_out = new int[(width) * (height)];
    int *gpu_out_device, *kernal_device, *gpu_grid_device;

    cudaMalloc((void **)&gpu_out_device, mem_size_A);
    cudaMalloc((void **)&kernal_device, sizeof(int) * (kernal_size * kernal_size));
    cudaMalloc((void **)&gpu_grid_device, mem_size_A);

    cudaMemcpy(gpu_grid_device, gpu_grid, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(kernal_device, kernel, sizeof(int) * (kernal_size * kernal_size), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_out_device, gpu_out, mem_size_A, cudaMemcpyHostToDevice);
    
    gpu_convultion<<<(1*1), (height * width)>>>(gpu_grid_device, kernal_device, gpu_out_device, width, height, kernal_size);
    cudaMemcpy(gpu_grid, gpu_out_device, mem_size_A, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Check CPU_GRID and GPU_GRID

    for (int i = 0; i < width*height; i++)
    {
            printf("CPU,GPU: %d,%d\n",cpu_kernal_out[i], gpu_grid[i]);
    }
    

    // int *space_gpu_grid, *space_cpu_grid;

    // cudaMalloc((void **)&space_gpu_grid, mem_size_A);
    // cudaMalloc((void **)&space_cpu_grid, mem_size_A);

    // cudaMemcpy(space_gpu_grid, gpu_grid, mem_size_A, cudaMemcpyHostToDevice);
    // cudaMemcpy(space_cpu_grid, cpu_kernal_out, mem_size_A, cudaMemcpyHostToDevice);
    // // Number of blocks(vector) and number of threads per block(vector)
    // check<<<(1), (height, width)>>>(space_gpu_grid, space_cpu_grid, width, height);
    // cudaDeviceSynchronize();

    printf("Done\n");
    return 1;
}