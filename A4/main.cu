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
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            sum = 0;
            for (int ki = 0; ki < kernal_size; ki++)
            {
                for (int kj = 0; kj < kernal_size; kj++)
                {
                    ii = j + kj - center;
                    jj = i + ki - center;
                    if (ii >= 0 && ii < Nx && jj >= 0 && jj < Ny) {
                        int img_value = img[jj * Nx + ii];
                        int kernel_value = kernel[ki * kernal_size + kj];
                        sum += img_value * kernel_value;
                    }
                }
            }
            imgf[i * Nx + j] = sum;
        }
    }
}

__global__ void gpu_convultion(int *img, int *kernel, int *imgf, int Nx, int Ny, int kernal_size)
{
    int center = (kernal_size - 1) / 2;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= Nx || iy >= Ny) return;

    int sum = 0;
    for (int ki = 0; ki < kernal_size; ki++)
    {
        for (int kj = 0; kj < kernal_size; kj++)
        {
            int ii = ix + kj - center;
            int jj = iy + ki - center;
            if (ii >= 0 && ii < Nx && jj >= 0 && jj < Ny) {
                int img_value = img[jj * Nx + ii];
                int kernel_value = kernel[ki * kernal_size + kj];
                sum += img_value * kernel_value;
            }
        }
    }
    imgf[iy * Nx + ix] = sum;
}

int *init_grid(int width, int height)
{
    int *grid = new int[width * height];
    for (unsigned i = 0; i < width * height; ++i)
    {
        grid[i] = rand() % 255;
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
    int width = 5;
    int height = 5;
    srand(92567191);

    int *gpu_grid, *cpu_grid;
    gpu_grid = init_grid(width, height);
    cpu_grid = new int[width * height];
    memcpy(cpu_grid, gpu_grid, width * height * sizeof(int));

    int kernal_size = 3;
    int *kernel = init_grid(kernal_size, kernal_size);

    int *cpu_kernal_out = new int[width * height];
    cpu_convultion(cpu_grid, kernel, cpu_kernal_out, width, height, kernal_size);

    int *gpu_out = new int[width * height];
    int *gpu_out_device, *kernal_device, *gpu_grid_device;
    cudaMalloc((void **)&gpu_out_device, width * height * sizeof(int));
    cudaMalloc((void **)&kernal_device, kernal_size * kernal_size * sizeof(int));
    cudaMalloc((void **)&gpu_grid_device, width * height * sizeof(int));

    cudaMemcpy(gpu_grid_device, gpu_grid, width * height * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(kernal_device, kernel, kernal_size * kernal_size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gpu_convultion<<<numBlocks, threadsPerBlock>>>(gpu_grid_device, kernal_device, gpu_out_device, width, height, kernal_size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpu_out, gpu_out_device, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    printf("CPU Output:\n");
    for (int i = 0; i < width * height; i++)
    {
        printf("CPU[%d]: %d\n", i, cpu_kernal_out[i]);
    }

    printf("GPU Output:\n");
    for (int i = 0; i < width * height; i++)
    {
        printf("GPU[%d]: %d\n", i, gpu_out[i]);
    }

    check<<<(width * height + 255) / 256, 256>>>(gpu_out_device, cpu_kernal_out, width, height);
    cudaDeviceSynchronize();

    printf("Done\n");

    cudaFree(gpu_out_device);
    cudaFree(kernal_device);
    cudaFree(gpu_grid_device);
    delete[] gpu_grid;
    delete[] cpu_grid;
    delete[] kernel;
    delete[] cpu_kernal_out;
    delete[] gpu_out;

    return 0;
}
