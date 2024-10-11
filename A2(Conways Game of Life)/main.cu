#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

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

__global__ void calc_new_grid(int *A, int *output, int numRows, int numColumns)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int i = row * numRows + col;
    int up = (row - 1) * numRows + col;
    int down = (row + 1) * numRows + col;
    int left = row * numRows + (col - 1);
    int right = row * numRows + (col + 1);
    int count = 0;

    if (up < 0)
    {
        up = (numRows)*numRows + col;
    }
    if (down > numRows)
    {
        down = (0) * numRows + col;
    }
    if (right > numColumns)
    {
        right = (col + 1);
    }
    if (left < 0)
    {
        left = row * numRows;
    }

    if (A[up])
    {
        count++;
    }
    if (A[down])
    {
        count++;
    }
    if (A[right])
    {
        count++;
    }
    if (A[left])
    {
        count++;
    }

    if (A[i] && count > 2)
    {
        output[i] = 1;
    }
    if (!A[i] && count == 3)
    {
        output[i] = 1;
    }
    if (count < 2 && count > 3)
    {
        output[i] = 0;
    }
}

void print_grid(int *grid, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        // Print visual representation
        for (int j = 0; j < width; j++)
        {
            printf(grid[i * width + j] ? "█" : " ");
        }
        printf("    "); // Add some space between representations

        // Print 1s and 0s
        for (int j = 0; j < width; j++)
        {
            printf("%d ", grid[i * width + j]);
        }
        printf("\n");
    }
}

__global__ void myKernel()
{
    printf("Hello, world from the device!\n");
}

int main(int argc, char **argv)
{
    int GRIDWITH = 25;
    int GRIDHEIGHT = 25;
    int NUMOFGENS = 20;
    // 8 Digit seed
    srand(92007191);

    int *grid = init_grid(GRIDWITH, GRIDHEIGHT);
    int *output = new int[GRIDWITH * GRIDHEIGHT];

    // device variables
    int *deviceA;
    int *deviceB;
    // float *deviceC;

    unsigned int size_A = GRIDWITH * GRIDHEIGHT;
    unsigned int mem_size_A = sizeof(int) * size_A;

    cudaMalloc((void **)&deviceA, mem_size_A);
    cudaMalloc((void **)&deviceB, mem_size_A);

    // dim3 threads(GRIDWITH, GRIDHEIGHT, 1);
    // dim3 grid(((1 - 1) / threads.x) + 1, ((1 - 1) / threads.y) + 1, 1);

    // dim3  	dimBlock	= (16, 16, 1);

    print_grid(grid, GRIDHEIGHT, GRIDWITH);
    printf("\n \n");

    for (int i = 0; i < NUMOFGENS; i++)
    {

        cudaMemcpy(deviceA, grid, mem_size_A, cudaMemcpyHostToDevice);
        calc_new_grid<<<(25, 25), (25, 25)>>>(deviceA, deviceB, GRIDWITH, GRIDHEIGHT);
        cudaMemcpy(output, deviceB, mem_size_A, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        print_grid(output, GRIDHEIGHT, GRIDWITH);
        printf("\n \n");
        grid = output;
    }

    return 1;
}