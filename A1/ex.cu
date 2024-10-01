#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

#define TILE_WIDTH 16

//
// Matrix Multiplication CPU for error checking
//
void matrixmult(float *fa, float *fb, float *fc, int Hight, int Width)
{
	int row, col, k;
	float Pvalue = 0;
	for (row = 0; row < Hight; row++)
	{
		for (col = 0; col < Width; col++)
		{
			Pvalue = 0;
			for (k = 0; k < Width; k++)
			{
				Pvalue += fa[row * Width + k] * fb[k * Width + col];
			}
			fc[row * Width + col] = Pvalue;
		}
	}
}

// Compute C=A*B in GPU non shared memory
//  Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C,
							   int numARows, int numAColumns,
							   int numBRows, int numBColumns,
							   int numCRows, int numCColumns)
{
	// TODO
}

void initialize(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		data[i] = .5 * (i + 1);
}

int main(int argc, char **argv)
{

	int numARows = 10000;			   // number of rows in the matrix A
	int numAColumns = 10000;		   // number of columns in the matrix A
	int numBRows = 10000;			   // number of rows in the matrix B
	int numBColumns = 10000;		   // number of columns in the matrix B
	int numCRows = numARows;	   // number of rows in the matrix C (you have to set this)
	int numCColumns = numBColumns; // number of columns in the matrix C (you have to set this)

	struct timespec begin, end;
	double elapsed;

	// check if you can do the MM
	if (numAColumns != numBRows)
	{
		printf("This matrix cannot be multiplied");
		return -1;
	}
	// alloc memory
	float *hostA = new float[numARows * numAColumns];
	initialize(hostA, numARows * numAColumns);
	float *hostB = new float[numBRows * numBColumns];
	initialize(hostB, numBRows * numBColumns);
	float *hostC = new float[numCRows * numCColumns]; // The output C matrix
	// do MM on CPU for timing
	clock_gettime(CLOCK_MONOTONIC, &begin);

	// matrixmult(hostA, hostB, hostC, numARows, numAColumns);

	clock_gettime(CLOCK_MONOTONIC, &end);

	elapsed = end.tv_sec - begin.tv_sec;
	elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
	printf("CPU Elapsed Time: %f \n", elapsed);


	// print MM result
	int row, col;
	/*
	for (row=0; row<numCRows; row++){
		  for(col=0; col<numCColumns; col++) {
			  printf("%lf ",hostC[row*numCColumns+col]);
		  }
		  printf("\n");
	}
	printf("\n");
	*/

	// device variables
	float *deviceA;
	float *deviceB;
	float *deviceC;

	//@@ Allocate GPU memory here
	unsigned int size_A = numARows * numAColumns;
	unsigned int mem_size_A = sizeof(float) * size_A;
	unsigned int size_B = numBRows * numBColumns;
	unsigned int mem_size_B = sizeof(float) * size_B;
	unsigned int size_C = numCRows * numCColumns;
	unsigned int mem_size_C = sizeof(float) * size_C;

	cudaMalloc((void **)&deviceA, mem_size_A);
	cudaMalloc((void **)&deviceB, mem_size_B);
	cudaMalloc((void **)&deviceC, mem_size_C);

	//@@ Copy memory to the GPU here
	cudaMemcpy(deviceA, hostA, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, mem_size_B, cudaMemcpyHostToDevice);

	clock_gettime(CLOCK_MONOTONIC, &begin);

	//@@ Initialize the grid and block dimensions here
	dim3 threads(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 grid(((numCColumns - 1) / threads.x) + 1, ((numCRows - 1) / threads.y) + 1, 1);
	matrixMultiply<<<grid, threads>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
	cudaDeviceSynchronize();
	// Copy the GPU memory back to the CPU here
	cudaMemcpy(hostC, deviceC, mem_size_C, cudaMemcpyDeviceToHost);
	// some test print MM result
	//    for (row=0; row<numCRows; row++){
	//  		for(col=0; col<numCColumns; col++) {
	//  			printf("%lf ",hostC[row*numCColumns+col]);
	//  		}
	//  		printf("\n");
	//    }

	clock_gettime(CLOCK_MONOTONIC, &end);

	elapsed = end.tv_sec - begin.tv_sec;
	elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
	printf("GPU Elapsed Time: %f \n", elapsed);

	//@@ Free the GPU memory here
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);

	free(hostA);
	free(hostB);
	free(hostC);

	return 0;
}