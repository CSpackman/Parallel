#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include <cuda_runtime.h>

#define SOFTENING 1e-9f

typedef struct
{
    float x, y, z, vx, vy, vz;
} Body;

// CUDA error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void randomizeBodies(float *data, int n)
{
    // Initialize body positions and velocities with random values
    // This function populates the data array with random float values between -1 and 1
    for (int i = 0; i < n; i++)
    {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

__global__ void bodyForceKernel(Body *p, float dt, int n)
{
    // CUDA kernel to calculate forces and update velocities for each body
    // Each thread handles one body's calculations
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

        // Calculate forces from all other bodies
        for (int j = 0; j < n; j++)
        {
            // Calculate distance between bodies
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            // Use softening to avoid division by zero
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            // Accumulate forces
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        // Update velocities based on calculated forces
        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}

__global__ void integratePositions(Body *p, float dt, int n)
{
    // CUDA kernel to update positions based on velocities
    // Each thread updates one body's position
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

int main(const int argc, const char **argv)
{
    int nBodies = 30000;
    if (argc > 1) 
        nBodies = atoi(argv[1]);

    const float dt = 0.01f; // time step
    const int nIters = 10;  // simulation iterations

    int bytes = nBodies * sizeof(Body);
    float *buf = (float *)malloc(bytes);
    Body *p = (Body *)buf;

    randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

    // Print initial positions and velocities of the first 3 bodies
    printf("Initial state:\n");
    for (int i = 0; i < 3 && i < nBodies; i++) {
        printf("Body %d: pos(%.3f, %.3f, %.3f) vel(%.3f, %.3f, %.3f)\n", i,
               p[i].x, p[i].y, p[i].z, p[i].vx, p[i].vy, p[i].vz);
    }

    // CUDA setup: Allocate memory on GPU and create streams for concurrent execution
    Body *d_bodies;
    cudaCheckError(cudaMalloc(&d_bodies, bytes));

    // Create CUDA streams for potential concurrent kernel execution
    cudaStream_t stream1, stream2;
    cudaCheckError(cudaStreamCreate(&stream1));
    cudaCheckError(cudaStreamCreate(&stream2));

    // Set up CUDA events for timing the simulation
    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));

    double totalTime = 0.0;

    // Copy data to GPU
    cudaCheckError(cudaMemcpyAsync(d_bodies, p, bytes, cudaMemcpyHostToDevice, stream1));

    for (int iter = 1; iter <= nIters; iter++)
    {
        cudaCheckError(cudaEventRecord(start, 0));

        int threadsPerBlock = 256;
        int blocksPerGrid = (nBodies + threadsPerBlock - 1) / threadsPerBlock;

        bodyForceKernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_bodies, dt, nBodies);
        integratePositions<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_bodies, dt, nBodies);

        cudaCheckError(cudaEventRecord(stop, 0));
        cudaCheckError(cudaEventSynchronize(stop));

        float milliseconds = 0;
        cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));

        if (iter > 1) { // First iter is warm up
            totalTime += milliseconds / 1000.0;
        }

#ifndef SHMOO
        printf("Iteration %d: %.3f seconds\n", iter, milliseconds / 1000.0);
#endif
    }

    double avgTime = totalTime / (double)(nIters - 1);

    // Copy data back to CPU
    cudaCheckError(cudaMemcpyAsync(p, d_bodies, bytes, cudaMemcpyDeviceToHost, stream1));

    // Print final positions and velocities of the first 3 bodies
    printf("\nFinal state:\n");
    for (int i = 0; i < 3 && i < nBodies; i++) {
        printf("Body %d: pos(%.3f, %.3f, %.3f) vel(%.3f, %.3f, %.3f)\n", i,
               p[i].x, p[i].y, p[i].z, p[i].vx, p[i].vy, p[i].vz);
    }

#ifdef SHMOO
    printf("%d, %0.3f\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
#else
    printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
#endif

    // Cleanup: Free allocated memory and destroy CUDA streams and events
    free(buf);
    cudaCheckError(cudaFree(d_bodies));
    cudaCheckError(cudaStreamDestroy(stream1));
    cudaCheckError(cudaStreamDestroy(stream2));
    cudaCheckError(cudaEventDestroy(start));
    cudaCheckError(cudaEventDestroy(stop));

    return 0;
}
