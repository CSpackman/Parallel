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
    for (int stride = n/2; stride > 0; stride /= 2) {
        for(int j = 0; j < stride; j ++){
            input[j] += input[j + stride];
        }
        if (n % 2 != 0 && stride % 2 != 0) {
            input[0] += input[stride];
        }

        for (int i = 0; i < n; i++) {
            printf("%d ", input[i]);
        }
        printf("\n");
    }
}

__global__ void reduction_a(int *input, int n){

}

__global__ void reduction_b(int *input, int n){

}

int main(const int argc, const char **argv)
{
    int N = 10;

    // create cpu array, init with random
    int *host_array = new int[N];
    init_random(host_array, N);

    // create gpu array with cudaMalloc
    int *device_array;
    cudaMalloc((void **)&device_array, N * sizeof(int));

    // copy data from host to device
    cudaMemcpy(device_array, host_array, N * sizeof(int), cudaMemcpyHostToDevice); 

    // cpu_reduction_a test
    printf("original array a: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", host_array[i]);
    }
    printf("\n");

    cpu_reduction_a(host_array, N);
    
    printf("reduced array a: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", host_array[i]);
    }
    printf("\n\n");

    // cpu_reduction_b test
    init_random(host_array, N);

        printf("original array b: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", host_array[i]);
    }
    printf("\n");

    cpu_reduction_b(host_array, N);
    
    printf("reduced array b: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", host_array[i]);
    }
    printf("\n\n");

    // gpu_reduction_a test

    // gpu_reduction_b test

    // segment scan

    // free memory
    delete[] host_array;
    cudaFree(device_array);

    return 0;
}
