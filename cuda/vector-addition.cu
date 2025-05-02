#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define N (256*256*256)
#define BLOCK_SIZE 256

__global__
void vecadd(float *x, float *y, float *z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = x[i] + y[i];
    }
}

int main() {
    int n = N; // N = 256*256*256
    float *h_x = (float*) malloc(n * sizeof(float));
    float *h_y = (float*) malloc(n * sizeof(float));
    float *h_z = (float*) malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        h_x[i] = drand48();
        h_y[i] = 1.0f - h_x[i];  // So that x + y = 1.0
    }

    float *d_x, *d_y, *d_z;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_z, n * sizeof(float));

    cudaEvent_t start, stop;
    float time_with_copy = 0.0f, time_kernel_only = 0.0f;

    // Create CUDA events for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);



    // ----------- Start timer for copy + kernel ---------------
    cudaEventRecord(start, 0);

    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vecadd<<<numBlocks, BLOCK_SIZE>>>(d_x, d_y, d_z, n);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    const char *msg = cudaGetErrorName(err);
    printf("error = |%s|\n", msg);

    cudaMemcpy(h_z, d_z, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    // ------------ Stop timer for copy + kernel ----------------------





    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_with_copy, start, stop);

    // ------------- Memory Copy -------------------------------
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);




    // ------------- Start timer for kernel only ----------------------------
    cudaEventRecord(start, 0);

    vecadd<<<numBlocks, BLOCK_SIZE>>>(d_x, d_y, d_z, n);
    cudaDeviceSynchronize();
    printf("error = |%s|\n", msg);
    cudaEventRecord(stop, 0);
    // ------------- Stop timer for kernel -------------------------------




    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_kernel_only, start, stop);

    // Copy back again to make sure kernel ran
    cudaMemcpy(h_z, d_z, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute max absolute error
    float maxerr = 0.0f;
    for (int i = 0; i < n; i++) {
        maxerr = fmaxf(maxerr, fabs(1.0f - h_z[i]));
    }

    std::cout << "Max absolute error: " << maxerr << std::endl;
    std::cout << "Time (kernel only): " << time_kernel_only << " ms" << std::endl;
    std::cout << "Time (with copy): " << time_with_copy << " ms" << std::endl;

    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
