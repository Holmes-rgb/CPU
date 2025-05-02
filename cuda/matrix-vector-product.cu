#include <chrono>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define N 5000
#define BLOCK_SIZE 256

__global__
void mxv(float *M, float *x, float *y, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        float sum = 0.0f;
        int idx = row * n;
        for (int col = 0; col < n; col++) {
            sum += M[idx + col] * x[col];
        }
        y[row] = sum;
    }
}

void mxv_cpu(float *M, float *x, float *y, int n) {
    for (int row = 0; row < n; row++) {
        float sum = 0.0f;
        int idx = row * n;
        for (int col = 0; col < n; col++) {
            sum += M[idx + col] * x[col];
        }
        y[row] = sum;
    }
}

// Relative error function
float compute_relative_error(float *a, float *b, int n) {
    float norm_diff = 0.0f;
    float norm_b = 0.0f;
    for (int i = 0; i < n; i++) {
        norm_diff += (a[i] - b[i]) * (a[i] - b[i]);
        norm_b += b[i] * b[i];
    }
    return std::sqrt(norm_diff) / std::sqrt(norm_b);
}

int main() {
    const int n = N;

    // alocate memory
    float *M, *x, *y_gpu, *y_cpu;
    cudaMallocManaged(&M, n * n * sizeof(float));
    cudaMallocManaged(&x, n * sizeof(float));
    cudaMallocManaged(&y_gpu, n * sizeof(float));
    cudaMallocManaged(&y_cpu, n * sizeof(float));

    for (int i = 0; i < n * n; i++) {
        M[i] = drand48();
    }

    for (int i = 0; i < n; i++) {
        x[i] = drand48();
        y_gpu[i] = 0.0f;
        y_cpu[i] = 0.0f;
    }

    //-------- GPU Timing -------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mxv<<<numBlocks, BLOCK_SIZE>>>(M, x, y_gpu, n);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    const char *msg = cudaGetErrorName(err);
    printf("error = |%s|\n", msg);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuTime = 0.0f;
    cudaEventElapsedTime(&gpuTime, start, stop);

    //---------- CPU Timing -------------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    mxv_cpu(M, x, y_cpu, n);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;

    float relative_error = compute_relative_error(y_cpu, y_gpu, n);

    std::cout << "GPU time: " << gpuTime << " ms\n";
    std::cout << "CPU time: " << cpu_duration.count() << " ms\n";
    std::cout << "Relative error: " << relative_error << std::endl;

    // Free memory
    cudaFree(M);
    cudaFree(x);
    cudaFree(y_gpu);
    cudaFree(y_cpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}