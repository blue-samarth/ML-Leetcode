#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void vector_add(const float* A, const float*B, float*C, int N) {
	int indx = blockIdx.x * blockDim.x + threadIdx.x;
	if (indx < N) {
		C[indx] = A[indx] + B[indx];
	}
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

int main() {
    const int N = 10000000;  // 10 million elements
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = i * 0.5f;
        h_B[i] = i * 1.5f;
    }

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Measure GPU execution time
    auto start = std::chrono::high_resolution_clock::now();
    solve(d_A, d_B, d_C, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end - start;

    // Copy result back
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "GPU execution time: " << gpu_time.count() << " seconds\n";
    std::cout << "C[0] = " << h_C[0] << ", C[N-1] = " << h_C[N-1] << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
