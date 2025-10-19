#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M && col < K){
		float summ = 0.0;
		for (int i = 0; i < N; i++){
			summ += A[row*N + i] * B[i*K + col];
		}
		C[row*K + col] = summ;
	}

}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

int main() {
    int M = 3162;  // Rows of A and C
    int N = 3162;  // Columns of A and rows of B
    int K = 3162;  // Columns of B and C

    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    std::cout << "Allocating matrices with ~" << (M * N) << " elements each...\n";

    // Allocate host memory
    std::vector<float> h_A(M * N);
    std::vector<float> h_B(N * K);
    std::vector<float> h_C(M * K, 0);

    // Fill matrices with random floats
    for (int i = 0; i < M * N; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < N * K; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    // Copy host to device
    cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice);

    // Measure GPU execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    solve(d_A, d_B, d_C, M, N, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Matrix multiplication completed in " << milliseconds << " ms.\n";

    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    // Print first few results
    std::cout << "Sample output (first 5 values): ";
    for (int i = 0; i < 5; ++i) std::cout << h_C[i] << " ";
    std::cout << "\n";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
