#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int col_indx = blockIdx.x * blockDim.x  + threadIdx.x;
    int row_indx = blockIdx.y * blockDim.y + threadIdx.y;

    if (row_indx < rows && col_indx < cols) {
        output[col_indx * rows + row_indx] = input[row_indx * cols + col_indx];
    } 
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

int main() {
    // Create a matrix with around 10 million elements
    const int rows = 2000;
    const int cols = 5000;  // 2000 * 5000 = 10,000,000 elements
    const int N = rows * cols;

    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_input = new float[N];
    float *h_output = new float[N];

    // Initialize input matrix
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i % 1000) * 0.1f;
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy input matrix to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Measure GPU execution time
    auto start = std::chrono::high_resolution_clock::now();
    solve(d_input, d_output, rows, cols);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> gpu_time = end - start;

    // Copy output back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print performance and a few values
    std::cout << "Matrix transpose of size " << rows << "x" << cols << " completed.\n";
    std::cout << "GPU execution time: " << gpu_time.count() << " seconds\n";
    std::cout << "Output[0] = " << h_output[0] << ", Output[last] = " << h_output[N - 1] << "\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    return 0;
}
