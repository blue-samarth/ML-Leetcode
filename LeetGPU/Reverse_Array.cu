#include <cuda_runtime.h>
#include <iostream>
#include <chrono>


__global__ void reverse_array(float* input, int N) {
    int indx = blockIdx.x * blockDim.x + threadIdx.x;

    if (indx < N/2) {
        float temp = input[indx];
        input[indx] = input[N - indx - 1];
        input[N - indx - 1] = temp;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}

int main() {
    const int N = 10000000;  // 10 million elements

    // Allocate host memory
    float* h_input = new float[N];

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_input[i] = i * 1.0f;
    }

    // Allocate GPU memory
    float* d_input;
    cudaMalloc(&d_input, N * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Measure GPU execution time
    auto start = std::chrono::high_resolution_clock::now();
    solve(d_input, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end - start;

    // Copy result back
    cudaMemcpy(h_input, d_input, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print sample outputs
    std::cout << "GPU execution time: " << gpu_time.count() << " seconds\n";
    std::cout << "Input[0] after reverse = " << h_input[0]
              << ", Input[N-1] after reverse = " << h_input[N - 1] << "\n";

    // Verify correctness
    bool correct = (h_input[0] == (N - 1) * 1.0f) && (h_input[N - 1] == 0.0f);
    if (correct)
        std::cout << "✅ Reverse operation successful.\n";
    else
        std::cout << "❌ Reverse operation failed.\n";

    // Cleanup
    cudaFree(d_input);
    delete[] h_input;

    return 0;
}
