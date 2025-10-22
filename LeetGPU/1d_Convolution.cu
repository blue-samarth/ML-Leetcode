#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
	int indx = blockDim.x * blockIdx.x + threadIdx.x;
	if (indx < input_size - kernel_size +1) {
		float sum = 0;
		for (int i = 0; i < kernel_size; i++) { sum += input[indx +i] * kernel[i]; }
		output[indx] = sum;
	}	
}

extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}

int main() {
    // Input parameters
    const int input_size = 10'000'000;       // 10 million elements
    const int kernel_size = 1024;            // Example kernel size
    const int output_size = input_size - kernel_size + 1;

    size_t input_bytes = input_size * sizeof(float);
    size_t kernel_bytes = kernel_size * sizeof(float);
    size_t output_bytes = output_size * sizeof(float);

    // Allocate host memory
    float* h_input = new float[input_size];
    float* h_kernel = new float[kernel_size];
    float* h_output = new float[output_size];

    // Initialize input and kernel
    for (int i = 0; i < input_size; ++i)
        h_input[i] = static_cast<float>(i % 1000) * 0.01f;

    for (int i = 0; i < kernel_size; ++i)
        h_kernel[i] = static_cast<float>(i % 10) * 0.1f;

    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_kernel, kernel_bytes);
    cudaMalloc(&d_output, output_bytes);

    // Copy input and kernel to device
    cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);

    // Measure GPU execution time
    auto start = std::chrono::high_resolution_clock::now();
    solve(d_input, d_kernel, d_output, input_size, kernel_size);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end - start;

    // Copy output back to host
    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    // Print performance and a few sample outputs
    std::cout << "1D convolution completed.\n";
    std::cout << "Input size: " << input_size << ", Kernel size: " << kernel_size << "\n";
    std::cout << "GPU execution time: " << gpu_time.count() << " seconds\n";
    std::cout << "Output[0] = " << h_output[0]
              << ", Output[last] = " << h_output[output_size - 1] << "\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output;

    return 0;
}
