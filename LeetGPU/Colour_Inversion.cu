#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int indx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global thread ID
    int total_pixels = width * height; // Total number of pixels

    if (indx < total_pixels) { // Boundary check
	int base_indx = indx*4;
	image[base_indx + 0] ^= 255;
	image[base_indx + 1] ^= 255;
	image[base_indx + 2] ^= 255;
    }
}

extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}

int main() {
    // Image dimensions
    int width = 1920;
    int height = 1080;
    int num_pixels = width * height;
    int num_channels = 4;
    size_t image_size = num_pixels * num_channels * sizeof(unsigned char);

    // Host memory allocation
    std::vector<unsigned char> h_image(image_size);

    // Initialize the image with a gradient pattern (R=G=B=i%256, A=255)
    for (int i = 0; i < num_pixels; ++i) {
        h_image[i * 4 + 0] = static_cast<unsigned char>(i % 256); // R
        h_image[i * 4 + 1] = static_cast<unsigned char>(i % 256); // G
        h_image[i * 4 + 2] = static_cast<unsigned char>(i % 256); // B
        h_image[i * 4 + 3] = 255; // A
    }

    // Device memory allocation
    unsigned char* d_image;
    cudaMalloc(&d_image, image_size);

    // Copy input to device
    cudaMemcpy(d_image, h_image.data(), image_size, cudaMemcpyHostToDevice);

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    solve(d_image, width, height);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> gpu_time = end - start;

    // Copy result back
    cudaMemcpy(h_image.data(), d_image, image_size, cudaMemcpyDeviceToHost);

    // Print performance and verify a few pixels
    std::cout << "Image inversion completed for " << width << "x" << height << "\n";
    std::cout << "GPU execution time: " << gpu_time.count() << " seconds\n";

    // Verify a few pixels
    for (int i = 0; i < 5; ++i) {
        int base = i * 4;
        std::cout << "Pixel " << i << " -> ("
                  << (int)h_image[base + 0] << ", "
                  << (int)h_image[base + 1] << ", "
                  << (int)h_image[base + 2] << ", "
                  << (int)h_image[base + 3] << ")\n";
    }

    // Cleanup
    cudaFree(d_image);

    return 0;
}
