# 1D Convolution - CUDA & Mojo Implementations

## 📋 Overview

This document explains the GPU kernels for a "valid" 1D convolution used in this repo: the CUDA kernel in `1d_Convolution.cu` and the Mojo kernel in `mojo_gpu_leetcode/1D_Convolution.mojo`.

Both kernels compute the same operation: each GPU thread computes one element of the output by taking the dot-product of a sliding window of the input with the kernel. The valid output size is `output_size = input_size - kernel_size + 1`.

---

## 🎯 What This Does

- Computes a valid 1D convolution: output[t] = sum_{i=0..kernel_size-1} input[t + i] * kernel[i]
- Work distribution: **one thread per output element** (embarrassingly parallel)

---

## ⚡ Why GPU for 1D Convolution?

- Each output element is independent -> massive parallelism.
- High throughput for long inputs (millions of elements).
- Use GPU when input_size × kernel_size work justifies data transfer and kernel launch overhead.

---

## 🔧 Kernel logic (the snippets)

### CUDA kernel (from `1d_Convolution.cu`)

```cuda
__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int indx = blockDim.x * blockIdx.x + threadIdx.x;
    if (indx < input_size - kernel_size +1) {
        float sum = 0;
        for (int i = 0; i < kernel_size; i++) { sum += input[indx +i] * kernel[i]; }
        output[indx] = sum;
    }
}
```

### Mojo kernel (from `mojo_gpu_leetcode/1D_Convolution.mojo`)

```mojo
fn convolution_1d_kernel(input: UnsafePointer[Float32], kernel: UnsafePointer[Float32], 
                         output: UnsafePointer[Float32], input_size: Int32, kernel_size: Int32):
    var indx = Int32(block_idx.x * block_dim.x + thread_idx.x)
    if indx < input_size - kernel_size +1:
        var summ: Float32 = 0.0
        for i in range(kernel_size):
                summ += input[indx +i] * kernel[i]
        output[indx] = summ
```

---

## 📊 Line-by-line explanation (CUDA)

1. `__global__ void convolution_1d_kernel(...)` — declares a device kernel callable from host.
2. `int indx = blockDim.x * blockIdx.x + threadIdx.x;` — compute the global thread index (unique output position this thread will compute).
   - `blockDim.x`: threads per block
   - `blockIdx.x`: block index in the grid
   - `threadIdx.x`: thread index inside the block
3. `if (indx < input_size - kernel_size +1) {` — bounds check. Only threads that map to valid output indices proceed.
   - output_size = `input_size - kernel_size + 1`
4. `float sum = 0;` — accumulator for the dot-product.
5. `for (int i = 0; i < kernel_size; i++) { sum += input[indx + i] * kernel[i]; }` — inner loop performing the dot product between the kernel and the input window starting at `indx`.
   - Reads `input[indx + 0..kernel_size-1]` and `kernel[0..kernel_size-1]`.
6. `output[indx] = sum;` — write the computed value to the output buffer. Each thread writes a distinct location.

### Key CUDA notes
- No synchronization between threads is needed — threads write disjoint output elements.
- The kernel uses global memory reads for input and kernel. For small kernels, copying the kernel to shared memory per block can save bandwidth.
- If the launched grid overshoots the output size, the bounds check prevents out-of-bounds memory access.

---

## 📊 Line-by-line explanation (Mojo)

1. `fn convolution_1d_kernel(...)` — declares a Mojo GPU kernel; parameters are `UnsafePointer[Float32]` (device pointers) and sizes as `Int32`.
2. `var indx = Int32(block_idx.x * block_dim.x + thread_idx.x)` — same global thread index formula expressed with Mojo helpers (`block_idx`, `block_dim`, `thread_idx`).
3. `if indx < input_size - kernel_size +1:` — bounds check identical to CUDA.
4. `var summ: Float32 = 0.0` — accumulator initialization.
5. `for i in range(kernel_size): summ += input[indx + i] * kernel[i]` — the dot-product loop; same memory access pattern.
6. `output[indx] = summ` — write the result.

### Key Mojo notes
- Logic and memory-access pattern are the same as CUDA; only the language/runtime wrapper differs.
- Mojo kernel assumes the runtime's `DeviceContext` will provide proper grid/block dispatch and device pointers.

---

## 🔬 Example numeric walkthrough

Input: `[1, 2, 3, 4]`, Kernel: `[1, 0, -1]` (kernel_size=3)

- input_size = 4 -> output_size = 4 - 3 + 1 = 2
- Thread 0 (indx=0) computes: 1*1 + 2*0 + 3*(-1) = 1 + 0 - 3 = -2 -> output[0]
- Thread 1 (indx=1) computes: 2*1 + 3*0 + 4*(-1) = 2 + 0 - 4 = -2 -> output[1]

Both kernels produce `[-2, -2]`.

---

## ⚠️ Preconditions, limitations, and optimization tips

- Preconditions:
  - `input_size >= kernel_size` (otherwise output size is non-positive or invalid).
  - Pointers must refer to device-accessible memory when `solve` is called from the host side.

- Limitations:
  - The provided kernels implement the naive O(input_size × kernel_size) convolution.
  - For large kernels, per-thread work is heavy and memory bandwidth may dominate.

- Optimization tips:
  1. Load the kernel into shared memory (one copy per block) when it fits — reduces repeated kernel reads.
  2. Tile the input and reuse values across threads in a block to reduce global memory traffic.
  3. For very large kernels, consider FFT-based convolution (cuFFT or library) which can be faster.
  4. Consider loop unrolling or vectorized loads if alignment/size allow.

---

## 🎓 Key takeaways

1. Both kernels perform the same per-thread dot-product logic; only host/dispatch differs.
2. Each thread computes a single output element and writes to a unique location (no races).
3. Bounds checking is critical when grid size overshoots the output size.
4. Shared memory and tiling are standard optimizations when kernel_size is moderate.

---
