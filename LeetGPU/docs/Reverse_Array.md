````markdown
# Reverse Array - CUDA & Mojo Implementation

## 📋 Overview

This document explains the GPU implementations of an array reversal routine implemented in `Reverse_Array.cu` (CUDA) and `mojo_gpu_leetcode/Reverse_Array.mojo` (Mojo). The core logic in both implementations is a `void reverse_array` function that performs an in-place reversal of an array on the device.

---

## 🎯 What This Does

`reverse_array` reverses the order of elements in a flat array in-place. For an input array A of length N:

Before: A = [a0, a1, a2, ..., aN-1]

After:  A = [aN-1, aN-2, ..., a2, a1, a0]

The function is designed to run on the accelerator (GPU) where each thread swaps one or more pairs of elements.

---

## 🤔 Why Reverse Array on GPU?

1. Large arrays benefit from parallel swaps to reduce wall-clock time.
2. Useful primitive for algorithms that require data reordering (FFT, certain sorting kernels, stream compactions).
3. Demonstrates safe in-place updates with proper index mapping and boundary checks.

---

## ⚡ Why GPU Approach (High-level)

### Parallelism
- Each swap (i, N-1-i) is independent — perfect for parallel execution.

### Performance
- For very large N the GPU can perform O(N/2) swaps concurrently (subject to hardware limits).

### Memory
- In-place algorithm: O(1) extra memory. Writes are to distinct locations when threads coordinate properly (each pair handled by exactly one thread).

---

## 🔧 Algorithm Design (Thread mapping)

### One thread per pair
- Thread index `tid` handles the pair `(tid, N-1-tid)` for `tid < N/2`.

### Work distribution if threads < N/2
- Each thread can process multiple pairs using strided indexing:
```
for (int idx = tid; idx < N/2; idx += gridDim.x * blockDim.x) {
    swap(A[idx], A[N-1-idx]);
}
```

### Boundary checks
- Only threads where `idx < N/2` perform swaps.
- For odd N the middle element at index `N/2` remains unchanged.

---

## 📊 Core Function: `void reverse_array`

The signature differs slightly between CUDA and Mojo but the algorithmic contract is the same.

Contract (common):
- Inputs: pointer/array `A`, integer `N` (length), each element is a device-accessible value (e.g., `int`, `float`, or templated type).
- Behavior: Reverse array in-place.
- Error modes: invalid pointer, N <= 0 (no-op), N == 1 (no-op).

### CUDA Kernel Style

Typical kernel wrapper and helper:

```cuda
__global__ void reverse_array_kernel(type* A, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < N/2; i += stride) {
        int j = N - 1 - i;
        // swap A[i] and A[j]
        type tmp = A[i];
        A[i] = A[j];
        A[j] = tmp;
    }
}

extern "C" void reverse_array(type* A, int N) {
    int threads = 256;
    int blocks = (N/2 + threads - 1) / threads; // ensure enough threads to cover N/2
    reverse_array_kernel<<<blocks, threads>>>(A, N);
    cudaDeviceSynchronize();
}
```

Notes:
- `type` stands for the element type used in the repository (often `int` or `float`).
- Using `N/2` when computing blocks reduces launched threads but grid-stride loop keeps it correct if more parallelism is desired.

### Mojo-style (pseudo/Mojo)

Mojo implementation follows the same mapping concept but uses Mojo/Accelerator abstractions to launch parallel loops. Example (conceptual):

```mojo
fn reverse_array<A: Copy>(device_array: device[A], N: i32) -> void {
  let tid = accelerator.thread_index();
  let stride = accelerator.grid_size();
  var i = tid;
  while i < N/2 {
    let j = N - 1 - i;
    let tmp = device_array[i];
    device_array[i] = device_array[j];
    device_array[j] = tmp;
    i += stride;
  }
}
```

Notes:
- Mojo's actual API for device kernels differs by runtime; above is conceptual and demonstrates identical index math and safety checks.

---

## 📈 Complexity

- Work: O(N) total operations (N/2 swaps)
- Parallel time: O(N/p) where p is number of threads (ignoring memory latency)
- Space: O(1) extra (in-place)

---

## ⚠️ Important Notes

1. Race conditions: Avoid having two threads swap the same pair. Using `i < N/2` and mapping each `i` uniquely prevents this.
2. Atomicity: Individual element writes must be atomic relative to element size; typical primitive types (32-bit/64-bit) are safe for single writes if no two threads write the same location.
3. Alignment: For best performance ensure device memory is properly aligned for the element type.
4. Large element types: For types larger than device word size consider using memcpy-style swaps or pairwise 64/128-bit loads/stores if supported.

---

## 🔬 Example Execution Trace

Input: A = [1, 2, 3, 4, 5, 6]

Threads (logical) handle pairs:
- Thread 0: swap(0,5) -> [6,2,3,4,5,1]
- Thread 1: swap(1,4) -> [6,5,3,4,2,1]
- Thread 2: swap(2,3) -> [6,5,4,3,2,1]

Final: [6,5,4,3,2,1]

For odd length N=5:
- Pairs: (0,4), (1,3) — middle element index 2 unchanged

---

## 🎮 Launch Configuration Examples (CUDA)

Small array N=1024

```
threadsPerBlock = 256
blocks = ceil((N/2) / threadsPerBlock) = ceil(512/256) = 2
reverse_array_kernel<<<2,256>>>(A, 1024);
```

Large array N=10,000,000

```
threadsPerBlock = 256
blocks = ceil((5,000,000)/256) = 19,531
reverse_array_kernel<<<19531,256>>>(A, 10_000_000);
```

Use grid-stride loop in the kernel if you'd rather launch a smaller grid and iterate with `stride = gridDim.x * blockDim.x`.

---

## 📚 Further Reading

- Parallel array reversal patterns
- CUDA grid-stride loops and occupancy notes
- Memory alignment and coalescing strategies

---

**File(s)**: `Reverse_Array.cu`, `mojo_gpu_leetcode/Reverse_Array.mojo`  
**Author**: ML-Leetcode Project  
**Date**: October 23, 2025

````
