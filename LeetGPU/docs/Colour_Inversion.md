# Color Inversion - CUDA Implementation

## 📋 Overview

This document explains the CUDA implementation of image color inversion in `Colour_Inversion.cu`.

---

## 🎯 What This Does

This program inverts the colors of an RGBA image using GPU parallel processing.

### Color Inversion Formula
For each pixel, RGB channels are inverted using:
```
new_value = 255 - old_value
```

### Examples
- **Red** `(255, 0, 0)` → **Cyan** `(0, 255, 255)`
- **Green** `(0, 255, 0)` → **Magenta** `(255, 0, 255)`
- **Blue** `(0, 0, 255)` → **Yellow** `(255, 255, 0)`
- **White** `(255, 255, 255)` → **Black** `(0, 0, 0)`
- **Alpha channel remains unchanged** to preserve transparency

---

## 🤔 Why Color Inversion?

Color inversion is a fundamental image processing operation used in:

1. **Photo Editing** - Creating negative effects
2. **Accessibility** - Improving contrast for visually impaired users
3. **Dark Mode** - Converting light themes to dark themes
4. **Medical Imaging** - Highlighting specific features in X-rays and scans
5. **Computer Vision** - Preprocessing for machine learning models
6. **Printing** - Creating film negatives

---

## ⚡ Why GPU/CUDA Approach?

### 1. **Massive Parallelism**
- Each pixel can be inverted **independently** (embarrassingly parallel problem)
- A 1920×1080 image has **2,073,600 pixels** that can be processed **simultaneously**
- CPU processes sequentially; GPU processes **thousands in parallel**

### 2. **Performance Gains**
| Approach | Time Complexity | Real-world Performance |
|----------|----------------|------------------------|
| **CPU Sequential** | O(n) operations, one at a time | Baseline |
| **GPU Parallel** | O(n/p) where p = thousands of cores | **10-100x faster** |

### 3. **Memory Efficiency**
- Image data stays in **GPU memory** (no repeated CPU↔GPU transfers)
- Each thread accesses its **own memory region** (no race conditions)
- **Coalesced memory access** pattern improves bandwidth utilization

### 4. **Scalability**
- Same code works for **any image size** (just adjust grid/block dimensions)
- Automatically utilizes **all available GPU cores**
- Easily extended to **batch processing** multiple images

---

## 🔧 Algorithm Design

### Thread-to-Pixel Mapping
- **One thread per pixel** (not per channel)
- Minimizes thread divergence
- Each thread processes **4 bytes (RGBA)** with **3 arithmetic operations**

### Memory Layout
```
Image array: [R0, G0, B0, A0, R1, G1, B1, A1, R2, G2, B2, A2, ...]
              ↑              ↑              ↑
           Pixel 0       Pixel 1        Pixel 2
```

### Why This Layout?
- **Sequential access** by each thread improves cache utilization
- **No memory conflicts** between threads
- **Simple index calculation**: `base_idx = pixel_idx * 4`

---

## 📊 Code Breakdown

### Kernel Function: `invert_kernel`

```cuda
__global__ void invert_kernel(unsigned char* image, int width, int height)
```

#### Step 1: Calculate Thread ID
```cuda
int indx = blockIdx.x * blockDim.x + threadIdx.x;
```
- Each thread gets a **unique index** based on its position in the grid
- `blockIdx.x`: Which block this thread belongs to
- `blockDim.x`: Number of threads per block
- `threadIdx.x`: Position within the block

**Example:**
- Block 2, Thread 5, Block size 256 → Thread ID = 2 × 256 + 5 = **517**

#### Step 2: Calculate Total Pixels
```cuda
int total_pixels = width * height;
```
- Total number of pixels to process
- Used for boundary checking

#### Step 3: Boundary Check
```cuda
if (indx < total_pixels) {
```
- **Critical safety check**
- Prevents out-of-bounds memory access
- Necessary because total pixels may not be evenly divisible by block size

**Example:**
- 1000 pixels, 256 threads per block → Launch 4 blocks = 1024 threads
- Last 24 threads (1000-1023) will fail this check and do nothing

#### Step 4: Calculate Base Index
```cuda
int base_idx = indx * 4;
```
- Each pixel occupies **4 consecutive bytes** (RGBA)
- Multiply by 4 to jump to this pixel's data

**Example:**
- Pixel 100 → Base index = 100 × 4 = **400**
- Accesses bytes 400, 401, 402, 403

#### Step 5: Invert RGB Channels
```cuda
image[base_idx + 0] = 255 - image[base_idx + 0]; // Red
image[base_idx + 1] = 255 - image[base_idx + 1]; // Green
image[base_idx + 2] = 255 - image[base_idx + 2]; // Blue
// image[base_idx + 3] is Alpha - NOT modified
```
- **255** is the maximum value for 8-bit color (2⁸ - 1)
- Subtracting from 255 creates the inverse color
- **Alpha channel preserved** to maintain transparency

---

## 🚀 Host Function: `solve`

```cuda
extern "C" void solve(unsigned char* image, int width, int height)
```

### Step 1: Configure Threads Per Block
```cuda
int threadsPerBlock = 256;
```
- **256 is optimal** for most modern GPUs
- Powers of 2 (128, 256, 512, 1024) are efficient
- Balance between occupancy and resource usage

### Step 2: Calculate Blocks Per Grid
```cuda
int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
```
- **Ceiling division** to ensure all pixels are covered
- Formula: `ceil(total_pixels / threadsPerBlock)`

**Example:**
- 1920×1080 = 2,073,600 pixels
- Blocks = (2,073,600 + 255) / 256 = **8,100 blocks**

### Step 3: Launch Kernel
```cuda
invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
```
- Triple angle brackets `<<<>>>` is CUDA kernel launch syntax
- Spawns 8,100 blocks × 256 threads = **2,073,600 threads**

### Step 4: Synchronize
```cuda
cudaDeviceSynchronize();
```
- **Blocks** until all GPU work is complete
- Ensures image is fully processed before returning
- Critical for correctness in asynchronous GPU execution

---

## 📈 Performance Analysis

### Time Complexity
- **Per thread**: O(1) - constant time operations
- **Overall**: O(n) with n threads running in parallel
- **Sequential CPU**: O(n) but processed one at a time

### Space Complexity
- **O(1)** - In-place modification
- No additional memory allocated
- Original image is overwritten

### Memory Access Pattern
- **Coalesced reads/writes** - threads access consecutive memory locations
- Maximizes memory bandwidth utilization
- Each warp (32 threads) accesses 128 consecutive bytes

---

## 🎮 Launch Configuration Examples

### Small Image (640×480)
```cuda
Total pixels: 307,200
Threads per block: 256
Blocks: ceil(307,200 / 256) = 1,200
Launch: invert_kernel<<<1200, 256>>>(image, 640, 480);
```

### Medium Image (1920×1080 - Full HD)
```cuda
Total pixels: 2,073,600
Threads per block: 256
Blocks: ceil(2,073,600 / 256) = 8,100
Launch: invert_kernel<<<8100, 256>>>(image, 1920, 1080);
```

### Large Image (3840×2160 - 4K)
```cuda
Total pixels: 8,294,400
Threads per block: 256
Blocks: ceil(8,294,400 / 256) = 32,400
Launch: invert_kernel<<<32400, 256>>>(image, 3840, 2160);
```

---

## ⚠️ Important Notes

### Thread Safety
- ✅ **Thread-safe**: Each thread writes to distinct memory locations
- ✅ **No race conditions**: No two threads access the same pixel
- ✅ **No synchronization needed** within the kernel

### Preconditions
- Image must be **allocated in GPU device memory**
- Image size must be exactly **width × height × 4 bytes**
- Pointer must be valid and accessible from device code

### Postconditions
- RGB channels are inverted
- Alpha channel remains unchanged
- Original image data is overwritten

### Limitations
- **In-place operation**: Original image is lost (save a copy if needed)
- **RGBA format only**: Does not handle RGB (3-channel) images
- **8-bit per channel**: Assumes unsigned char (0-255)

---

## 🔬 Example Execution Trace

### Input Image (2×2 pixels)
```
Pixel 0: [255, 100, 50, 200]  (Light orange, partially transparent)
Pixel 1: [0, 255, 0, 255]     (Green, opaque)
Pixel 2: [0, 0, 255, 128]     (Blue, half transparent)
Pixel 3: [255, 255, 255, 0]   (White, fully transparent)
```

### Memory Layout
```
[255, 100, 50, 200, 0, 255, 0, 255, 0, 0, 255, 128, 255, 255, 255, 0]
```

### Execution
- **Thread 0** processes Pixel 0 (indices 0-3)
- **Thread 1** processes Pixel 1 (indices 4-7)
- **Thread 2** processes Pixel 2 (indices 8-11)
- **Thread 3** processes Pixel 3 (indices 12-15)

### Output Image
```
Pixel 0: [0, 155, 205, 200]    (Inverted orange, alpha unchanged)
Pixel 1: [255, 0, 255, 255]    (Magenta, alpha unchanged)
Pixel 2: [255, 255, 0, 128]    (Yellow, alpha unchanged)
Pixel 3: [0, 0, 0, 0]          (Black, alpha unchanged)
```

---

## 🎓 Key Takeaways

1. **GPU parallelism** provides massive speedup for pixel-wise operations
2. **One thread per pixel** is an efficient mapping for this problem
3. **Boundary checking** is critical for correctness
4. **Alpha preservation** maintains transparency information
5. **Coalesced memory access** maximizes GPU bandwidth
6. **Simple algorithm** with complex performance optimizations under the hood

---

## 📚 Further Reading

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Image Processing on GPU](https://developer.nvidia.com/gpugems/gpugems2/part-iv-general-purpose-computation-gpus-primer)
- [Memory Coalescing in CUDA](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)

---

**File**: `Colour_Inversion.cu`  
**Author**: ML-Leetcode Project  
**Date**: October 21, 2025
