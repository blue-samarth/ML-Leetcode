from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math.math import ceildiv

fn matrix_multiplication_kernel(A: UnsafePointer[Float32], B: UnsafePointer[Float32], C: UnsafePointer[Float32], M: Int32, N: Int32, K: Int32):
    var row = Int32(block_dim.y * block_idx.y + thread_idx.y)
    var col = Int32(block_dim.x * block_idx.x + thread_idx.x)

    if row < M and col < K:
        var summ: Float32 = 0.0
        for i in range(N):
                summ += A[row*N + i] * B[i*K + col]
        C[row*K + col] = summ


# A, B, C are device pointers (i.e. pointers to memory on the GPU)
@export                         
def solve(A: UnsafePointer[Float32], B: UnsafePointer[Float32], C: UnsafePointer[Float32], M: Int32, N: Int32, K: Int32):
    var BLOCK_SIZE: Int32 = 16
    var ctx = DeviceContext()
    
    var grid_dim_x = ceildiv(K, BLOCK_SIZE)
    var grid_dim_y = ceildiv(M, BLOCK_SIZE)
    
    ctx.enqueue_function[matrix_multiplication_kernel](
        A, B, C, M, N, K,
        grid_dim = (grid_dim_x, grid_dim_y),
        block_dim = (BLOCK_SIZE, BLOCK_SIZE)
    )
    
    ctx.synchronize() 
