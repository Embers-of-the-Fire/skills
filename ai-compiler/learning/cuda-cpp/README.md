# CUDA C++

## CUDA API

Typically CUDA programming relies on CUDA runtime API, which is built on top of lower-level CUDA driver API.

GPU C++ code is compiled with CUDA Compiler, `nvcc`.

## Kernels

### Definition

```cpp
// Kernel definition
__global__ void vecAdd(float* A, float* B, float* C)
{

}
```

### Execution

To launch a kernel, we could use "triple chevron notation" which is recommended or `cudaLaunchKernelEx`.

```cpp
__global__ void vecAdd(float* A, float* B, float* C)
{

}

int main()
{
    vecAdd<<<1, 256>>>(A, B, C);
    //       ↑   ↑--- tread block dim
    //       |
    //    grid dim
}
```

The invocation launches a single thread block containing 256 threads.

The execution of kernels are **asynchronous**, and the host code will not wait for the kernel to complete executing.
Synchronization must be manually used to determine whether the kernel has completed.
(This section is discussed later.)

To use 2/3-dim grids or blocks, CUDA offers a `dim3` builtin type.

```cpp
int main()
{
    dim3 grid(16, 16);
    dim3 block(8, 8);
    MatAdd<<<grid, block>>>(A, B, C);
}
```

### Intrinsics

CUDA provides builtin intrinsics to access the running state and the configuration.

- `threadIdx`: The index of a thread within the thread block.
- `blockDim`: The dimensions of the thread block.
- `blockIdx`: The index of a thread block within the grid.
- `gridDim`: The dimensions of the grid.

Each of the intrinsics are a vector with `.x`, `.y` and `.z` attribute.
Dimensions not specified by a configuration will default to `1`.
Indexes are all zero indexed.

For example:
```cpp
__global__ void vecAdd(float* A, float* B, float* C)
{
    // calculate which element this thread is responsible for computing.
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    
    // compute
    C[workIndex] = A[workIndex] + B[workIndex]
}

int main()
{
    // 4 thread blocks, 256 threads each, 1024 threads in total
    vecAdd<<<4, 256>>>(A, B, C);
}
```

However, the input data is not always the same count as the threads we launched, so we have to do bounds checking.

```cpp
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}
```

The number of blocks and threads could be calculated during runtime, as they're sent to GPU.

```cpp
int threads = 256;
int blocks = (vectorLength + threads - 1) / threads;
// or, using utility from CCCL
int blocks = cuda::ceil_div(vectorLength, threads);
vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
```

## Memory Management

To compute on GPU, data must be copied or available to GPU cores.

### Unified Memory

Unified memory is driver-managed data between host and devices.
The memory is allocated using the `cudaMallocManaged` API or by declaring a variable specified with `__managed__`.
The memory is accessible to both the GPU and the CPU.

```cpp
void unifiedMemExample(int vectorLength)
{
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // Data initialization
    initArray(A, vectorLength);
    initArray(B, vectorLength);
    
    // Launch the kernel.
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // Wait for the kernel to complete execution
    cudaDeviceSynchronize();

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);
}
```

### Explicit Memory

We could also use explicitly managed memory to improve performance.

```cpp
void explicitMemExample(int vectorLength)
{
    // host memory
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));
    
    // device memory
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    // Allocate Host Memory using `cudaMallocHost` API.
    // This is best practice when buffers will be used for copies between CPU and GPU memory
    cudaMallocHost(&A, vectorLength*sizeof(float));
    cudaMallocHost(&B, vectorLength*sizeof(float));
    cudaMallocHost(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // Allocate memory on the GPU
    cudaMalloc(&devA, vectorLength*sizeof(float));
    cudaMalloc(&devB, vectorLength*sizeof(float));
    cudaMalloc(&devC, vectorLength*sizeof(float));

    // Copy data to the GPU
    cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, vectorLength*sizeof(float));
    // end-allocate-and-copy

    // Launch the kernel
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC);
    // wait for kernel execution to complete
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault);

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Explicit Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
    }

    // clean up
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparisonResult);
}
```

**Note:**
- `cudaMemcpy` has a third param `cudaMemcpyKind_t`, with potential variant `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, `cudaMemcpyDeviceToDevice` or `cudaMemcpyDefault` for CUDA-determined by inspecting the source and destination addresses.
- `cudaMemcpy` is **synchronous**, and there is asynchronous APIs which will be discussed later.
- `cudaMallocHost` will allocate page-locked memory on CPU, which might cause degraded performance.
  Page-lock only buffers for sending or receiving data from the GPU.

Note that there are some hints for driver to manage unified memory, so explicit memory management is now always the better choice.

## Synchronization

The simplest way to synchronize the host and devices is to use `cudaDeviceSynchronize` which blocks the host.
However, that action will blocks the host until all streams ends, so for real-world usage, using stream sync APIs or CUDA Events is recommended and will be discussed later.

## Full Example

See [`vecAdd_unifiedMemory.cu`](./vecAdd_unifiedMemory.cu) and [`vecAdd_explicitMemory.cu](./vecAdd_explicitMemory.cu) for more information.

## Runtime Lifecycle

Each device has to be running on a CUDA context, which is created by the CUDA runtime.
Use `cudaInitDevice` or `cudaSetDevice` to initialize the runtime and the context for a specified device.
The runtime will set itself device 0 and initialize if not initialized.
`cudaDeviceReset` will destroy the context, and requests after reset will create a new context.

## Error Handling

### Error Identification

Every CUDA API returns a value of `cudaError_t`.
A `cudaSuccess` indicates that there is no error, and `cudaGetErrorString` could be used to get a human readable string to describe the meaning of a specific `cudaError_t`.
Use the following macro could help validate it.

```cpp
#define CUDA_CHECK(expr_to_check) do {                 \
    cudaError_t result  = expr_to_check;               \
    if(result != cudaSuccess)                          \
    {                                                  \
        fprintf(stderr,                                \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                              \
                __LINE__,                              \
                result,                                \
                cudaGetErrorString(result));           \
    }                                                  \
} while(0)
```

### Handling Errors

The error is stored as a global state for each host thread, which defaults to `cudaSuccess` and is overwritten whenever an error occurs.
`cudaGetLastError` pops the current error, and `cudaPeekLastError` gets without resetting it.
Note that **triple chevron notation** launches are not actually function calls and do not return a `cudaError_t`.

When it comes to function calls, the error state is not cleared if they are returned by an API function.

```cpp
vecAdd<<<blocks, threads>>>(devA, devB, devC);
// check error state after kernel launch
CUDA_CHECK(cudaGetLastError());
// wait for kernel execution to complete
// The `CUDA_CHECK` will report errors that occurred during execution of the kernel
CUDA_CHECK(cudaDeviceSynchronize());
```

### Logging

Use `CUDA_LOG_FILE` environment variable could redirect CUDA's internal log to an external file.

```text
$ env CUDA_LOG_FILE=cudaLog.txt ./errlog
CUDA Runtime Error: /home/cuda/intro-cpp/errorLogIllustration.cu:24:1 = invalid argument
$ cat cudaLog.txt
[12:46:23.854][137216133754880][CUDA][E] One or more of block dimensions of (4096,1,1) exceeds corresponding maximum value of (1024,1024,64)
[12:46:23.854][137216133754880][CUDA][E] Returning 1 (CUDA_ERROR_INVALID_VALUE) from cuLaunchKernel
```

`stdout` or `stderr` is also accepted by `CUDA_LOG_FILE`.

For more information about logging, see [error log management section](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html#error-log-management).

## Specifiers

### Function

- `__global__`: Entry point for a kernel
- `__device__`: Device-specific function, callable from either `__device__` or `__global__` functions
- `__host__`: Host-only function

### Variable

- `__device__`: Store in Global Memory
- `__constant__`: Store in Constant Memory
- `__managed__`: Store as Unified Memory
- `__shared__`: Store in Shared Memory.
- Others: Register, then local memory, then system memory.
