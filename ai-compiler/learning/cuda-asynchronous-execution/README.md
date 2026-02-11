# CUDA Asynchronous Execution

## Overlap Support

CUDA allows concurrent execution of the following tasks:
- computation on the host
- computation on the device
- memory transfers from the host to the device
- memory transfers within the memory of a given device
- memory transfers among devices

The concurrency gateway, or the asynchronous interface, is expressed that a dispatching function call or kernel launch returns immediately.

Generally the interface provides three main ways to synchronize:
- **blocking approach**, where the function blocks
- **non-blocking approach**, or polling approach
- **callback approach**, where a function is registered to be executed after the action is done.

## CUDA Streams

*CUDA Stream* works like a *work-queue* into which programs can add operations to be executed in order.
An application could use multiple streams simultaneously, where the runtime select a task to execute from the streams.
The API function calls and kernel launches operating in a stream are all asynchronous.

**Note**: Operations or kernel launches without a specific stream are queued into the default stream.

### Lifecycle

```cpp
cudaStream_t stream;        // Stream handle
cudaStreamCreate(&stream);  // Create a new stream

// stream based operations ...

cudaStreamDestroy(stream);  // Destroy the stream
```

**Note**: If the device is still doing work in stream when the application calls `cudaStreamDestroy`, the stream will complete all the work before being destroyed.

### Launching Kernels

```cpp
kernel<<<grid, block, shared_mem_size, stream>>>(...);
```

**Note**: The `stream` is a value of `cudaStream_t`.

### Launching Memory Transfers

```cpp
// Copy `size` bytes from `src` to `dst` in stream `stream`
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
```

**Note**: The memory must be pinned and page-locked to be asynchronously transferred to device, so buffers must be allocated with `cudaMallocHost`.

### Synchronization

The simplest way is to use `cudaStreamSynchronize(stream)` which will block until all the work in the stream is done.

If we do not want to block the program but to inspect the stream's status, use `cudaStreamQuery`:
```cpp
// Have a peek at the stream
// returns cudaSuccess if the stream is empty
// returns cudaErrorNotReady if the stream is not empty
cudaError_t status = cudaStreamQuery(stream);

switch (status) {
    case cudaSuccess:
        // The stream is empty
        std::cout << "The stream is empty" << std::endl;
        break;
    case cudaErrorNotReady:
        // The stream is not empty
        std::cout << "The stream is not empty" << std::endl;
        break;
    default:
        // An error occurred - we should handle this
        break;
};
```

## CUDA Events

CUDA Events are used for inserting markers into a CUDA Stream.

### Lifecycle

```cpp
cudaEvent_t event;

// Create the event
cudaEventCreate(&event);

// do some work involving the event

// Once the work is done and the event is no longer needed
// we can destroy the event
cudaEventDestroy(event);
```

### Inserting Events into Streams

```cpp
cudaEvent_t event;
cudaStream_t stream;

// Create the event
cudaEventCreate(&event);

// Insert the event into the stream
cudaEventRecord(event, stream);
```

### Time Operations

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaEvent_t start;
cudaEvent_t stop;

// create the events
cudaEventCreate(&start);
cudaEventCreate(&stop);

 // record the start event
cudaEventRecord(start, stream);

// launch the kernel
kernel<<<grid, block, 0, stream>>>(...);

// record the stop event
cudaEventRecord(stop, stream);

// wait for the stream to complete
// both events will have been triggered
cudaStreamSynchronize(stream);

// get the timing
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
std::cout << "Kernel execution time: " << elapsedTime << " ms" << std::endl;

// clean up
cudaEventDestroy(start);
cudaEventDestroy(stop);
cudaStreamDestroy(stream);
```

### Inspecting the Status

The `cudaEventSynchronize` function will block until the event has completed.

```cpp
cudaEvent_t event;
cudaStream_t stream;

// create the stream
cudaStreamCreate(&stream);

// create the event
cudaEventCreate(&event);

// launch a kernel into the stream
kernel<<<grid, block, 0, stream>>>(...);

// Record the event
cudaEventRecord(event, stream);

// launch a kernel into the stream
kernel2<<<grid, block, 0, stream>>>(...);

// Wait for the event to complete
// Kernel 1 will be  guaranteed to have completed
// and we can launch the dependent task.
cudaEventSynchronize(event);
dependentCPUtask();

// Wait for the stream to be empty
// Kernel 2 is guaranteed to have completed
cudaStreamSynchronize(stream);

// destroy the event
cudaEventDestroy(event);

// destroy the stream
cudaStreamDestroy(stream);
```

And similar to steam queries, we could use `cudaEventQuery` to inspect the status in a non-blocking way.

```cpp
cudaEvent_t event;
cudaStream_t stream1;
cudaStream_t stream2;

size_t size = LARGE_NUMBER;
float *d_data;

// Create some data
cudaMalloc(&d_data, size);
float *h_data = (float *)malloc(size);

// create the streams
cudaStreamCreate(&stream1);   // Processing stream
cudaStreamCreate(&stream2);   // Copying stream
bool copyStarted = false;

//  create the event
cudaEventCreate(&event);

// launch kernel1 into the stream
kernel1<<<grid, block, 0, stream1>>>(d_data, size);
// enqueue an event following kernel1
cudaEventRecord(event, stream1);

// launch kernel2 into the stream
kernel2<<<grid, block, 0, stream1>>>();

// while the kernels are running do some work on the CPU
// but check if kernel1 has completed because then we will start
// a device to host copy in stream2
while ( not allCPUWorkDone() || not copyStarted ) {
    doNextChunkOfCPUWork();

    // peek to see if kernel 1 has completed
    // if so enqueue a non-blocking copy into stream2
    if ( not copyStarted ) {
        if( cudaEventQuery(event) == cudaSuccess ) {
            cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream2);
            copyStarted = true;
        }
    }
}

// wait for both streams to be done
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

// destroy the event
cudaEventDestroy(event);

// destroy the streams and free the data
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
cudaFree(d_data);
free(h_data);
```

## Callback Functions

Two functions could be used for launching functions on the host from within a stream: `cudaLaunchHostFunc` and `cudaAddCallback` (slated for deprecation).

```cpp
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*func)(void *), void *data);
```
- `stream`: The stream to launch the callback function into
- `func`: The callback function
- `data`: A pointer to the data to pass to the callback

**Note**: The host function may not call any CUDA APIs.

The following execution guarantees are provided:
- The stream is considered idle for the duration of the function's execution.
- The  start of execution has the same effect as synchronizing an event recorded in the same stream immediately prior to the function.
- Completion of the function does not cause a stream to become active except as described above.

## Error Handling

```cpp
// Some work occurs in streams.
cudaStreamSynchronize(stream);

// Look at the last error but do not clear it
cudaError_t err = cudaPeekAtLastError();
if (err != cudaSuccess) {
    printf("Error with name: %s\n", cudaGetErrorName(err));
    printf("Error description: %s\n", cudaGetErrorString(err));
}

// Look at the last error and clear it
cudaError_t err2 = cudaGetLastError();
if (err2 != cudaSuccess) {
    printf("Error with name: %s\n", cudaGetErrorName(err2));
    printf("Error description: %s\n", cudaGetErrorString(err2));
}

if (err2 != err) {
    printf("As expected, cudaPeekAtLastError() did not clear the error\n");
}

// Check again
cudaError_t err3 = cudaGetLastError();
if (err3 == cudaSuccess) {
    printf("As expected, cudaGetLastError() cleared the error\n");
}
```

## Non-blocking and Blocking Streams

Blocking streams will be blocked when interacting with the default stream.
To create a non-blocking stream, use `cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)`.

The default stream with ID 0 works a bit different:
```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

kernel1<<<grid, block, 0, stream1>>>(...);
kernel2<<<grid, block>>>(...);
kernel3<<<grid, block, 0, stream2>>>(...);

cudaDeviceSynchronize();
```

`kernel2` will wait for `kernel1` to complete and `kernel3` will wait for `kernel2` to complete.
We could avoid this forced synchronization by creating non-blocking streams.
This could also be avoided globally by enabling per-thread default stream option.

## Per-thread Default Stream

CUDA allows for each host thread to have its own independent default stream by enabling the compiler option `--default-stream per-thread` or defining the `CUDA_API_PER_THREAD_DEFAULT_STREAM` macro.

## Prioritization

By assign priorities to streams developers could hint the runtime to schedule streams better:
```cpp
int minPriority, maxPriority;

// Query the priority range for the device
cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority);

// Create two streams with different priorities
// cudaStreamDefault indicates the stream should be created with default flags
// in other words they will be blocking streams with respect to the legacy default stream
// One could also use the option `cudaStreamNonBlocking` here to create a non-blocking streams
cudaStream_t stream1, stream2;
cudaStreamCreateWithPriority(&stream1, cudaStreamDefault, minPriority);  // Lowest priority
cudaStreamCreateWithPriority(&stream2, cudaStreamDefault, maxPriority);  // Highest priority
```
