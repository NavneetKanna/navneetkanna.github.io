---
layout: ../../layouts/BlogPost.astro
title: "Understanding Metal and MSL"
date: 2024-01-30
---

- Metal is an API/framework provided by apple to interact with the GPU; MSL is a language used to write kernels using the features provided by metal.
- MSL is C++ 14 based.

## Steps to execute a kernel function 

1. Get an instance of the GPU to communicate too, using *MTLCreateSystemDefaultDevice()*
```python
device = Metal.MTLCreateSystemDefaultDevice()
```
2. Get an instance of the metal library that contains the compute function we want to run. This can be done in two ways:
    - Write the kernel code inside of a docstring and call *device.newLibraryWithSource_options_error_(prg, ..)*
    ```python
    options = Metal.MTLCompileOptions.new()
    lib = device.newLibraryWithSource_options_error_(prg, options, None)
    func_name = lib[0].newFunctionWithName_("addition_compute_function")
    ```
    - Or, write the kernel code in a seperate file with extension *.metal* and [compile it](https://developer.apple.com/documentation/metal/shader_libraries/building_a_shader_library_by_precompiling_source_files?language=objc)
    ```bash
    xcrun -sdk macosx metal -o test.ir  -c test.metal
    xcrun -sdk macosx metallib -o test.metallib test.ir
    ```
    and then call *newLibraryWithURL_error_("test.metallib)*
    ```python
    lib = device.newLibraryWithURL_error_("test.metallib", None)
    func_name = lib[0].newFunctionWithName_("addition_compute_function")
    ```
    Printing *func_name* should deisplay the compute kernel function name, device, function type and attributes (maybe the things displayed might vary based on the version of ...)
    Also, printing *lib* will show errors in the compute kernel code, if there are any. 
3. The compute kernel code is still not yet an executable code, to make it one, we have to create a pipeline. [*A pipeline specifies the steps that the GPU performs to complete a specific task*](https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu?language=objc)
```python
func_pso = device.newComputePipelineStateWithFunction_error_(func_name, None)
```
PSO defines how input data is interpreted. A compute pipeline can run a single compute function, hence multiple compute pipelines are needed for multiple compute functions
4. Create a command queue, which is used to send work(command buffers) to the GPU
```python
q = device.newCommandQueue()
```
5. Create data buffers,  
```python
buff1 = device.newBufferWithLength_options_(int, Metal.MTLResourceStorageModeShared)
buff2 = ...
```
Right now the buffers are allocations of memory without a predefined format. *MTLResourceStorageModeShared()* indicates that both the CPU and GPU uses a shared memory
6. Create a command buffer, a command buffer holds sequence of encoded commands
```python
cmd_buf = q.commandBuffer()
```
7. Create an encoder
```python
encoder = cmd_buf.computeCommandEncoder()
```
8. Set the pipeline state and compute kernel function arguments data
```python
encoder.setComputePipelineState_(func_pso[0])
encoder.setBuffer_offset_atIndex_(buff1, 0, 0)
encoder.setBuffer_offset_atIndex_(buff2, 0, 1)
```
The second parameter is offest: an offset of 0 means the command will access the data from the beginning of a buffer. The third parameter is the index of the argument in the compute kernel function
9. Specify the grid size (thread count) and the thread group size
```python
grid_size = Metal.MTLSizeMake(arrayLength, 1, 1)
threadGroupSize = func_pso[0].maxTotalThreadsPerThreadgroup()
if threadGroupSize > len(array1): threadGroupSize = len(array1)
thread_group_size = Metal.MTLSizeMake(threadGroupSize, 1, 1)
```
10. Encode the command to dispatch the threads
```python
encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, thread_group_size)
```
By specifying the grid size and number of threads per thread group, metal will calculate the right amount (non-uniform as well) of thread groups needed. If you dont want metal to calculate the number of thread groups, you can use 
```python
encoder.dispatchThreadgroups_threadsPerThreadgroup_()
```
11. End the encoder, when there are no more commands to encode
```python
encoder.endEncoding()
```
12. Run the command buffer by commiting it to the queue
```python
cmd_buf.commit()
```
13. Optionally, the program can wait or do some other task while the GPU is running
```python
cmd_buf.waitUntilCompleted()
```

So to summarise:
- Get the device
- Get the compute kernel function
- Create a pipeline for the compute kernel function which tells the GPU what to execute 
- Create a command queue, I like to think of it as a queue to the GPU
- Create data buffers which are needed for the kernerl function
- Create a command buffer which holds encoded commands such as the pipeline and arguments
- Create an encoder which is used to encode commands
- Encode the pipeline and arguments
- Encode the command to dispatch the threads
- End the encoding 
- Commit the command buffer to the command queue so that it can be executed by the GPU

Command buffer and command encoder objects are lightweight and can be created multiple times, however, Command queues, Data buffers, Compute states, Libraries are expensive and should be resued.

The full code 

```python
import Metal
import time 
import random
import array

count = 1000000 
idx = 990

array1 = [random.randint(1, count) for i in range(count)]
array1 = array.array('f', array1)

array2 = [random.randint(1, count) for i in range(count)]
array2 = array.array('f', array2)

print(f"ans: {array1[idx] + array2[idx]}")
device = Metal.MTLCreateSystemDefaultDevice()
prg = """
kernel void addition_compute_function(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = inA[index] + inB[index];
}
"""

# lib = device.newLibraryWithURL_error_("test.metallib", None)
options = Metal.MTLCompileOptions.new()
lib = device.newLibraryWithSource_options_error_(prg, options, None)
func_name = lib[0].newFunctionWithName_("addition_compute_function")

func_pso = device.newComputePipelineStateWithFunction_error_(func_name, None)

q = device.newCommandQueue()

# buff1 = device.newBufferWithBytes_length_options_(array1, len(array1.tobytes()), Metal.MTLResourceStorageModeShared)
# buff2 = device.newBufferWithBytes_length_options_(array2, len(array2.tobytes()), Metal.MTLResourceStorageModeShared)
buff1 = device.newBufferWithBytesNoCopy_length_options_deallocator_(array1, len(array1.tobytes()), Metal.MTLResourceStorageModeShared, None)
buff2 = device.newBufferWithBytesNoCopy_length_options_deallocator_(array2, len(array2.tobytes()), Metal.MTLResourceStorageModeShared, None)
buff3 = device.newBufferWithLength_options_(len(array1.tobytes()), Metal.MTLResourceStorageModeShared)

cmd_buf = q.commandBuffer()

encoder = cmd_buf.computeCommandEncoder()
encoder.setComputePipelineState_(func_pso[0])
encoder.setBuffer_offset_atIndex_(buff1, 0, 0)
encoder.setBuffer_offset_atIndex_(buff2, 0, 1)
encoder.setBuffer_offset_atIndex_(buff3, 0, 2)

grid_size = Metal.MTLSizeMake(len(array1), 1, 1)
threadGroupSize = func_pso[0].maxTotalThreadsPerThreadgroup()
if threadGroupSize > len(array1): threadGroupSize = len(array1)
thread_group_size = Metal.MTLSizeMake(threadGroupSize, 1, 1)

encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, thread_group_size)

encoder.endEncoding()

gpu_start = time.perf_counter()
cmd_buf.commit()
cmd_buf.waitUntilCompleted()
gpu_end = time.perf_counter()

print(f"time to execute on gpu: {gpu_end - gpu_start:.04f}s")

# In the pyobjc as_buffer() doc I think the wording is wrong, it should be return count bytes and not count elements
v = buff3.contents().as_buffer(len(array1.tobytes())).cast('f')

cpu_start = time.perf_counter()
result = [sum(x) for x in zip(array1, array2)]
cpu_end = time.perf_counter()

print(f"time to execute on cpu: {cpu_end - cpu_start:.04f}s")

print(f"{idx} values are {v[idx]} and {result[idx]}")
print(f"gpu is {((cpu_end-cpu_start) / (gpu_end-gpu_start)):.4f} times faster")


"""
Using np array

import numpy as np

array1 = np.random.rand(count).astype(np.float32)
array1_mv = np.require(array1, requirements='C').data

array2 = np.random.rand(count).astype(np.float32)
array2_mv = np.require(array2, requirements='C').data

buff1 = device.newBufferWithBytes_length_options_(array1_mv, array1.nbytes, Metal.MTLResourceStorageModeShared)
buff2 = device.newBufferWithBytes_length_options_(array2_mv, array2.nbytes, Metal.MTLResourceStorageModeShared)
buff3 = device.newBufferWithLength_options_(array1.nbytes, Metal.MTLResourceStorageModeShared)

v = buff3.contents().as_buffer(array1.nbytes).cast('f')
"""
```
