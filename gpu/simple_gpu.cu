#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "workload.h"


// ============================================================
// CUDA Error Checking
// ============================================================

void CheckCUDA(
    cudaError_t error,
    const char* message
) {
    if (error != cudaSuccess) {

        std::fprintf(
            stderr,
            "%s: %s\n",
            message,
            cudaGetErrorString(error)
        );

        std::exit(EXIT_FAILURE);
    }
}


// ============================================================
// CUDA Warm-Up Kernel
// Used only to initialize CUDA before benchmarking
// ============================================================

__global__
void WarmupKernel() {
}


// ============================================================
// Simple Computation - GPU
// Same computation used by the CPU implementation
// ============================================================

__device__
unsigned long long ExecuteSimpleTaskGPU(
    unsigned long long input,
    int iterations
) {
    unsigned long long value = input;

    for (int i = 0; i < iterations; ++i) {

        value =
            value * 1664525ULL
            + 1013904223ULL;

        value ^= value >> 16;
    }

    return value;
}


// ============================================================
// CUDA Kernel
// One GPU thread executes one independent task
// ============================================================

__global__
void SimpleKernel(
    const SimpleTask* tasks,
    int task_count,
    int iterations,
    unsigned long long* results
) {
    int index =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (index < task_count) {

        results[index] =
            ExecuteSimpleTaskGPU(
                tasks[index].input,
                iterations
            );
    }
}


// ============================================================
// Main
// ============================================================

int main() {

    using Clock =
        std::chrono::steady_clock;


    const int task_count = 1000;
    const int iterations = 100000;

    const int threads_per_block = 256;

    const int blocks =
        (task_count + threads_per_block - 1)
        / threads_per_block;


    // --------------------------------------------------------
    // Generate the exact same shared workload
    // --------------------------------------------------------

    std::vector<SimpleTask> tasks(
        task_count
    );

    std::vector<unsigned long long> results(
        task_count
    );


    GenerateSimpleWorkload(
        tasks.data(),
        task_count
    );


    // ========================================================
    // CUDA Warm-Up
    //
    // This initializes CUDA before formal timing begins.
    // Warm-up time is NOT included in any benchmark result.
    // ========================================================

    WarmupKernel<<<1, 1>>>();


    CheckCUDA(
        cudaGetLastError(),
        "Warm-up kernel launch failed"
    );


    CheckCUDA(
        cudaDeviceSynchronize(),
        "CUDA warm-up failed"
    );


    // --------------------------------------------------------
    // CUDA Events for Kernel Timing
    // --------------------------------------------------------

    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;


    CheckCUDA(
        cudaEventCreate(
            &kernel_start
        ),
        "cudaEventCreate failed"
    );


    CheckCUDA(
        cudaEventCreate(
            &kernel_stop
        ),
        "cudaEventCreate failed"
    );


    // --------------------------------------------------------
    // Device Memory
    // --------------------------------------------------------

    SimpleTask* d_tasks =
        nullptr;


    unsigned long long* d_results =
        nullptr;


    // ========================================================
    // 1. GPU FULL TIME START
    //
    // cudaMalloc
    //      ↓
    // H2D
    //      ↓
    // Kernel
    //      ↓
    // D2H
    //      ↓
    // cudaFree
    // ========================================================

    auto full_start =
        Clock::now();


    // --------------------------------------------------------
    // Allocate GPU Memory
    // --------------------------------------------------------

    CheckCUDA(
        cudaMalloc(
            &d_tasks,
            task_count * sizeof(SimpleTask)
        ),
        "cudaMalloc tasks failed"
    );


    CheckCUDA(
        cudaMalloc(
            &d_results,
            task_count *
            sizeof(unsigned long long)
        ),
        "cudaMalloc results failed"
    );


    // ========================================================
    // 2. H2D-to-D2H TIME START
    //
    // Start immediately BEFORE transmitting tasks to GPU.
    //
    // H2D
    //      ↓
    // Kernel
    //      ↓
    // D2H
    //
    // Stop immediately AFTER results return to CPU.
    // ========================================================

    auto h2d_d2h_start =
        Clock::now();


    // --------------------------------------------------------
    // Host -> Device
    // --------------------------------------------------------

    CheckCUDA(
        cudaMemcpy(
            d_tasks,
            tasks.data(),
            task_count * sizeof(SimpleTask),
            cudaMemcpyHostToDevice
        ),
        "Host-to-device copy failed"
    );


    // ========================================================
    // 3. GPU EXECUTION / KERNEL TIME START
    // ========================================================

    CheckCUDA(
        cudaEventRecord(
            kernel_start
        ),
        "Kernel start event failed"
    );


    // --------------------------------------------------------
    // Execute GPU Workload
    // --------------------------------------------------------

    SimpleKernel<<<
        blocks,
        threads_per_block
    >>>(
        d_tasks,
        task_count,
        iterations,
        d_results
    );


    CheckCUDA(
        cudaGetLastError(),
        "Kernel launch failed"
    );


    // ========================================================
    // GPU EXECUTION / KERNEL TIME STOP
    // ========================================================

    CheckCUDA(
        cudaEventRecord(
            kernel_stop
        ),
        "Kernel stop event failed"
    );


    CheckCUDA(
        cudaEventSynchronize(
            kernel_stop
        ),
        "Kernel execution failed"
    );


    // --------------------------------------------------------
    // Device -> Host
    // --------------------------------------------------------

    CheckCUDA(
        cudaMemcpy(
            results.data(),
            d_results,
            task_count *
            sizeof(unsigned long long),
            cudaMemcpyDeviceToHost
        ),
        "Device-to-host copy failed"
    );


    // ========================================================
    // H2D-to-D2H TIME STOP
    //
    // Result transfer has now completed.
    // ========================================================

    auto h2d_d2h_end =
        Clock::now();


    // --------------------------------------------------------
    // Free GPU Memory
    // --------------------------------------------------------

    CheckCUDA(
        cudaFree(
            d_tasks
        ),
        "cudaFree tasks failed"
    );


    CheckCUDA(
        cudaFree(
            d_results
        ),
        "cudaFree results failed"
    );


    // ========================================================
    // GPU FULL TIME STOP
    // ========================================================

    auto full_end =
        Clock::now();


    // ========================================================
    // Calculate Timing
    // ========================================================

    float kernel_time =
        0.0f;


    CheckCUDA(
        cudaEventElapsedTime(
            &kernel_time,
            kernel_start,
            kernel_stop
        ),
        "Kernel timing failed"
    );


    double h2d_d2h_time =

        std::chrono::duration<
            double,
            std::milli
        >(
            h2d_d2h_end
            - h2d_d2h_start
        ).count();


    double full_time =

        std::chrono::duration<
            double,
            std::milli
        >(
            full_end
            - full_start
        ).count();


    // --------------------------------------------------------
    // Destroy CUDA Events
    // --------------------------------------------------------

    CheckCUDA(
        cudaEventDestroy(
            kernel_start
        ),
        "cudaEventDestroy failed"
    );


    CheckCUDA(
        cudaEventDestroy(
            kernel_stop
        ),
        "cudaEventDestroy failed"
    );


    // ========================================================
    // Output
    // ========================================================

    printf(
        "\n"
        "================ SIMPLE GPU TEST ================\n"
    );


    printf(
        "Tasks: %d\n",
        task_count
    );


    printf(
        "Iterations per task: %d\n",
        iterations
    );


    printf(
        "Blocks: %d\n",
        blocks
    );


    printf(
        "Threads per block: %d\n",
        threads_per_block
    );


    printf(
        "\nGPU Timing\n"
    );


    printf(
        "GPU Execution Time (Kernel):    %.4f ms\n",
        kernel_time
    );


    printf(
        "GPU H2D-to-D2H Time:            %.4f ms\n",
        h2d_d2h_time
    );


    printf(
        "GPU Full Time:                  %.4f ms\n",
        full_time
    );


    printf(
        "\nSample Results\n"
    );


    printf(
        "Task 0: %llu\n",
        results[0]
    );


    printf(
        "Task 1: %llu\n",
        results[1]
    );


    printf(
        "Task 2: %llu\n",
        results[2]
    );


    printf(
        "=================================================\n"
    );


    return 0;
}