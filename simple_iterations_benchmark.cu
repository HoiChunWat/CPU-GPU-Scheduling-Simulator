#include <cuda_runtime.h>

#include <algorithm>
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
// Simple Computation - CPU
// ============================================================

unsigned long long ExecuteSimpleTaskCPU(
    unsigned long long input,
    int iterations
) {
    unsigned long long value = input;

    for (int i = 0; i < iterations; ++i) {

        value =
            value * 1664525ULL
            + 1013904223ULL;

        value ^=
            value >> 16;
    }

    return value;
}


// ============================================================
// Simple Computation - GPU
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

        value ^=
            value >> 16;
    }

    return value;
}


// ============================================================
// CUDA Kernel
// One GPU thread executes one task
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
// CPU Executor
// ============================================================

double RunSimpleCPU(
    const SimpleTask tasks[],
    int task_count,
    int iterations,
    unsigned long long results[]
) {
    auto start =
        std::chrono::steady_clock::now();


    for (int i = 0; i < task_count; ++i) {

        results[i] =
            ExecuteSimpleTaskCPU(
                tasks[i].input,
                iterations
            );
    }


    auto end =
        std::chrono::steady_clock::now();


    return std::chrono::duration<
        double,
        std::milli
    >(
        end - start
    ).count();
}


// ============================================================
// GPU Timing Result
// ============================================================

struct GPUTiming {

    double kernel_time;

    double h2d_d2h_time;

    double full_time;
};


// ============================================================
// Benchmark Result
// One result row for one Task Count
// ============================================================

struct BenchmarkResult {

    int iterations;

    int blocks;

    int matched;

    bool passed;

    double cpu_time;

    double kernel_time;

    double h2d_d2h_time;

    double full_time;

    double kernel_speedup;

    double h2d_d2h_speedup;

    double full_speedup;
};


// ============================================================
// GPU Executor
//
// Kernel:
//     Kernel only
//
// H2D-to-D2H:
//     H2D -> Kernel -> D2H
//
// Full:
//     cudaMalloc -> H2D -> Kernel -> D2H -> cudaFree
// ============================================================

GPUTiming RunSimpleGPU(
    const SimpleTask tasks[],
    int task_count,
    int iterations,
    int blocks,
    int threads_per_block,
    unsigned long long results[]
) {
    using Clock =
        std::chrono::steady_clock;


    // --------------------------------------------------------
    // CUDA Events
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
    // FULL TIME START
    // ========================================================

    auto full_start =
        Clock::now();


    // --------------------------------------------------------
    // cudaMalloc
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
    // H2D-to-D2H TIME START
    // ========================================================

    auto h2d_d2h_start =
        Clock::now();


    // --------------------------------------------------------
    // Host -> Device
    // --------------------------------------------------------

    CheckCUDA(
        cudaMemcpy(
            d_tasks,
            tasks,
            task_count * sizeof(SimpleTask),
            cudaMemcpyHostToDevice
        ),
        "Host-to-device copy failed"
    );


    // ========================================================
    // KERNEL TIME START
    // ========================================================

    CheckCUDA(
        cudaEventRecord(
            kernel_start
        ),
        "Kernel start event failed"
    );


    // --------------------------------------------------------
    // GPU Kernel
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
    // KERNEL TIME STOP
    // ========================================================

    CheckCUDA(
        cudaEventRecord(
            kernel_stop
        ),
        "Kernel stop event failed"
    );


    // --------------------------------------------------------
    // Device -> Host
    // --------------------------------------------------------

    CheckCUDA(
        cudaMemcpy(
            results,
            d_results,
            task_count *
            sizeof(unsigned long long),
            cudaMemcpyDeviceToHost
        ),
        "Device-to-host copy failed"
    );


    // ========================================================
    // H2D-to-D2H TIME STOP
    // ========================================================

    auto h2d_d2h_end =
        Clock::now();


    // --------------------------------------------------------
    // Kernel Timing
    // --------------------------------------------------------

    CheckCUDA(
        cudaEventSynchronize(
            kernel_stop
        ),
        "Kernel synchronization failed"
    );


    float kernel_time_float =
        0.0f;


    CheckCUDA(
        cudaEventElapsedTime(
            &kernel_time_float,
            kernel_start,
            kernel_stop
        ),
        "Kernel timing failed"
    );


    // --------------------------------------------------------
    // cudaFree
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
    // FULL TIME STOP
    // ========================================================

    auto full_end =
        Clock::now();


    // --------------------------------------------------------
    // Calculate Timing
    // --------------------------------------------------------

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
    // Destroy Events
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


    return {

        static_cast<double>(
            kernel_time_float
        ),

        h2d_d2h_time,

        full_time
    };
}


// ============================================================
// Median
// ============================================================

double Median(
    std::vector<double> values
) {
    std::sort(
        values.begin(),
        values.end()
    );


    int size =
        static_cast<int>(
            values.size()
        );


    if (size % 2 == 1) {

        return values[
            size / 2
        ];
    }


    return (
        values[
            size / 2 - 1
        ]
        +
        values[
            size / 2
        ]
    ) / 2.0;
}


// ============================================================
// GPU Benchmark
// ============================================================

GPUTiming BenchmarkGPU(
    int task_count,
    int iterations,
    int warmup_runs,
    int measured_runs,
    int threads_per_block,
    std::vector<unsigned long long>& final_results
) {
    const int blocks =
        (task_count + threads_per_block - 1)
        / threads_per_block;


    std::vector<SimpleTask> tasks(
        task_count
    );


    final_results.resize(
        task_count
    );


    GenerateSimpleWorkload(
        tasks.data(),
        task_count
    );


    // ========================================================
    // GPU WARM-UP
    // ========================================================

    for (int run = 0; run < warmup_runs; ++run) {

        RunSimpleGPU(
            tasks.data(),
            task_count,
            iterations,
            blocks,
            threads_per_block,
            final_results.data()
        );
    }


    // ========================================================
    // GPU MEASURED RUNS
    // ========================================================

    std::vector<double> kernel_times;

    std::vector<double> h2d_d2h_times;

    std::vector<double> full_times;


    kernel_times.reserve(
        measured_runs
    );

    h2d_d2h_times.reserve(
        measured_runs
    );

    full_times.reserve(
        measured_runs
    );


    for (int run = 0; run < measured_runs; ++run) {

        GPUTiming timing =
            RunSimpleGPU(
                tasks.data(),
                task_count,
                iterations,
                blocks,
                threads_per_block,
                final_results.data()
            );


        kernel_times.push_back(
            timing.kernel_time
        );


        h2d_d2h_times.push_back(
            timing.h2d_d2h_time
        );


        full_times.push_back(
            timing.full_time
        );
    }


    return {

        Median(
            kernel_times
        ),

        Median(
            h2d_d2h_times
        ),

        Median(
            full_times
        )
    };
}

// ============================================================
// CPU Benchmark
// ============================================================

double BenchmarkCPU(
    int task_count,
    int iterations,
    int warmup_runs,
    int measured_runs,
    std::vector<unsigned long long>& final_results
) {
    std::vector<SimpleTask> tasks(
        task_count
    );


    final_results.resize(
        task_count
    );


    GenerateSimpleWorkload(
        tasks.data(),
        task_count
    );


    // ========================================================
    // CPU WARM-UP
    // ========================================================

    for (int run = 0; run < warmup_runs; ++run) {

        RunSimpleCPU(
            tasks.data(),
            task_count,
            iterations,
            final_results.data()
        );
    }


    // ========================================================
    // CPU MEASURED RUNS
    // ========================================================

    std::vector<double> cpu_times;


    cpu_times.reserve(
        measured_runs
    );


    for (int run = 0; run < measured_runs; ++run) {

        double cpu_time =
            RunSimpleCPU(
                tasks.data(),
                task_count,
                iterations,
                final_results.data()
            );


        cpu_times.push_back(
            cpu_time
        );
    }


    return Median(
        cpu_times
    );
}


// ============================================================
// Main
// ============================================================

int main() {

    // --------------------------------------------------------
    // Fixed Configuration
    // --------------------------------------------------------

    const int task_count =
        1000;


    const int warmup_runs =
        5;


    const int measured_runs =
        10;


    const int threads_per_block =
        256;


    // --------------------------------------------------------
    // Iterations per Task Sweep
    // --------------------------------------------------------

    const std::vector<int> iteration_counts = {
        10,
        50,
        100,
        250,
        300,
        350,
        375,
        400,
        450,
        500,
        1000,
        2500,
        5000,
        10000,
        25000,
        50000,
        100000
    };


    // --------------------------------------------------------
    // CUDA Initialization
    // --------------------------------------------------------

    CheckCUDA(
        cudaFree(0),
        "CUDA initialization failed"
    );

    // --------------------------------------------------------
    // GPU Hardware / Occupancy Information
    // --------------------------------------------------------

    cudaDeviceProp device_prop;


    CheckCUDA(
        cudaGetDeviceProperties(
            &device_prop,
            0
        ),
        "cudaGetDeviceProperties failed"
    );


    int active_blocks_per_sm =
        0;


    CheckCUDA(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            SimpleKernel,
            threads_per_block,
            0
        ),
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed"
    );


    int maximum_concurrent_blocks =

        device_prop.multiProcessorCount
        * active_blocks_per_sm;


    // --------------------------------------------------------
    // Result Storage
    // --------------------------------------------------------

    std::vector<BenchmarkResult> results(
        iteration_counts.size()
    );


    std::vector<
        std::vector<unsigned long long>
    > cpu_results(
        iteration_counts.size()
    );


    std::vector<
        std::vector<unsigned long long>
    > gpu_results(
        iteration_counts.size()
    );


    // ========================================================
    // PHASE 1: GPU BENCHMARKS
    //
    // Run ALL GPU workloads first.
    // No CPU benchmark is executed during this phase.
    // ========================================================

    std::printf(
        "\n================ GPU PHASE =================\n"
    );


    for (
        std::size_t i = 0;
        i < iteration_counts.size();
        ++i
    ) {
        int iterations =
            iteration_counts[i];


        std::printf(
            "GPU Iterations = %d...\n",
            iterations
        );


        GPUTiming gpu_time =
            BenchmarkGPU(
                task_count,
                iterations,
                warmup_runs,
                measured_runs,
                threads_per_block,
                gpu_results[i]
            );


        results[i].iterations =
            iterations;


        results[i].kernel_time =
            gpu_time.kernel_time;


        results[i].h2d_d2h_time =
            gpu_time.h2d_d2h_time;


        results[i].full_time =
            gpu_time.full_time;
    }


    // ========================================================
    // PHASE 2: CPU BENCHMARKS
    //
    // GPU scaling is already completely finished.
    // ========================================================

    std::printf(
        "\n================ CPU PHASE =================\n"
    );


    for (
        std::size_t i = 0;
        i < iteration_counts.size();
        ++i
    ) {
        int iterations =
            iteration_counts[i];


        std::printf(
            "CPU Iterations = %d...\n",
            iterations
        );


        results[i].cpu_time =
            BenchmarkCPU(
                task_count,
                iterations,
                warmup_runs,
                measured_runs,
                cpu_results[i]
            );
    }


    // ========================================================
    // CORRECTNESS + SPEEDUP
    // ========================================================

    for (
        std::size_t i = 0;
        i < iteration_counts.size();
        ++i
    ) {
        int matched =
            0;


        for (
            int task = 0;
            task < task_count;
            ++task
        ) {
            if (
                cpu_results[i][task]
                ==
                gpu_results[i][task]
            ) {
                matched++;
            }
        }


        results[i].matched =
            matched;


        results[i].passed =

            (
                matched
                ==
                task_count
            );


        results[i].kernel_speedup =

            results[i].cpu_time
            /
            results[i].kernel_time;


        results[i].h2d_d2h_speedup =

            results[i].cpu_time
            /
            results[i].h2d_d2h_time;


        results[i].full_speedup =

            results[i].cpu_time
            /
            results[i].full_time;
    }


    // ========================================================
    // Configuration
    // ========================================================

    const int blocks =
        (task_count + threads_per_block - 1)
        / threads_per_block;

    std::printf(
        "\n"
        "============= ITERATIONS PER TASK SCALING =============\n"
    );


    std::printf(
        "Tasks: %d\n",
        task_count
    );

    std::printf(
        "Blocks:              %d\n",
        blocks
    );

    std::printf(
        "Warm-up Runs:        %d\n",
        warmup_runs
    );


    std::printf(
        "Measured Runs:       %d\n",
        measured_runs
    );


    std::printf(
        "Reported Statistic:  Median\n"
    );


    std::printf(
        "Threads per Block:   %d\n",
        threads_per_block
    );


    std::printf(
        "\nGPU Configuration\n"
        "-----------------------------------------------------\n"
    );


    std::printf(
        "GPU:                  %s\n",
        device_prop.name
    );


    std::printf(
        "SM Count:             %d\n",
        device_prop.multiProcessorCount
    );


    std::printf(
        "Active Blocks / SM:   %d\n",
        active_blocks_per_sm
    );


    std::printf(
        "Maximum Concurrent Blocks: %d\n\n",
        maximum_concurrent_blocks
    );


    // ========================================================
    // Performance Table
    // ========================================================

    std::printf(
        "%-12s %-10s %-10s %-12s %-10s "
        "%-11s %-11s %-11s %-8s\n",
        "Iterations",
        "CPU(ms)",
        "Kernel",
        "H2D-D2H",
        "Full(ms)",
        "K-Speedup",
        "HD-Speedup",
        "F-Speedup",
        "Check"
    );


    std::printf(
        "------------------------------------------------------------"
        "--------------------------------------------------\n"
    );


    for (const BenchmarkResult& result : results) {

        std::printf(
            "%-12d "
            "%-10.4f "
            "%-10.4f "
            "%-12.4f "
            "%-10.4f "
            "%-10.2f "
            "%-10.2f "
            "%-10.2f "
            "%-8s\n",

            result.iterations,

            result.cpu_time,

            result.kernel_time,

            result.h2d_d2h_time,

            result.full_time,

            result.kernel_speedup,

            result.h2d_d2h_speedup,

            result.full_speedup,

            result.passed
                ? "PASS"
                : "FAIL"
        );
    }


    std::printf(
        "================================================================"
        "==============================================\n"
    );


    // ========================================================
    // Final Correctness Status
    // ========================================================

    bool all_passed =
        true;


    for (const BenchmarkResult& result : results) {

        if (!result.passed) {

            all_passed =
                false;
        }
    }


    std::printf(
        "\nAll Correctness Tests: %s\n",
        all_passed
            ? "PASSED"
            : "FAILED"
    );


    return all_passed
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
