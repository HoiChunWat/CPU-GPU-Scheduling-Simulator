#include <cuda_runtime.h>

#include <string>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <set>
#include <tuple>
#include <vector>


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
// Matrix Input Generation
//
// A: M × K
// B: K × N
// C: M × N
//
// Deterministic input:
// Same M/K/N always generates the same matrices.
// ============================================================

void GenerateMatrices(
    float* A,
    float* B,
    int M,
    int K,
    int N
) {
    for (int row = 0; row < M; ++row) {

        for (int col = 0; col < K; ++col) {

            A[row * K + col] =
                static_cast<float>(
                    (row + col) % 10 + 1
                );
        }
    }


    for (int row = 0; row < K; ++row) {

        for (int col = 0; col < N; ++col) {

            B[row * N + col] =
                static_cast<float>(
                    (row * 2 + col) % 10 + 1
                );
        }
    }
}


// ============================================================
// CPU Matrix Multiplication
//
// C = A × B
//
// A: M × K
// B: K × N
// C: M × N
// ============================================================

void MatrixMultiplyCPU(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    for (int row = 0; row < M; ++row) {

        for (int col = 0; col < N; ++col) {

            float sum =
                0.0f;


            for (int k = 0; k < K; ++k) {

                sum +=
                    A[row * K + k]
                    *
                    B[k * N + col];
            }


            C[row * N + col] =
                sum;
        }
    }
}


// ============================================================
// CPU Matrix Multiplication Executor
// ============================================================

double RunMatrixCPU(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    auto start =
        std::chrono::steady_clock::now();


    MatrixMultiplyCPU(
        A,
        B,
        C,
        M,
        K,
        N
    );


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
// Naive GPU Matrix Multiplication
//
// One GPU thread computes one element of C.
//
// A: M × K
// B: K × N
// C: M × N
// ============================================================

__global__
void MatrixMultiplyKernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    int row =
        blockIdx.y * blockDim.y
        + threadIdx.y;


    int col =
        blockIdx.x * blockDim.x
        + threadIdx.x;


    if (
        row < M
        &&
        col < N
    ) {
        float sum =
            0.0f;


        for (int k = 0; k < K; ++k) {

            sum +=
                A[row * K + k]
                *
                B[k * N + col];
        }


        C[row * N + col] =
            sum;
    }
}


// ============================================================
// GPU Timing Result
//
// Kernel:
//     Kernel only
//
// H2D-to-D2H:
//     H2D + Kernel + D2H
//
// Full:
//     cudaMalloc + H2D + Kernel + D2H + cudaFree
// ============================================================

struct MatrixGPUTiming {

    double kernel_time;

    double h2d_d2h_time;

    double full_time;
};


// ============================================================
// GPU Matrix Multiplication Executor
// ============================================================

MatrixGPUTiming RunMatrixGPU(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    using Clock =
        std::chrono::steady_clock;


    // --------------------------------------------------------
    // Matrix Sizes
    // --------------------------------------------------------

    size_t size_A =
        static_cast<size_t>(M)
        * K
        * sizeof(float);


    size_t size_B =
        static_cast<size_t>(K)
        * N
        * sizeof(float);


    size_t size_C =
        static_cast<size_t>(M)
        * N
        * sizeof(float);


    // --------------------------------------------------------
    // CUDA Events
    // --------------------------------------------------------

    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;


    CheckCUDA(
        cudaEventCreate(
            &kernel_start
        ),
        "cudaEventCreate kernel_start failed"
    );


    CheckCUDA(
        cudaEventCreate(
            &kernel_stop
        ),
        "cudaEventCreate kernel_stop failed"
    );


    // --------------------------------------------------------
    // Device Memory
    // --------------------------------------------------------

    float* d_A =
        nullptr;

    float* d_B =
        nullptr;

    float* d_C =
        nullptr;


    // ========================================================
    // FULL TIME START
    // ========================================================

    auto full_start =
        Clock::now();


    CheckCUDA(
        cudaMalloc(
            &d_A,
            size_A
        ),
        "cudaMalloc A failed"
    );


    CheckCUDA(
        cudaMalloc(
            &d_B,
            size_B
        ),
        "cudaMalloc B failed"
    );


    CheckCUDA(
        cudaMalloc(
            &d_C,
            size_C
        ),
        "cudaMalloc C failed"
    );


    // ========================================================
    // H2D-to-D2H TIME START
    // ========================================================

    auto h2d_d2h_start =
        Clock::now();


    CheckCUDA(
        cudaMemcpy(
            d_A,
            A,
            size_A,
            cudaMemcpyHostToDevice
        ),
        "H2D copy A failed"
    );


    CheckCUDA(
        cudaMemcpy(
            d_B,
            B,
            size_B,
            cudaMemcpyHostToDevice
        ),
        "H2D copy B failed"
    );


    // --------------------------------------------------------
    // 2D Block / Grid
    // --------------------------------------------------------

    dim3 block(
        16,
        16
    );


    dim3 grid(

        (N + block.x - 1)
        / block.x,

        (M + block.y - 1)
        / block.y
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


    MatrixMultiplyKernel<<<
        grid,
        block
    >>>(
        d_A,
        d_B,
        d_C,
        M,
        K,
        N
    );


    CheckCUDA(
        cudaGetLastError(),
        "Matrix kernel launch failed"
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
    //
    // D2H cudaMemcpy waits for kernel completion.
    // --------------------------------------------------------

    CheckCUDA(
        cudaMemcpy(
            C,
            d_C,
            size_C,
            cudaMemcpyDeviceToHost
        ),
        "D2H copy C failed"
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
    // Free Device Memory
    // --------------------------------------------------------

    CheckCUDA(
        cudaFree(
            d_A
        ),
        "cudaFree A failed"
    );


    CheckCUDA(
        cudaFree(
            d_B
        ),
        "cudaFree B failed"
    );


    CheckCUDA(
        cudaFree(
            d_C
        ),
        "cudaFree C failed"
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
            -
            h2d_d2h_start
        ).count();


    double full_time =

        std::chrono::duration<
            double,
            std::milli
        >(
            full_end
            -
            full_start
        ).count();


    // --------------------------------------------------------
    // Destroy Events
    // --------------------------------------------------------

    CheckCUDA(
        cudaEventDestroy(
            kernel_start
        ),
        "cudaEventDestroy kernel_start failed"
    );


    CheckCUDA(
        cudaEventDestroy(
            kernel_stop
        ),
        "cudaEventDestroy kernel_stop failed"
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


    const int size =
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
// GPU Matrix Benchmark
// ============================================================

MatrixGPUTiming BenchmarkMatrixGPU(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N,
    int warmup_runs,
    int measured_runs
) {
    // --------------------------------------------------------
    // Warm-up Runs
    // --------------------------------------------------------

    for (int run = 0; run < warmup_runs; ++run) {

        RunMatrixGPU(
            A,
            B,
            C,
            M,
            K,
            N
        );
    }


    // --------------------------------------------------------
    // Measured Runs
    // --------------------------------------------------------

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

        MatrixGPUTiming timing =
            RunMatrixGPU(
                A,
                B,
                C,
                M,
                K,
                N
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
// CPU Matrix Benchmark
// ============================================================

double BenchmarkMatrixCPU(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N,
    int warmup_runs,
    int measured_runs
) {
    // --------------------------------------------------------
    // Warm-up Runs
    // --------------------------------------------------------

    for (int run = 0; run < warmup_runs; ++run) {

        RunMatrixCPU(
            A,
            B,
            C,
            M,
            K,
            N
        );
    }


    // --------------------------------------------------------
    // Measured Runs
    // --------------------------------------------------------

    std::vector<double> cpu_times;


    cpu_times.reserve(
        measured_runs
    );


    for (int run = 0; run < measured_runs; ++run) {

        double cpu_time =
            RunMatrixCPU(
                A,
                B,
                C,
                M,
                K,
                N
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
// Matrix Correctness Check
// ============================================================

bool CompareMatrices(
    const float* cpu_result,
    const float* gpu_result,
    int M,
    int N,
    float tolerance
) {
    for (int row = 0; row < M; ++row) {

        for (int col = 0; col < N; ++col) {

            const int index =
                row * N + col;


            const float difference =
                std::fabs(
                    cpu_result[index]
                    -
                    gpu_result[index]
                );


            if (difference > tolerance) {

                std::printf(
                    "\nMismatch at C[%d][%d]: "
                    "CPU = %.6f, GPU = %.6f\n",
                    row,
                    col,
                    cpu_result[index],
                    gpu_result[index]
                );


                return false;
            }
        }
    }


    return true;
}


// ============================================================
// Workload Result
//
// Execution(K):
//     Work performed per output element / GPU thread.
//
// Output(M*N):
//     Number of C output elements.
//     Also logical GPU parallelism.
//
// TotalWork(M*K*N):
//     Overall computation scale.
//
// R:
//     MKN / [2(MK + KN + MN)]
// ============================================================

struct MatrixWorkloadResult {

    int sample_id;

    int M;

    int K;

    int N;


    unsigned long long output_elements;

    unsigned long long total_work;

    unsigned long long operations;

    unsigned long long transfer_bytes;


    int grid_x;

    int grid_y;

    int total_blocks;


    double R;


    double cpu_time;

    double kernel_time;

    double h2d_d2h_time;

    double full_time;


    double kernel_speedup;

    double h2d_d2h_speedup;

    double full_speedup;


    bool passed;
};


// ============================================================
// Calculate Workload Features
//
// Execution(K) = K
//
// Output(M*N) = M*N
//
// TotalWork(M*K*N) = M*K*N
//
// Approximate Operations = 2*M*K*N
//
// TransferBytes =
//     4 * (M*K + K*N + M*N)
//
// R =
//     M*K*N
//     -----------------------
//     2(M*K + K*N + M*N)
// ============================================================

void CalculateFeatures(
    MatrixWorkloadResult& result
) {
    const unsigned long long M =
        static_cast<unsigned long long>(
            result.M
        );


    const unsigned long long K =
        static_cast<unsigned long long>(
            result.K
        );


    const unsigned long long N =
        static_cast<unsigned long long>(
            result.N
        );


    result.output_elements =
        M * N;


    result.total_work =
        M * K * N;


    result.operations =
        2ULL
        * result.total_work;


    const unsigned long long transfer_elements =

        M * K
        +
        K * N
        +
        M * N;


    result.transfer_bytes =
        4ULL
        *
        transfer_elements;


    result.R =

        static_cast<double>(
            result.total_work
        )

        /

        (
            2.0
            *
            static_cast<double>(
                transfer_elements
            )
        );


    result.grid_x =
        (
            result.N
            + 16
            - 1
        )
        / 16;


    result.grid_y =
        (
            result.M
            + 16
            - 1
        )
        / 16;


    result.total_blocks =
        result.grid_x
        *
        result.grid_y;
}

// ============================================================
// Format Large Integer
//
// Example:
// 2359296 -> 2,359,296
// ============================================================

std::string FormatNumber(
    unsigned long long value
) {
    std::string text =
        std::to_string(value);


    for (
        int position =
            static_cast<int>(text.size()) - 3;

        position > 0;

        position -= 3
    ) {
        text.insert(
            position,
            ","
        );
    }


    return text;
}

// ============================================================
// Main
//
// Random M/K/N Experiment
//
// Goal:
// Test whether R can predict CPU vs GPU offloading.
//
// Sample selection:
// 1. Generate many random candidate M/K/N combinations.
// 2. Calculate R for all candidates.
// 3. Sort candidates by R.
// 4. Select 200 workloads evenly across the R range.
//
// Benchmark execution order is shuffled.
// Results are sorted by R only after timing is complete.
// ============================================================

int main() {

    // --------------------------------------------------------
    // Benchmark Configuration
    // --------------------------------------------------------

    const int candidate_count =
        1200;


    const int sample_count =
        200;


    const int warmup_runs =
        5;


    const int measured_runs =
        10;


    const float tolerance =
        1e-4f;


    // --------------------------------------------------------
    // Candidate Dimensions
    // --------------------------------------------------------

    const std::vector<int> dimension_choices = {

        16,
        32,
        48,
        64,
        96,
        128,
        160,
        192,
        256,
        320,
        384,
        512
    };


    // --------------------------------------------------------
    // Reproducible Random Generator
    // --------------------------------------------------------

    const unsigned int workload_seed =
        20261005;


    std::mt19937 workload_rng(
        workload_seed
    );


    std::uniform_int_distribution<int> dimension_distribution(
        0,
        static_cast<int>(
            dimension_choices.size()
        ) - 1
    );


    // ========================================================
    // GENERATE RANDOM CANDIDATES
    // ========================================================

    std::vector<MatrixWorkloadResult> candidates;


    candidates.reserve(
        candidate_count
    );


    std::set<
        std::tuple<int, int, int>
    > used_dimensions;


    while (
        static_cast<int>(
            candidates.size()
        )
        <
        candidate_count
    ) {
        int M =
            dimension_choices[
                dimension_distribution(
                    workload_rng
                )
            ];


        int K =
            dimension_choices[
                dimension_distribution(
                    workload_rng
                )
            ];


        int N =
            dimension_choices[
                dimension_distribution(
                    workload_rng
                )
            ];


        std::tuple<int, int, int> key = {
            M,
            K,
            N
        };


        if (
            !used_dimensions.insert(
                key
            ).second
        ) {
            continue;
        }


        MatrixWorkloadResult result{};


        result.M =
            M;


        result.K =
            K;


        result.N =
            N;


        CalculateFeatures(
            result
        );


        candidates.push_back(
            result
        );
    }


    // --------------------------------------------------------
    // Sort Candidates by R
    // --------------------------------------------------------

    std::sort(
        candidates.begin(),
        candidates.end(),

        [](
            const MatrixWorkloadResult& a,
            const MatrixWorkloadResult& b
        ) {
            return a.R
                <
                b.R;
        }
    );


    // ========================================================
    // SELECT 200 SAMPLES EVENLY ACROSS THE R RANGE
    // ========================================================

    std::vector<MatrixWorkloadResult> results;


    results.reserve(
        sample_count
    );


    for (int i = 0; i < sample_count; ++i) {

        const double fraction =

            static_cast<double>(i)

            /

            static_cast<double>(
                sample_count - 1
            );


        const std::size_t candidate_index =

            static_cast<std::size_t>(
                std::llround(
                    fraction
                    *
                    static_cast<double>(
                        candidate_count - 1
                    )
                )
            );


        MatrixWorkloadResult result =
            candidates[
                candidate_index
            ];


        result.sample_id =
            i + 1;


        results.push_back(
            result
        );
    }


    // --------------------------------------------------------
    // Initialize CUDA Before Timing
    // --------------------------------------------------------

    CheckCUDA(
        cudaFree(0),
        "CUDA initialization failed"
    );


    // --------------------------------------------------------
    // GPU Information
    // --------------------------------------------------------

    cudaDeviceProp device_properties;


    CheckCUDA(
        cudaGetDeviceProperties(
            &device_properties,
            0
        ),
        "cudaGetDeviceProperties failed"
    );


    std::printf(
        "\n"
        "============================================================\n"
        " CPU-GPU MATRIX OFFLOADING BENCHMARK\n"
        "============================================================\n"
    );


    std::printf(
        "GPU      : %s\n",
        device_properties.name
    );


    std::printf(
        "Samples  : %d | Warm-up: %d | Measured: %d | Median\n",
        sample_count,
        warmup_runs,
        measured_runs
    );


    std::printf(
        "CUDA     : Block 16x16 | Candidates: %d | Seed: %u\n",
        candidate_count,
        workload_seed
    );


    std::printf(
        "Feature R: MKN / [2(MK + KN + MN)]\n"
    );


    // ========================================================
    // CREATE RANDOM EXECUTION ORDERS
    // ========================================================

    std::vector<int> gpu_order(
        sample_count
    );


    std::iota(
        gpu_order.begin(),
        gpu_order.end(),
        0
    );


    std::mt19937 gpu_order_rng(
        20261006
    );


    std::shuffle(
        gpu_order.begin(),
        gpu_order.end(),
        gpu_order_rng
    );


    std::vector<int> cpu_order(
        sample_count
    );


    std::iota(
        cpu_order.begin(),
        cpu_order.end(),
        0
    );


    std::mt19937 cpu_order_rng(
        20261007
    );


    std::shuffle(
        cpu_order.begin(),
        cpu_order.end(),
        cpu_order_rng
    );


    // --------------------------------------------------------
    // GPU Output Storage for Correctness Checking
    // --------------------------------------------------------

    std::vector<
        std::vector<float>
    > gpu_results(
        sample_count
    );


    // ========================================================
    // PHASE 1: GPU
    // ========================================================

    std::printf(
        "\n"
        "================ GPU PHASE =================\n"
        "Running %d workloads...\n",
        sample_count
    );


    for (
        int position = 0;
        position < sample_count;
        ++position
    ) {
        const int index =
            gpu_order[position];


        MatrixWorkloadResult& result =
            results[index];


        const int M =
            result.M;


        const int K =
            result.K;


        const int N =
            result.N;


        std::vector<float> A(
            static_cast<size_t>(M)
            * K
        );


        std::vector<float> B(
            static_cast<size_t>(K)
            * N
        );


        gpu_results[index].resize(
            static_cast<size_t>(M)
            * N
        );


        GenerateMatrices(
            A.data(),
            B.data(),
            M,
            K,
            N
        );


        MatrixGPUTiming timing =
            BenchmarkMatrixGPU(
                A.data(),
                B.data(),
                gpu_results[index].data(),
                M,
                K,
                N,
                warmup_runs,
                measured_runs
            );


        result.kernel_time =
            timing.kernel_time;


        result.h2d_d2h_time =
            timing.h2d_d2h_time;


        result.full_time =
            timing.full_time;
    }


    // ========================================================
    // PHASE 2: CPU
    // ========================================================

    std::printf(
        "\n"
        "================ CPU PHASE =================\n"
        "Running %d workloads...\n",
        sample_count
    );


    for (
        int position = 0;
        position < sample_count;
        ++position
    ) {
        const int index =
            cpu_order[position];


        MatrixWorkloadResult& result =
            results[index];


        const int M =
            result.M;


        const int K =
            result.K;


        const int N =
            result.N;


        std::vector<float> A(
            static_cast<size_t>(M)
            * K
        );


        std::vector<float> B(
            static_cast<size_t>(K)
            * N
        );


        std::vector<float> C_cpu(
            static_cast<size_t>(M)
            * N
        );


        GenerateMatrices(
            A.data(),
            B.data(),
            M,
            K,
            N
        );


        result.cpu_time =
            BenchmarkMatrixCPU(
                A.data(),
                B.data(),
                C_cpu.data(),
                M,
                K,
                N,
                warmup_runs,
                measured_runs
            );


        result.passed =
            CompareMatrices(
                C_cpu.data(),
                gpu_results[index].data(),
                M,
                N,
                tolerance
            );


        result.kernel_speedup =

            result.cpu_time
            /
            result.kernel_time;


        result.h2d_d2h_speedup =

            result.cpu_time
            /
            result.h2d_d2h_time;


        result.full_speedup =

            result.cpu_time
            /
            result.full_time;


        // GPU output no longer needed.

        std::vector<float>().swap(
            gpu_results[index]
        );
    }


    // ============================================================
    // SORT FINAL RESULTS
    //
    // 1. Work(M*K*N) : small -> large
    // 2. Same Work   : K small -> large
    // 3. Same K      : M small -> large
    // 4. Same M      : N small -> large
    // ============================================================

    std::sort(
        results.begin(),
        results.end(),

        [](
            const MatrixWorkloadResult& a,
            const MatrixWorkloadResult& b
        ) {
            if (
                a.total_work
                !=
                b.total_work
            ) {
                return a.total_work
                    <
                    b.total_work;
            }


            if (
                a.K
                !=
                b.K
            ) {
                return a.K
                    <
                    b.K;
            }


            if (
                a.M
                !=
                b.M
            ) {
                return a.M
                    <
                    b.M;
            }


            return a.N
                <
                b.N;
        }
    );


    // ============================================================
    // WRITE CSV
    // ============================================================

    const char* csv_filename =
        "matrix_random_work_results.csv";


    std::FILE* csv =
        std::fopen(
            csv_filename,
            "w"
        );


    if (csv == nullptr) {

        std::fprintf(
            stderr,
            "Failed to create CSV file.\n"
        );

        return EXIT_FAILURE;
    }


    // ------------------------------------------------------------
    // CSV Header
    // ------------------------------------------------------------

    std::fprintf(
        csv,

        "Rank,"
        "SampleID,"
        "M,"
        "K,"
        "N,"
        "Work_MKN,"
        "Out_MN,"
        "R,"
        "CPU_ms,"
        "Kernel_ms,"
        "H2D_D2H_ms,"
        "Full_ms,"
        "KernelSpeedup,"
        "H2D_D2H_Speedup,"
        "FullSpeedup,"
        "Winner,"
        "Correctness,"
        "Operations_2MKN,"
        "TransferBytes,"
        "GridX,"
        "GridY,"
        "TotalBlocks\n"
    );


    // ============================================================
    // SUMMARY VARIABLES
    // ============================================================

    int cpu_wins =
        0;


    int gpu_wins =
        0;


    bool all_passed =
        true;


    // ------------------------------------------------------------
    // Level 1: Work(MKN) Boundary
    // ------------------------------------------------------------

    bool gpu_work_found =
        false;


    bool cpu_work_found =
        false;


    unsigned long long first_gpu_win_work =
        0;


    unsigned long long last_cpu_win_work =
        0;

    // ------------------------------------------------------------
    // Row positions used ONLY for terminal overlap markers.
    // They do NOT change the Work/K decision boundaries.
    // ------------------------------------------------------------

    bool gpu_rank_found = false;
    bool cpu_rank_found = false;

    std::size_t first_gpu_win_rank = 0;
    std::size_t last_cpu_win_rank = 0;

    // ============================================================
    // PASS 1
    //
    // Find:
    //
    // Lowest Work where GPU wins
    // Highest Work where CPU wins
    //
    // These define the observed Work overlap.
    // ============================================================

    for (
        std::size_t rank = 0;
        rank < results.size();
        ++rank
    ) {
        const MatrixWorkloadResult& result =
            results[rank];

        const bool gpu_winner =
            result.full_speedup > 1.0;

        if (gpu_winner) {

            ++gpu_wins;

            // Lowest Work where GPU wins
            if (
                !gpu_work_found
                ||
                result.total_work < first_gpu_win_work
            ) {
                first_gpu_win_work =
                    result.total_work;

                gpu_work_found =
                    true;
            }

            // First actual GPU-winning ROW
            if (!gpu_rank_found) {

                first_gpu_win_rank =
                    rank;

                gpu_rank_found =
                    true;
            }
        }
        else {

            ++cpu_wins;

            // Highest Work where CPU wins
            if (
                !cpu_work_found
                ||
                result.total_work > last_cpu_win_work
            ) {
                last_cpu_win_work =
                    result.total_work;

                cpu_work_found =
                    true;
            }

            // Because results are sorted,
            // continually update to the latest CPU-winning row.
            last_cpu_win_rank =
                rank;

            cpu_rank_found =
                true;
        }

        if (!result.passed) {

            all_passed =
                false;
        }
    }


    // ============================================================
    // CHECK WHETHER WORK OVERLAPS
    // ============================================================

    const bool work_overlap_exists =

        gpu_work_found

        &&

        cpu_work_found

        &&

        first_gpu_win_work
        <=
        last_cpu_win_work;

    const bool row_overlap_exists =

        gpu_rank_found

        &&

        cpu_rank_found

        &&

        first_gpu_win_rank
        <=
        last_cpu_win_rank;
    // ============================================================
    // LEVEL 2: K ANALYSIS
    //
    // IMPORTANT:
    //
    // Only examine K inside the Work(MKN) overlap.
    //
    // Find:
    //
    // Lowest K where GPU wins
    // Highest K where CPU wins
    //
    // If:
    //
    // lowest_GPU_K <= highest_CPU_K
    //
    // then K also has an overlap region.
    // ============================================================

    bool gpu_k_found =
        false;


    bool cpu_k_found =
        false;


    int first_gpu_win_k =
        0;


    int last_cpu_win_k =
        0;


    int work_overlap_samples =
        0;


    if (work_overlap_exists) {

        for (
            const MatrixWorkloadResult& result
            :
            results
        ) {
            const bool inside_work_overlap =

                result.total_work
                    >=
                    first_gpu_win_work

                &&

                result.total_work
                    <=
                    last_cpu_win_work;


            if (!inside_work_overlap) {

                continue;
            }


            ++work_overlap_samples;


            const bool gpu_winner =
                result.full_speedup
                >
                1.0;


            if (gpu_winner) {

                if (
                    !gpu_k_found
                    ||
                    result.K
                    <
                    first_gpu_win_k
                ) {
                    first_gpu_win_k =
                        result.K;


                    gpu_k_found =
                        true;
                }
            }
            else {

                if (
                    !cpu_k_found
                    ||
                    result.K
                    >
                    last_cpu_win_k
                ) {
                    last_cpu_win_k =
                        result.K;


                    cpu_k_found =
                        true;
                }
            }
        }
    }


    const bool k_overlap_exists =

        gpu_k_found

        &&

        cpu_k_found

        &&

        first_gpu_win_k
        <=
        last_cpu_win_k;


    // ============================================================
    // WRITE CSV ROWS
    //
    // CSV stays machine-friendly:
    // no commas inside integer values.
    // ============================================================

    for (
        std::size_t rank = 0;
        rank < results.size();
        ++rank
    ) {
        const MatrixWorkloadResult& result =
            results[rank];


        const char* winner =

            result.full_speedup
                >
                1.0

            ? "GPU"
            : "CPU";


        std::fprintf(
            csv,

            "%zu,"
            "%d,"
            "%d,"
            "%d,"
            "%d,"
            "%llu,"
            "%llu,"
            "%.8f,"
            "%.6f,"
            "%.6f,"
            "%.6f,"
            "%.6f,"
            "%.6f,"
            "%.6f,"
            "%.6f,"
            "%s,"
            "%s,"
            "%llu,"
            "%llu,"
            "%d,"
            "%d,"
            "%d\n",

            rank + 1,

            result.sample_id,

            result.M,

            result.K,

            result.N,

            result.total_work,

            result.output_elements,

            result.R,

            result.cpu_time,

            result.kernel_time,

            result.h2d_d2h_time,

            result.full_time,

            result.kernel_speedup,

            result.h2d_d2h_speedup,

            result.full_speedup,

            winner,

            result.passed
                ? "PASS"
                : "FAIL",

            result.operations,

            result.transfer_bytes,

            result.grid_x,

            result.grid_y,

            result.total_blocks
        );
    }


    std::fclose(
        csv
    );


    // ============================================================
    // TERMINAL TABLE
    //
    // Sort:
    //
    // Work(MKN) ascending
    //
    // Same Work:
    //
    // K ascending
    //
    // Visual sections:
    //
    // WORKLOAD | TIMING | SPEEDUP
    // ============================================================

    std::printf(
        "\n"
        "======================================= RESULTS BY WORK(MKN) =======================================\n"
        "\n"
    );


    std::printf(
        "                     WORKLOAD"
        "                                      "
        "TIMING"
        "                         "
        "SPEEDUP\n"
    );


    std::printf(
        "%-4s | "
        "%4s %4s %4s | "
        "%13s %10s %8s || "
        "%9s %8s %9s %9s || "
        "%7s %7s %7s | "
        "%6s | "
        "%5s\n",

        "Rank",

        "M",
        "K",
        "N",

        "Work(MKN)",
        "Out(MN)",
        "R",

        "CPU(ms)",
        "Kernel",
        "H2D-D2H",
        "Full(ms)",

        "K-Spd",
        "HD-Spd",
        "F-Spd",

        "Winner",
        "Check"
    );


    std::printf(
        "-----+---------------+------------------------------------++"
        "------------------------------------------++"
        "-------------------------+--------+------\n"
    );


    // ------------------------------------------------------------
    // Table Rows + Work Overlap Markers
    // ------------------------------------------------------------

    bool overlap_started =
        false;


    bool overlap_ended =
        false;


    for (
        std::size_t rank = 0;
        rank < results.size();
        ++rank
    ) {
        const MatrixWorkloadResult& result =
            results[rank];


        // --------------------------------------------------------
        // Enter Work Overlap
        // --------------------------------------------------------

        if (
            work_overlap_exists

            &&

            !overlap_started

            &&

            rank == first_gpu_win_rank
        ) {
            std::string overlap_low =
                FormatNumber(
                    first_gpu_win_work
                );


            std::string overlap_high =
                FormatNumber(
                    last_cpu_win_work
                );


            std::printf(
                "\n"
                "========================= OBSERVED WORK OVERLAP START =========================\n"
            );


            std::printf(
                " Work(MKN): %s ~ %s"
                "   |   Within this region, inspect K\n",
                overlap_low.c_str(),
                overlap_high.c_str()
            );


            std::printf(
                "===============================================================================\n"
            );


            overlap_started =
                true;
        }


        const char* winner =

            result.full_speedup
                >
                1.0

            ? "GPU"
            : "CPU";


        std::string work_text =
            FormatNumber(
                result.total_work
            );


        std::string output_text =
            FormatNumber(
                result.output_elements
            );


        std::printf(
            "%4zu | "
            "%4d %4d %4d | "
            "%13s %10s %8.4f || "
            "%9.4f %8.4f %9.4f %9.4f || "
            "%7.2f %7.2f %7.2f | "
            "%6s | "
            "%5s\n",

            rank + 1,

            result.M,
            result.K,
            result.N,

            work_text.c_str(),
            output_text.c_str(),
            result.R,

            result.cpu_time,
            result.kernel_time,
            result.h2d_d2h_time,
            result.full_time,

            result.kernel_speedup,
            result.h2d_d2h_speedup,
            result.full_speedup,

            winner,

            result.passed
                ? "PASS"
                : "FAIL"
        );


        // --------------------------------------------------------
        // Leave Work Overlap
        // --------------------------------------------------------

        if (
            row_overlap_exists
            &&
            overlap_started
            &&
            !overlap_ended
            &&
            rank == last_cpu_win_rank
        ) {
            std::printf(
                "========================== OBSERVED WORK OVERLAP END ==========================\n"
                "\n"
            );


            overlap_ended =
                true;
        }
    }


    // ============================================================
    // DECISION SUMMARY
    // ============================================================

    std::printf(
        "\n"
        "============================================================\n"
        "                    OFFLOADING DECISION\n"
        "============================================================\n"
    );


    std::printf(
        "Observed from this benchmark dataset\n"
        "\n"
    );


    // ============================================================
    // LEVEL 1 — TOTAL WORK
    // ============================================================

    std::printf(
        "[LEVEL 1] TOTAL WORK — M*K*N\n"
        "\n"
    );


    if (
        gpu_work_found
        &&
        cpu_work_found
    ) {
        std::string low_work =
            FormatNumber(
                first_gpu_win_work
            );


        std::string high_work =
            FormatNumber(
                last_cpu_win_work
            );


        if (work_overlap_exists) {

            std::printf(
                "  Observed CPU-only   : Work(MKN) < %s\n",
                low_work.c_str()
            );


            std::printf(
                "  WORK OVERLAP        : %s ~ %s\n",
                low_work.c_str(),
                high_work.c_str()
            );


            std::printf(
                "  Observed GPU-only   : Work(MKN) > %s\n",
                high_work.c_str()
            );


            std::printf(
                "  Samples in overlap  : %d\n",
                work_overlap_samples
            );
        }
        else {

            std::printf(
                "  Work separation     : CLEAN\n"
            );


            std::printf(
                "  Highest CPU Work    : %s\n",
                high_work.c_str()
            );


            std::printf(
                "  Lowest GPU Work     : %s\n",
                low_work.c_str()
            );
        }
    }
    else {

        std::printf(
            "  Insufficient CPU/GPU winner classes.\n"
        );
    }


    // ============================================================
    // LEVEL 2 — K
    // ============================================================

    std::printf(
        "\n"
        "[LEVEL 2] K — WITHIN WORK OVERLAP ONLY\n"
        "\n"
    );


    if (!work_overlap_exists) {

        std::printf(
            "  Not required: Work already separates CPU and GPU.\n"
        );
    }
    else if (
        !gpu_k_found
        ||
        !cpu_k_found
    ) {

        std::printf(
            "  Insufficient CPU/GPU samples inside Work overlap.\n"
        );
    }
    else {

        std::printf(
            "  Lowest GPU-winning K : %d\n",
            first_gpu_win_k
        );


        std::printf(
            "  Highest CPU-winning K: %d\n",
            last_cpu_win_k
        );


        if (k_overlap_exists) {

            std::printf(
                "\n"
            );


            std::printf(
                "  Observed CPU-only K  : K < %d\n",
                first_gpu_win_k
            );


            std::printf(
                "  K OVERLAP            : %d ~ %d\n",
                first_gpu_win_k,
                last_cpu_win_k
            );


            std::printf(
                "  Observed GPU-only K  : K > %d\n",
                last_cpu_win_k
            );
        }
        else {

            std::printf(
                "\n"
                "  K separation         : CLEAN\n"
            );


            std::printf(
                "  CPU observed through : K <= %d\n",
                last_cpu_win_k
            );


            std::printf(
                "  GPU observed from    : K >= %d\n",
                first_gpu_win_k
            );
        }
    }


    // ============================================================
    // DECISION PATH
    // ============================================================

    std::printf(
        "\n"
        "----------------------- DECISION PATH ------------------------\n"
    );


    if (work_overlap_exists) {

        std::string low_work =
            FormatNumber(
                first_gpu_win_work
            );


        std::string high_work =
            FormatNumber(
                last_cpu_win_work
            );


        std::printf(
            "\n"
            "                      Work(MKN)\n"
            "                          |\n"
            "             +------------+------------+\n"
            "             |                         |\n"
            "       Work < %-12s       Work > %-12s\n"
            "             |                         |\n"
            "            CPU                       GPU\n"
            "             \\                        /\n"
            "              \\_____ Work overlap ___/\n"
            "                          |\n"
            "                          K\n",
            low_work.c_str(),
            high_work.c_str()
        );


        if (
            gpu_k_found
            &&
            cpu_k_found
        ) {
            if (k_overlap_exists) {

                std::printf(
                    "             +------------+------------+\n"
                    "             |            |            |\n"
                    "          K < %-5d    %3d ~ %-3d    K > %-5d\n"
                    "             |            |            |\n"
                    "            CPU       ambiguous        GPU\n",

                    first_gpu_win_k,

                    first_gpu_win_k,
                    last_cpu_win_k,

                    last_cpu_win_k
                );
            }
            else {

                std::printf(
                    "             +-------------------------+\n"
                    "             |                         |\n"
                    "        K <= %-5d                 K >= %-5d\n"
                    "             |                         |\n"
                    "            CPU                       GPU\n",

                    last_cpu_win_k,
                    first_gpu_win_k
                );
            }
        }
    }
    else {

        std::printf(
            "\n"
            "  Work(MKN) already provides a clean observed separation.\n"
        );
    }


    // ============================================================
    // BENCHMARK STATUS
    // ============================================================

    std::printf(
        "\n"
        "------------------------- STATUS -----------------------------\n"
    );


    std::printf(
        "CPU Wins     : %d\n",
        cpu_wins
    );


    std::printf(
        "GPU Wins     : %d\n",
        gpu_wins
    );


    std::printf(
        "Correctness  : %s\n",
        all_passed
            ? "PASS"
            : "FAIL"
    );


    std::printf(
        "CSV          : %s\n",
        csv_filename
    );


    std::printf(
        "============================================================\n"
    );


    return all_passed
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}