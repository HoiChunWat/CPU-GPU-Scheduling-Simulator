#include <chrono>
#include <cstdio>
#include <vector>

#include "workload.h"

// Simple Computation
// Execute one task using a sequential dependency chain
unsigned long long ExecuteSimpleTaskCPU(
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

// CPU Executor
// Execute the shared workload on the CPU
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


    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

// Main
int main() {

    const int task_count = 1000;
    const int iterations = 100000;


    std::vector<SimpleTask> tasks(
        task_count
    );

    std::vector<unsigned long long> results(
        task_count
    );


    // Generate one deterministic shared workload.
    GenerateSimpleWorkload(
        tasks.data(),
        task_count
    );


    printf(
        "\n"
        "================ SIMPLE CPU TEST ================\n"
    );

    printf(
        "Tasks: %d\n",
        task_count
    );

    printf(
        "Iterations per task: %d\n",
        iterations
    );


    double cpu_time =
        RunSimpleCPU(
            tasks.data(),
            task_count,
            iterations,
            results.data()
        );


    printf(
        "\nCPU Time: %.4f ms\n",
        cpu_time
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