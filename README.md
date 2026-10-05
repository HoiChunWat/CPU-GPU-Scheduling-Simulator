# CPU-GPU Scheduling Simulator

A C/C++/CUDA scheduling and parallel-computing project that evolves from a single-CPU scheduling simulator into a pthread-based multicore benchmark and an empirical CPU-GPU offloading study.

The current implementation contains three major stages:

- **`scheduler_baseline.c`** — single-CPU scheduling simulation with FCFS, SJF, Priority, and Round Robin.
- **`scheduler_parallel.c`** — pthread-based parallel execution of the same scheduling policies with scalability measurements from 1 to 32 worker threads.
- **CUDA CPU-GPU offloading benchmarks** — simple-computation and matrix-multiplication workloads used to study GPU offloading crossover behavior and empirical CPU/GPU decision boundaries.

> **Project status:** CPU baseline, multicore pthread benchmarking, CUDA workload benchmarking, matrix offloading analysis, and 11-run repeatability testing are complete. The next step is to convert the empirical results into an actual CPU/GPU device-selection scheduler and validate it on unseen workloads.

---

## Table of Contents

- [Project Goals](#project-goals)
- [System Architecture](#system-architecture)
- [Hardware and Software Environment](#hardware-and-software-environment)
- [Phase 1 — Single-CPU Scheduling Simulator](#phase-1--single-cpu-scheduling-simulator)
- [Phase 2 — pthread Parallel Scheduler](#phase-2--pthread-parallel-scheduler)
- [Parallel Benchmark Results](#parallel-benchmark-results)
- [Performance Analysis](#performance-analysis)
- [Phase 3 — CPU-GPU Offloading Experiments](#phase-3--cpu-gpu-offloading-experiments)
- [CPU-GPU Benchmark Results](#cpu-gpu-benchmark-results)
- [Repeatability Analysis](#repeatability-analysis)
- [Design Decisions](#design-decisions)
- [Build and Run](#build-and-run)
- [Repository Structure](#repository-structure)
- [Scope and Current Limitations](#scope-and-current-limitations)
- [Next Phase — CPU-GPU Heterogeneous Scheduling](#next-phase--cpu-gpu-heterogeneous-scheduling)
- [Roadmap](#roadmap)
- [Notes on Reproducibility](#notes-on-reproducibility)
- [Author](#author)

---

## Project Goals

This project was built to study three related questions:

1. **How do classical CPU scheduling policies behave in a single-CPU simulation?**
2. **How does real CPU-bound work scale when the scheduling logic is executed by multiple pthread workers?**
3. **When does CUDA GPU offloading become faster than a single-threaded CPU baseline, and which workload features best predict that crossover?**

The next goal is to convert the measured CPU/GPU crossover behavior into a heterogeneous scheduler that can route work to either CPU or GPU execution based on workload characteristics and measured execution cost.

---

## System Architecture

The project is organized into three stages. All three stages now have working implementations; the remaining work is to turn the measured CPU/GPU crossover behavior into a validated device-selection scheduler.

```mermaid
flowchart LR

    subgraph P3["Phase 3 — CPU-GPU Offloading Experiments"]
        C1["Shared Workload Generator"] --> C2["CPU-GPU Benchmarking"]

        C2 --> C3["Simple Computation"]
        C2 --> C4["Matrix Multiplication"]

        C3 --> C5["CPU vs CUDA GPU"]
        C4 --> C6["CPU vs CUDA GPU"]

        C5 --> C7["Task-count / Iteration Analysis"]
        C6 --> C8["Work(MKN) / K / R Analysis"]

        C7 --> C9["Crossover Region"]
        C8 --> C9

        C9 --> C10["Future Device-selection Scheduler"]
    end


    subgraph P2["Phase 2 — Parallel CPU Benchmark"]
        B1["Random Task Generator"] --> B2["1,000-Task Workload"]

        B2 --> B3["scheduler_parallel.c"]
        B3 --> B4["pthread Worker Pool"]
        B4 --> B5{"Scheduling Policy"}

        B5 --> B6["FCFS"]
        B5 --> B7["SJF"]
        B5 --> B8["Priority"]
        B5 --> B9["Round Robin"]

        B6 --> B10["1 / 2 / 4 / 8 / 16 / 20 / 24 / 32 Threads"]
        B7 --> B10
        B8 --> B10
        B9 --> B10

        B10 --> B11["Runtime / Speedup / Parallel Efficiency"]
    end


    subgraph P1["Phase 1 — Baseline Scheduling Simulation"]
        A1["Random Task Generator"] --> A2["50-Task Workload"]

        A2 --> A3["scheduler_baseline.c"]
        A3 --> A4{"Scheduling Policy"}

        A4 --> A5["FCFS"]
        A4 --> A6["SJF"]
        A4 --> A7["Priority"]
        A4 --> A8["Round Robin"]

        A5 --> A9["Waiting / Turnaround / Response Time"]
        A6 --> A9
        A7 --> A9
        A8 --> A9
    end
```

## Repeatability Validation
flowchart LR
```mermaid
flowchart TD

    A1["matrix_benchmark.cu"] --> A2["1 Complete Benchmark Run"]
    A2 --> A3["Extract Work / K Boundaries"]

    A3 --> B1["repeatability_test.py"]
    B1 --> B2["Repeat 11 Times"]
    B2 --> B3["Collect 11 Boundary Sets"]

    B3 --> C1["Median Work Boundary"]
    B3 --> C2["Median K Boundary"]

    C1 --> D1["Stable Overlap Region"]
    C2 --> D1

    D1 --> E1["Validated CPU-GPU Decision Rule"]
```
end
    

### Why the two CPU stages use different workload sizes

| Stage | Workload | Primary purpose |
|---|---:|---|
| Baseline simulator | 50 tasks | Keep task-level scheduling behavior and latency metrics readable |
| Parallel benchmark | 1,000 tasks | Provide enough CPU-bound work for meaningful scalability measurements |

The baseline and parallel stages intentionally use different workload sizes because they serve different evaluation goals:

- **Baseline stage:** uses **50 tasks** so individual task parameters and scheduling metrics remain readable.
- **Parallel stage:** uses **1,000 tasks** to provide enough CPU-bound work for meaningful thread-scaling measurements.

---

## Hardware and Software Environment

The benchmark shown below was executed on the following development system.

| Component | Specification |
|---|---|
| Laptop | ASUS TUF Gaming F16 (FX607JU) |
| CPU | Intel Core i7-13650HX |
| Logical CPUs visible to WSL | 20 |
| System Memory | 16 GB RAM |
| Discrete GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| Dedicated GPU Memory | ~6 GB VRAM |
| Integrated GPU | Intel UHD Graphics |
| Host OS | Windows 11 Home 64-bit |
| Linux Environment | WSL2, Ubuntu 26.04 LTS |
| Languages | C, C++, CUDA C++ |
| Parallel Runtime | POSIX Threads (`pthread`) |
| Timing API | `clock_gettime(CLOCK_MONOTONIC)` |
| CUDA Toolkit | CUDA 13.4 (`nvcc` 13.4.92), target `sm_89` |

The CPU scheduling results are measured on the CPU, while the offloading experiments use the NVIDIA GeForce RTX 4050 Laptop GPU. CPU/GPU comparisons in Phase 3 use a **single-threaded CPU baseline** unless stated otherwise.

---

## Phase 1 — Single-CPU Scheduling Simulator

### Implemented Scheduling Policies

#### First-Come, First-Served (FCFS)

Tasks are sorted by arrival time and executed in arrival order. FCFS is simple and deterministic, but long tasks can delay shorter tasks that arrive later.

#### Shortest Job First (SJF)

Among tasks that have already arrived, the scheduler selects the unfinished task with the smallest burst time. This generally reduces average waiting and turnaround time for the generated workload.

#### Priority Scheduling

Among available tasks, the scheduler selects the task with the highest priority. In this implementation, a **smaller numeric value represents a higher priority**.

#### Round Robin (RR)

Tasks are executed cyclically with a time quantum of **2 burst units**. This allows tasks to receive CPU service earlier, improving responsiveness at the cost of more frequent switching between tasks.

---

### Task Model

Each task contains:

```c
typedef struct {
    char id[6];
    int arrival_time;
    int burst_time;
    int remaining_time;
    int priority;
    int waiting_time;
    int turnaround_time;
    int completion_time;
    int response_time;
} Task;
```

The baseline program generates random task IDs and random scheduling parameters, then gives each scheduling algorithm an independent copy of the same workload so that the comparison is performed on identical input data.

---

### Baseline Metrics

The simulator computes:

- **Waiting Time** — total time spent waiting before execution.
- **Turnaround Time** — completion time minus arrival time.
- **Response Time** — time from arrival until the task first receives CPU service.
- **Completion Time** — simulated time at which the task finishes.

For the sample 50-task run used in this README:

| Algorithm | Avg. Waiting Time | Avg. Turnaround Time | Avg. Response Time |
|---|---:|---:|---:|
| FCFS | 81.96 | 87.40 | 81.96 |
| SJF | 54.26 | 59.70 | 54.26 |
| Priority | 92.58 | 98.02 | 92.58 |
| Round Robin | 128.58 | 134.02 | **4.94** |

#### Baseline Observations

- **SJF produced the lowest average waiting and turnaround time** in this particular random workload because shorter tasks are selected earlier.
- **Round Robin produced the lowest response time by a large margin** because each ready task receives CPU time quickly instead of waiting for all earlier tasks to finish.
- The lower response time of Round Robin comes with a tradeoff: tasks may need several time slices before completion, which increases total waiting and turnaround time.
- FCFS response time equals its waiting time because each task runs to completion the first time it reaches the CPU.
- These values depend on the randomly generated workload and should be interpreted as one sample run rather than universal properties of the algorithms.

---

## Phase 2 — pthread Parallel Scheduler

The second stage replaces the purely simulated execution model with actual CPU-bound work executed by multiple pthread workers.

### Parallel Design

Each worker repeatedly:

1. Acquires the shared scheduling mutex.
2. Selects or dequeues one task according to the active scheduling mode.
3. Releases the mutex.
4. Executes synthetic CPU-bound work **outside the critical section**.
5. For Round Robin, reacquires the mutex to update `remaining_time`, requeue unfinished tasks, or increment the completion count.

Keeping the CPU-bound execution outside the mutex is essential. If the mutex were held while a task was executing, the program would effectively serialize the workload and lose multicore parallelism.

#### Shared-State Synchronization

A single `pthread_mutex_t` protects shared scheduler state such as:

- the FCFS dispatch index,
- SJF/Priority task-selection state,
- the Round Robin queue,
- Round Robin completion counters,
- task `remaining_time` updates.

Round Robin uses a circular queue of **task indices** rather than copying complete `Task` objects into the queue. The authoritative task state remains in the shared task array.

---

### Synthetic CPU Workload

Each burst unit performs a CPU-bound loop:

```c
for (long long j = 0; j < 5000000; j++) {
    result += j % 7;
}
```

This converts each simulated burst into measurable real CPU work, making it possible to study multicore scaling with pthread workers.

The parallel benchmark uses **1,000 generated tasks** and evaluates:

```text
1, 2, 4, 8, 16, 20, 24, and 32 worker threads
```

---

## Parallel Benchmark Results

### Runtime

| Threads | FCFS (s) | SJF (s) | Priority (s) | Round Robin (s) |
|---:|---:|---:|---:|---:|
| 1 | 22.7036 | 22.2104 | 22.1941 | 22.1631 |
| 2 | 12.0063 | 12.2420 | 12.2761 | 12.0304 |
| 4 | 6.5670 | 6.6474 | 6.6160 | 6.4014 |
| 8 | 4.4014 | 4.2248 | 4.2865 | 4.1756 |
| 16 | 2.8088 | 2.6964 | 2.6463 | 2.6487 |
| 20 | 2.4629 | 2.5443 | 2.3491 | 2.3006 |
| 24 | 2.4390 | 2.3437 | 2.3146 | 2.3081 |
| 32 | 2.3572 | 2.3462 | 2.4187 | 2.3065 |

![Runtime Scaling](assets/runtime_scaling.png)

---

### Speedup

Speedup is calculated as:

```text
Speedup(N) = T1 / TN
```

where `T1` is the one-thread runtime and `TN` is the runtime using `N` worker threads.

| Threads | FCFS | SJF | Priority | Round Robin |
|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.00x | 1.00x | 1.00x |
| 2 | 1.89x | 1.81x | 1.81x | 1.84x |
| 4 | 3.46x | 3.34x | 3.35x | 3.46x |
| 8 | 5.16x | 5.26x | 5.18x | 5.31x |
| 16 | 8.08x | 8.24x | 8.39x | 8.37x |
| 20 | 9.22x | 8.73x | 9.45x | 9.63x |
| 24 | 9.31x | 9.48x | 9.59x | 9.60x |
| 32 | **9.63x** | 9.47x | 9.18x | **9.61x** |

![Parallel Speedup](assets/speedup_scaling.png)

The observed peak speedup is approximately **9.6x**.

---

### Parallel Efficiency

Parallel efficiency is calculated as:

```text
Efficiency(N) = Speedup(N) / N * 100%
```

| Threads | FCFS | SJF | Priority | Round Robin |
|---:|---:|---:|---:|---:|
| 1 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2 | 94.5% | 90.7% | 90.4% | 92.1% |
| 4 | 86.4% | 83.5% | 83.9% | 86.6% |
| 8 | 64.5% | 65.7% | 64.7% | 66.3% |
| 16 | 50.5% | 51.5% | 52.4% | 52.3% |
| 20 | 46.1% | 43.6% | 47.2% | 48.2% |
| 24 | 38.8% | 39.5% | 40.0% | 40.0% |
| 32 | 30.1% | 29.6% | 28.7% | 30.0% |

![Parallel Efficiency](assets/parallel_efficiency.png)

---

## Performance Analysis

### 1. Strong scaling from 1 to 16 threads

Runtime decreases substantially as the worker count increases from 1 to 16 threads.

For example, Round Robin improves from:

```text
1 thread:   22.1631 s
16 threads:  2.6487 s
```

This corresponds to an **8.37x speedup**.

The workload is CPU-bound and contains many independent tasks, so additional workers can execute substantial portions of the workload concurrently.

---

### 2. Saturation near the available logical CPU count

The system exposes **20 logical CPUs** to WSL.

The benchmark begins to flatten around 20 worker threads:

```text
Round Robin
20 threads: 2.3006 s
24 threads: 2.3081 s
32 threads: 2.3065 s
```

At this point, adding more software threads does not add more hardware execution resources.

The 20-, 24-, and 32-thread results should therefore be viewed as the same general **saturation region**, not as evidence that 32 threads provide substantially more compute capability.

---

### 3. Why can 24 or 32 threads occasionally appear slightly faster?

A worker count above the number of visible logical CPUs can occasionally produce a slightly lower measured runtime, but the differences are small.

Possible causes include:

- operating-system scheduling variation,
- dynamic CPU frequency and boost behavior,
- background activity in Windows and WSL,
- cache-state variation,
- differences in load distribution near the end of the workload,
- measurement noise.

For example, FCFS measured:

```text
20 threads: 2.4629 s
24 threads: 2.4390 s
32 threads: 2.3572 s
```

This does **not** mean that 32 threads provide 32-way hardware parallelism. The system still exposes 20 logical CPUs. The result is better interpreted as normal variation inside the saturated performance region.

A more rigorous performance study would repeat each configuration multiple times and report the mean and standard deviation.

---

### 4. Why does efficiency decrease as thread count increases?

Parallel efficiency falls from roughly 90% at low thread counts to about 30% at 32 threads.

This is expected because speedup does not grow linearly with worker count.

The main limiting factors are:

#### Hardware concurrency limit

Once the number of workers approaches the available logical CPUs, new threads no longer receive independent hardware execution resources.

#### Mutex contention

Workers must serialize briefly when accessing shared scheduler state. The protected critical section is intentionally small, but contention increases as the number of workers grows.

#### Thread scheduling and context switching

When the program uses more runnable threads than available logical CPUs, the OS must schedule and switch between them.

#### Cache and memory-system contention

More concurrent workers compete for shared cache capacity, memory bandwidth, and other processor resources.

#### Non-parallel overhead

Thread creation, synchronization, queue operations, scheduling decisions, and benchmark bookkeeping cannot be perfectly parallelized.

---

### 5. Scheduling policy vs. thread count

The four scheduling policies show very similar total runtimes at a given thread count.

This is expected for this benchmark because every policy ultimately performs approximately the same total amount of CPU-bound arithmetic. Changing the scheduling order affects which task runs first, but worker count and available hardware parallelism have a much larger impact on total wall-clock runtime.

The project therefore evaluates two different kinds of behavior:

- **Baseline simulator:** scheduling quality through waiting, turnaround, and response time.
- **Parallel implementation:** execution scalability through runtime, speedup, and efficiency.

---


## Phase 3 — CPU-GPU Offloading Experiments

The third stage adds CUDA execution and measures when GPU offloading becomes beneficial compared with a **single-threaded CPU baseline**.

### Shared Workload Infrastructure

The simple-computation programs share the same workload definition through:

```text
common/workload.h
common/workload.cpp
```

The current CUDA-related source files are:

```text
cpu/simple_cpu.cpp
gpu/simple_gpu.cu
simple_benchmark.cu
simple_iterations_benchmark.cu
matrix_benchmark.cu
repeatability_test.py
```

### GPU Timing Definitions

Three GPU timing measurements are reported:

- **GPU Kernel Time** — kernel execution only.
- **GPU H2D-to-D2H Time** — H2D transfer + kernel + D2H transfer.
- **GPU Full Time** — `cudaMalloc + H2D + Kernel + D2H + cudaFree`.

CUDA initialization is performed before timing.

For each workload:

```text
5 warm-up runs
10 measured runs
median timing
```

The benchmark executes the GPU workload phase first and the CPU workload phase second. Workload execution order is shuffled separately for CPU and GPU measurements.

---

## CPU-GPU Benchmark Results

### Simple Computation — Task-Count Scaling

For the task-count experiment, computation intensity is fixed at **100,000 iterations per task**.

| Tasks | CPU (ms) | GPU Kernel (ms) | H2D-D2H (ms) | GPU Full (ms) |
|---:|---:|---:|---:|---:|
| 100 | 12.7582 | 1.2140 | 1.2804 | 1.7118 |
| 250 | 31.8816 | 1.4264 | 1.4967 | 1.9245 |
| 500 | 64.6220 | 1.4310 | 1.4996 | 1.9304 |
| 1,000 | 128.8157 | 1.4351 | 1.5015 | 1.9652 |
| 2,000 | 256.6449 | 1.4362 | 1.5155 | 1.9640 |
| 5,000 | 635.3893 | 1.4679 | 1.5513 | 2.0119 |
| 10,000 | 1264.5668 | 2.0112 | 2.0939 | 2.5160 |

![Simple Workload Runtime Scaling](assets/simple_task_scaling.png)

At 10,000 tasks, GPU Full Time is approximately **502.6x faster** than the single-threaded CPU baseline for this synthetic workload.

### Simple Computation — Iteration Crossover

With task count fixed at **1,000**, increasing iterations per task reveals the offloading crossover:

- GPU Kernel Time becomes favorable at very low computation intensity.
- H2D-to-D2H Time crosses the CPU baseline at roughly **50–100 iterations per task**.
- GPU Full Time crosses the CPU baseline at roughly **300–350 iterations per task**.

The full GPU path is therefore not automatically beneficial for very small workloads because allocation and transfer overhead must first be amortized.

---

### Matrix Multiplication

The matrix workload computes:

```text
A(M×K) × B(K×N) = C(M×N)
```

The CPU implementation uses a conventional triple loop. The CUDA implementation uses a naive kernel with one GPU thread per output element and a `16×16` thread block.

The main workload features are:

```text
Out(MN)      = M × N
Work(MKN)    = M × K × N
Operations   ≈ 2 × M × K × N
Transfer     = 4 × (MK + KN + MN) bytes
R            = MKN / [2(MK + KN + MN)]
```

The ratio `R` is used as an approximate compute-to-host/device-transfer indicator for the current one-shot offload model; it is not intended as a complete roofline arithmetic-intensity model.

### Matrix Runtime Scaling

A matrix-size sweep with `K = 256` shows the CPU/GPU Full-Time crossover between the smaller and larger matrix cases.

| N | CPU (ms) | GPU Kernel (ms) | H2D-D2H (ms) | GPU Full (ms) |
|---:|---:|---:|---:|---:|
| 16 | 0.0255 | 0.0113 | 0.1529 | 0.5553 |
| 32 | 0.0986 | 0.0154 | 0.1080 | 0.4805 |
| 64 | 0.4899 | 0.0113 | 0.1267 | 0.5264 |
| 128 | 1.9969 | 0.0464 | 0.2170 | 0.6129 |
| 256 | 7.8952 | 0.0816 | 0.3428 | 0.8048 |
| 512 | 31.5605 | 0.2430 | 0.7583 | 1.2798 |

![Matrix Runtime Scaling](assets/matrix_runtime_scaling.png)

The observed data showed that **total matrix work `MKN` is a stronger first-order offloading feature than `R` alone**. Within the Work transition region, `K` provides useful additional discrimination.

---

## Repeatability Analysis

The matrix offloading benchmark was repeated as **11 complete runs**.

Each complete run already contains:

```text
5 warm-up runs per workload
10 measured runs per workload
median timing
200 matrix workloads
```

The final boundary summary uses the discrete median across the 11 complete runs.

### Final Median Boundaries

| Metric | Median Boundary |
|---|---:|
| Lowest GPU-winning Work | 1,310,720 |
| Highest CPU-winning Work | 2,359,296 |
| Lowest GPU-winning K | 96 |
| Highest CPU-winning K | 192 |

Therefore, the current empirical transition regions are:

```text
Median Work overlap: 1,310,720 ~ 2,359,296
Median K overlap:    96 ~ 192
```

Across the 11 runs:

```text
Lowest GPU Work range  : 983,040 ~ 1,310,720
Highest CPU Work range : 1,966,080 ~ 2,359,296
Lowest GPU K range     : 48 ~ 160
Highest CPU K range    : 96 ~ 320
```

![Work Boundary Repeatability](assets/work_boundary_repeatability.png)

The Work boundary is relatively stable across repeated measurements, while `K` is a secondary feature with greater run-to-run variability. These values are empirical results for the current hardware, single-threaded CPU baseline, workload set, and naive CUDA matrix kernel; they should not be interpreted as universal CPU/GPU thresholds.

---

## Design Decisions

### Why use pthreads?

POSIX Threads provide direct control over worker creation, synchronization, and shared-memory execution. This makes the project suitable for studying the mechanics of multicore CPU parallelism rather than relying on a higher-level task runtime.

### Why execute work outside the mutex?

The mutex protects only task selection and shared-state updates. CPU-bound execution occurs after releasing the lock so that multiple workers can perform useful computation concurrently.

### Why store task indices in the Round Robin queue?

The RR queue stores integer indices into `shared_tasks[]`.

This avoids duplicating entire task structures and keeps one authoritative copy of each task's `remaining_time`.

### Why use a synthetic CPU-bound loop?

The original scheduling simulator advances logical time but does not consume measurable CPU time. A synthetic arithmetic workload creates real execution cost while preserving `burst_time` as the amount of work associated with each task.

---

## Build and Run

### Baseline simulator

```bash
gcc scheduler_baseline.c -o scheduler_baseline
./scheduler_baseline
```

### Parallel pthread version

```bash
gcc scheduler_parallel.c -o scheduler_parallel -pthread
./scheduler_parallel
```


### Simple CPU workload

```bash
nvcc cpu/simple_cpu.cpp common/workload.cpp -Icommon -o simple_cpu
./simple_cpu
```

### Simple GPU workload

```bash
nvcc gpu/simple_gpu.cu common/workload.cpp -Icommon -o simple_gpu
./simple_gpu
```

### Simple CPU-GPU benchmark

```bash
nvcc simple_benchmark.cu common/workload.cpp -Icommon -o simple_benchmark
./simple_benchmark
```

### Iteration crossover benchmark

```bash
nvcc simple_iterations_benchmark.cu common/workload.cpp -Icommon -o simple_iterations_benchmark
./simple_iterations_benchmark
```

### Matrix offloading benchmark

```bash
mkdir -p build
nvcc -O2 -std=c++17 -arch=sm_89 matrix_benchmark.cu -o build/matrix_benchmark
./build/matrix_benchmark
```

### 11-run repeatability test

```bash
python3 repeatability_test.py
```

---

## Repository Structure

```text
CPU-GPU-Scheduling-Simulator/
|
|-- scheduler_baseline.c
|-- scheduler_parallel.c
|
|-- cpu/
|   `-- simple_cpu.cpp
|
|-- gpu/
|   `-- simple_gpu.cu
|
|-- common/
|   |-- workload.h
|   `-- workload.cpp
|
|-- simple_benchmark.cu
|-- simple_iterations_benchmark.cu
|-- matrix_benchmark.cu
|-- repeatability_test.py
|
|-- assets/
|   |-- runtime_scaling.png
|   |-- speedup_scaling.png
|   |-- parallel_efficiency.png
|   |-- simple_task_scaling.png
|   |-- matrix_runtime_scaling.png
|   `-- work_boundary_repeatability.png
|
|-- README.md
`-- .gitignore
```

---

## Scope and Current Limitations

- The pthread scaling results shown above are from a single measured run per thread configuration; the newer CPU-GPU matrix study uses repeated median-based measurements as described in the repeatability section.
- The original pthread workload generator uses a time-based seed, so its workload characteristics can vary between program runs. The CPU-GPU matrix workload selection uses fixed seeds.
- The parallel SJF and Priority implementations select globally from unassigned tasks and are designed primarily for parallel-dispatch benchmarking rather than exact arrival-aware OS scheduling semantics.
- The parallel Round Robin benchmark initializes the workload into its ready queue before execution; it is not intended to model every detail of a production operating-system scheduler.
- CPU affinity is not currently pinned, so Linux/WSL may migrate pthreads between logical CPUs.
- CPU/GPU crossover thresholds are empirical and specific to the current RTX 4050 Laptop GPU, single-threaded CPU baseline, naive CUDA kernels, and sampled workloads.
- The matrix workload set was originally sampled across the `R` range, so it was not specifically designed to densely sample the final Work-overlap region.
- The final CPU/GPU device-selection scheduler and held-out validation are not yet implemented.

---

## Next Phase — CPU-GPU Heterogeneous Scheduling

The benchmark and repeatability stages are now complete. The next stage is to convert the measured crossover behavior into an actual device-selection policy.

The current empirical hierarchy is:

```text
Work(MKN) first
      |
      +-- outside transition region -> CPU or GPU
      |
      `-- inside transition region  -> inspect K
```

The next implementation steps are:

- aggregate the 11 × 200 benchmark measurements by exact workload,
- identify stable CPU, stable GPU, and boundary workloads,
- implement `SelectDevice(M, K, N)`,
- generate unseen validation workloads,
- compare the scheduler against **Always CPU**, **Always GPU**, and an **Oracle** that selects the measured faster device,
- report device-selection accuracy and scheduler runtime relative to the Oracle.

A key research question for the next phase is:

> **Can an empirical Work(MKN)-first, K-second decision model select the faster processor closely enough to approach an Oracle CPU/GPU offloading policy?**


---

## Roadmap

- [x] Single-CPU scheduling simulator
- [x] FCFS, SJF, Priority, and Round Robin
- [x] Waiting / turnaround / response-time analysis
- [x] pthread worker pool
- [x] Parallel FCFS / SJF / Priority / Round Robin
- [x] 1–32 thread scalability benchmark
- [x] Runtime / speedup / parallel-efficiency analysis
- [x] CUDA simple-computation workload
- [x] CPU/GPU task-count scaling benchmark
- [x] Computation-intensity / iteration crossover benchmark
- [x] CPU and CUDA matrix multiplication
- [x] Kernel / H2D-to-D2H / Full GPU timing
- [x] Work(MKN) and K offloading analysis
- [x] 11-run repeatability experiment
- [ ] Per-workload 11-run stability analysis
- [ ] CPU/GPU `SelectDevice(M, K, N)` scheduler
- [ ] Held-out unseen-workload validation
- [ ] Always-CPU / Always-GPU / Oracle comparison
- [ ] CPU affinity experiments


---

## Notes on Reproducibility

The original pthread workload generator uses a time-based random seed, so exact CPU scheduling results can vary across runs. The newer CPU-GPU matrix experiments use fixed workload and execution-order seeds so the selected workload set and benchmark ordering are reproducible.

For reproducible experiments, the random seed can be replaced with a fixed value such as:

```c
srand(42);
```

For the CPU-GPU matrix study, each workload uses 5 warm-up runs and 10 measured runs with median timing. The complete 200-workload benchmark is then repeated 11 times for boundary repeatability analysis.

---

## Author

**Hoi Chun Wat**  
M.S. Electrical and Computer Engineering  
University of Florida
