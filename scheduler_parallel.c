#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <pthread.h>

typedef struct {             // Task structure
    char id[6];              // Task ID
    int arrival_time;        // Arrival time
    int burst_time;          // CPU burst time
    int remaining_time;      // Remaining execution time
    int priority;            // Priority
} Task;

// ==================== shared variables and scheduling state ====================
Task *shared_tasks;              // Pointer to the shared Task array
int shared_task_count;           // Number of tasks in the shared array
int next_task = 0;               // Number of dispatched tasks / next FCFS index
int *task_taken;                 // Used by SJF/Priority: 0 = available, 1 = already assigned
int scheduling_mode = 0;         // 0=FCFS, 1=SJF, 2=Priority, 3=Round Robin
pthread_mutex_t task_mutex;      // Protects shared scheduling state

int *rr_queue;          // Stores task indices; actual task state remains in shared_tasks[]
int rr_front = 0;       // Queue front
int rr_rear = 0;        // Queue rear
int rr_count = 0;       // Number of tasks currently in the queue
int rr_quantum = 2;     // Execute at most 2 burst units per quantum
int rr_completed_count = 0;   // Number of completed RR tasks

// ==================== function declarations ====================

// pthread-related declarations
void *Worker(void *arg);    // Worker function executed by each pthread
double RunParallelBenchmark(int worker_count, int mode);    // Creates the requested number of threads and selects a scheduling mode

void ExecuteTask(int burst_time);   // Generates actual CPU-bound work

// Workload generation declarations
int IsDuplicateID(Task tasks[], int current_index, char id[]);  // Check whether a random Task ID is duplicated
void GenerateRandomWorkload(Task tasks[], int task_count);  // Generate a large synthetic workload
void SortTasksByArrival(Task tasks[], int task_count);  // Sort workload by arrival time

// parallel scheduling selectors
int SelectSJFTask(Task tasks[], int task_count, int taken[]);       // SJF selection logic
int SelectPriorityTask(Task tasks[], int task_count, int taken[]);  // Priority selection logic

// Add a task index to the rear of the RR queue
void EnqueueRR(int task_index);
// Remove a task index from the front of the RR queue
int DequeueRR();

// Output function declaration
void PrintThreadBenchmarkSummary(
    double time_1,
    double time_2,
    double time_4,
    double time_8,         // Display runtime, speedup, and parallel efficiency
    double time_16,
    double time_20,
    double time_24,
    double time_32
);

int main() {
    srand(time(NULL));  // Seed rand() with the current time

    // Generate 1000 random tasks
    int random_count = 1000;
    Task random_tasks[random_count];    // Allocate stack storage for the task array

    GenerateRandomWorkload(random_tasks, random_count);
    SortTasksByArrival(random_tasks, random_count);

    // Build the shared workload
    shared_tasks = random_tasks;    // Point to the same task array
    shared_task_count = random_count;
    next_task = 0;                  // Start from task index 0

    task_taken = malloc(shared_task_count * sizeof(int));   // Allocate memory for shared_task_count integers
    
    rr_queue = malloc(shared_task_count * sizeof(int));   // Allocate memory for shared_task_count integers

    if (task_taken == NULL) {
        printf("Failed to allocate memory for task_taken.\n");
        return 1;                                           // Return 1 and terminate the program
    }                                                       // Allocation failed

    if (rr_queue == NULL) {
        printf("Failed to allocate memory for rr_queue.\n");
        free(task_taken);
        return 1;                                           // Allocation failed
    }

    pthread_mutex_init(&task_mutex, NULL);    // Initialize task_mutex with default attributes

    // ==================== Parallel FCFS ====================
    printf("\n========== PARALLEL FCFS ==========\n");

    double fcfs_time_1 = RunParallelBenchmark(1, 0);
    double fcfs_time_2 = RunParallelBenchmark(2, 0);
    double fcfs_time_4 = RunParallelBenchmark(4, 0);
    double fcfs_time_8 = RunParallelBenchmark(8, 0);
    double fcfs_time_16 = RunParallelBenchmark(16, 0);
    double fcfs_time_20 = RunParallelBenchmark(20, 0);
    double fcfs_time_24 = RunParallelBenchmark(24, 0);
    double fcfs_time_32 = RunParallelBenchmark(32, 0);

    printf("\n========== FCFS THREAD BENCHMARK SUMMARY ==========\n");

    PrintThreadBenchmarkSummary(
        fcfs_time_1,
        fcfs_time_2,
        fcfs_time_4,
        fcfs_time_8,
        fcfs_time_16,
        fcfs_time_20,
        fcfs_time_24,
        fcfs_time_32
    );

    // ==================== Parallel SJF ====================
    printf("\n========== PARALLEL SJF ==========\n");

    double sjf_time_1 = RunParallelBenchmark(1, 1);
    double sjf_time_2 = RunParallelBenchmark(2, 1);
    double sjf_time_4 = RunParallelBenchmark(4, 1);
    double sjf_time_8 = RunParallelBenchmark(8, 1);
    double sjf_time_16 = RunParallelBenchmark(16, 1);
    double sjf_time_20 = RunParallelBenchmark(20, 1);
    double sjf_time_24 = RunParallelBenchmark(24, 1);
    double sjf_time_32 = RunParallelBenchmark(32, 1);

    printf("\n========== SJF THREAD BENCHMARK SUMMARY ==========\n");

    PrintThreadBenchmarkSummary(
        sjf_time_1,
        sjf_time_2,
        sjf_time_4,
        sjf_time_8,
        sjf_time_16,
        sjf_time_20,
        sjf_time_24,
        sjf_time_32
    );

    // ==================== Parallel Priority ====================
    printf("\n========== PARALLEL PRIORITY ==========\n");

    double priority_time_1 = RunParallelBenchmark(1, 2);
    double priority_time_2 = RunParallelBenchmark(2, 2);
    double priority_time_4 = RunParallelBenchmark(4, 2);
    double priority_time_8 = RunParallelBenchmark(8, 2);
    double priority_time_16 = RunParallelBenchmark(16, 2);
    double priority_time_20 = RunParallelBenchmark(20, 2);
    double priority_time_24 = RunParallelBenchmark(24, 2);
    double priority_time_32 = RunParallelBenchmark(32, 2);

    printf("\n========== PRIORITY THREAD BENCHMARK SUMMARY ==========\n");

    PrintThreadBenchmarkSummary(
        priority_time_1,
        priority_time_2,
        priority_time_4,
        priority_time_8,
        priority_time_16,
        priority_time_20,
        priority_time_24,
        priority_time_32
    );

    // ==================== Parallel Round Robin ====================
    printf("\n========== PARALLEL ROUND ROBIN ==========\n");

    double rr_time_1 = RunParallelBenchmark(1, 3);
    double rr_time_2 = RunParallelBenchmark(2, 3);
    double rr_time_4 = RunParallelBenchmark(4, 3);
    double rr_time_8 = RunParallelBenchmark(8, 3);
    double rr_time_16 = RunParallelBenchmark(16, 3);
    double rr_time_20 = RunParallelBenchmark(20, 3);
    double rr_time_24 = RunParallelBenchmark(24, 3);
    double rr_time_32 = RunParallelBenchmark(32, 3);

    printf("\n========== ROUND ROBIN THREAD BENCHMARK SUMMARY ==========\n");

    PrintThreadBenchmarkSummary(
        rr_time_1,
        rr_time_2,
        rr_time_4,
        rr_time_8,
        rr_time_16,
        rr_time_20,
        rr_time_24,
        rr_time_32
    );


    free(task_taken);
    free(rr_queue);
    pthread_mutex_destroy(&task_mutex);

    return 0;
}

// ==================== pthread worker ====================

void *Worker(void *arg) {
    int worker_id = *(int *)arg;

    while (1) {
        pthread_mutex_lock(&task_mutex);

        if (scheduling_mode == 3){                          
            if(rr_completed_count >= shared_task_count){ // Exit when all RR tasks are complete
                pthread_mutex_unlock(&task_mutex);
                break;
            }
        }
        else if (next_task >= shared_task_count) {       // Exit when all non-RR tasks have been dispatched
            pthread_mutex_unlock(&task_mutex);
            break;
        }

        int task_index = -1;
        int execution_time = 0;

        switch (scheduling_mode) {

            case 0:
                // FCFS: shared_tasks is already sorted by arrival_time
                task_index = next_task;
                next_task++;
                break;

            case 1:
                // SJF: choose the shortest task that has not been assigned
                task_index = SelectSJFTask(
                    shared_tasks,
                    shared_task_count,
                    task_taken
                );

                if (task_index != -1) {
                    task_taken[task_index] = 1;
                    next_task++;
                }
                break;

            case 2:
                task_index = SelectPriorityTask(
                    shared_tasks,
                    shared_task_count,
                    task_taken
                );

                if (task_index != -1) {
                    task_taken[task_index] = 1;
                    next_task++;
                }
                break;

            case 3:
                task_index = DequeueRR();

                if (task_index != -1) {
                    if (shared_tasks[task_index].remaining_time > rr_quantum) {
                        execution_time = rr_quantum;
                    }
                    else {
                        execution_time = shared_tasks[task_index].remaining_time;
                    }
                }
                break;

            default:
                break;
        }

        pthread_mutex_unlock(&task_mutex);

        // No valid task was selected
        if (task_index == -1) {
            if (scheduling_mode == 3) {
                continue;
            }
            else {
                break;
            }
        }

        if (scheduling_mode == 3) {
            ExecuteTask(execution_time);

            pthread_mutex_lock(&task_mutex);    // Protect RR shared-state updates with the mutex
            shared_tasks[task_index].remaining_time -= execution_time;
            if (shared_tasks[task_index].remaining_time > 0) {
                EnqueueRR(task_index);      // Requeue unfinished task
            }
            else {
                rr_completed_count++;       // Count the task as completed
            }

            pthread_mutex_unlock(&task_mutex);
        }
        else {
            Task task = shared_tasks[task_index];   // Copy the selected shared Task into the worker's local variable
            ExecuteTask(task.burst_time);
        }
    }

    return NULL;
}

// ==================== benchmark ====================

double RunParallelBenchmark(int worker_count, int mode) {
    scheduling_mode = mode;
    next_task = 0;

    // Reset task assignment state before each benchmark
    for (int i = 0; i < shared_task_count; i++) {
        task_taken[i] = 0;
    }
    // Initialize all tasks in the RR queue before the benchmark
    if (mode == 3) {
        rr_front = 0;
        rr_rear = 0;
        rr_count = 0;
        rr_completed_count = 0;

        for (int i = 0; i < shared_task_count; i++) {
            shared_tasks[i].remaining_time = shared_tasks[i].burst_time;
            EnqueueRR(i);
        }
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);     // Start timing

    pthread_t workers[worker_count];        // Create storage for the requested number of threads
    int worker_ids[worker_count];           // Create IDs for each worker thread

    for (int i = 0; i < worker_count; i++) {
        worker_ids[i] = i;

        pthread_create(                 // Create the pthread
            &workers[i],                // Location for the created thread handle
            NULL,                       
            Worker,                     // Function executed by the new thread
            &worker_ids[i]              // Argument passed to Worker
        );
    }

    for (int i = 0; i < worker_count; i++) {
        pthread_join(workers[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);   // Stop timing

    double elapsed_time =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf(
        "%d thread(s): %.4f seconds\n",
        worker_count,
        elapsed_time
    );

    return elapsed_time;
}

// ==================== workload ====================

int IsDuplicateID(Task tasks[], int current_index, char id[]) {
    for (int j = 0; j < current_index; j++) {
        if (strcmp(tasks[j].id, id) == 0) {
            return 1;
        }
    }

    return 0;
}

void GenerateRandomWorkload(Task tasks[], int task_count) {
    for (int i = 0; i < task_count; i++) {

        do {
            char random_letter = 'A' + rand() % 26;

            snprintf(
                tasks[i].id,
                sizeof(tasks[i].id),
                "%c%03d",
                random_letter,
                rand() % 1000
            );

        } while (IsDuplicateID(tasks, i, tasks[i].id));

        tasks[i].arrival_time = rand() % 100;
        tasks[i].burst_time = rand() % 10 + 1;
        tasks[i].remaining_time = tasks[i].burst_time;
        tasks[i].priority = rand() % 5 + 1;
    }
}

void SortTasksByArrival(Task tasks[], int task_count) {
    for (int i = 0; i < task_count - 1; i++) {
        for (int j = 0; j < task_count - 1 - i; j++) {

            if (tasks[j].arrival_time > tasks[j + 1].arrival_time) {
                Task temp = tasks[j];
                tasks[j] = tasks[j + 1];
                tasks[j + 1] = temp;
            }
        }
    }
}

void EnqueueRR(int task_index) {        // Add a task index to the rear of the queue
    rr_queue[rr_rear] = task_index;
    rr_rear = (rr_rear + 1) % shared_task_count;    // Advance rear by one position and wrap to 0 at the end
    rr_count++;
}

int DequeueRR() {
    if (rr_count == 0) {    // No task is available when the queue is empty
        return -1;
    }

    int task_index = rr_queue[rr_front];    // Read the task index at the queue front

    rr_front = (rr_front + 1) % shared_task_count;  // Advance front by one position
    rr_count--;

    return task_index;
}

// ==================== scheduling selectors ====================

int SelectSJFTask(Task tasks[], int task_count, int taken[]) {
    int selected = -1;

    for (int i = 0; i < task_count; i++) {

        if (taken[i] == 0) {

            if (selected == -1) {
                selected = i;
            }

            else if (
                tasks[i].burst_time <
                tasks[selected].burst_time
            ) {
                selected = i;
            }
        }
    }

    return selected;
}

int SelectPriorityTask(Task tasks[], int task_count, int taken[]) {
    int selected = -1;

    for (int i = 0; i < task_count; i++) {

        if (taken[i] == 0) {

            if (selected == -1) {
                selected = i;
            }

            else if (
                tasks[i].priority <
                tasks[selected].priority
            ) {
                selected = i;
            }
        }
    }

    return selected;
}

// ==================== real CPU workload ====================

void ExecuteTask(int burst_time) {
    volatile long long result = 0;

    for (int i = 0; i < burst_time; i++) {

        for (long long j = 0; j < 5000000; j++) {
            result += j % 7;
        }

        result = 0;
    }
}

// ==================== output ====================

void PrintThreadBenchmarkSummary(
    double time_1,
    double time_2,
    double time_4,
    double time_8,
    double time_16,
    double time_20,
    double time_24,
    double time_32
) {
    printf(
        "%-10s%12s%12s%22s\n",
        "Threads",
        "Runtime(s)",
        "Speedup",
        "Parallel Efficiency"
    );

    printf("--------------------------------------------------------\n");

    printf(
        "  %-3d        %-8.4f      %-6.2fx        %-6.1f%%\n",
        1,
        time_1,
        1.0,
        100.0
    );

    printf(
        "  %-3d        %-8.4f      %-6.2fx        %-6.1f%%\n",
        2,
        time_2,
        time_1 / time_2,
        (time_1 / time_2) / 2 * 100
    );

    printf(
        "  %-3d        %-8.4f      %-6.2fx        %-6.1f%%\n",
        4,
        time_4,
        time_1 / time_4,
        (time_1 / time_4) / 4 * 100
    );

    printf(
        "  %-3d        %-8.4f      %-6.2fx        %-6.1f%%\n",
        8,
        time_8,
        time_1 / time_8,
        (time_1 / time_8) / 8 * 100
    );

    printf(
        "  %-3d        %-8.4f      %-6.2fx        %-6.1f%%\n",
        16,
        time_16,
        time_1 / time_16,
        (time_1 / time_16) / 16 * 100
    );

    printf(
        "  %-3d        %-8.4f      %-6.2fx        %-6.1f%%\n",
        20,
        time_20,
        time_1 / time_20,
        (time_1 / time_20) / 20 * 100
    );

    printf(
        "  %-3d        %-8.4f      %-6.2fx        %-6.1f%%\n",
        24,
        time_24,
        time_1 / time_24,
        (time_1 / time_24) / 24 * 100
    );

    printf(
        "  %-3d        %-8.4f      %-6.2fx        %-6.1f%%\n",
        32,
        time_32,
        time_1 / time_32,
        (time_1 / time_32) /32 * 100
    );

    printf("=========================================================\n");
}
