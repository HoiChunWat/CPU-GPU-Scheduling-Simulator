#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

typedef struct {
    char id[6];              // Task ID
    int arrival_time;     // Arrival time
    int burst_time;       // Required CPU burst time
    int remaining_time;   // Remaining execution time
    int priority;         // Priority value
    int waiting_time;      // Waiting time
    int turnaround_time;   // Total turnaround time
    int completion_time;   // Completion time
    int response_time;     // Time from arrival until first CPU execution
} Task;

// Check whether a randomly generated task ID already exists
int IsDuplicateID(Task tasks[], int current_index, char id[]);
// Generate a random workload
void GenerateRandomWorkload(Task tasks[], int task_count);

// Sort tasks by arrival time
void SortTasksByArrival(Task tasks[], int task_count);

// Scheduling algorithm declarations
void FCFS(Task fcfs_tasks[], int task_count);
void SJF(Task sjf_tasks[], int task_count);
void Priority(Task priority_tasks[], int task_count);
void RoundRobin(Task rr_tasks[], int task_count);

void RunWorkload(Task tasks[], int task_count); // Run all scheduling algorithms

// Result output function
void PrintResults(Task tasks[], int task_count); // Print scheduling results

// Print benchmark summary
void PrintBenchmarkSummary(
    Task fcfs_tasks[],
    Task sjf_tasks[],
    Task priority_tasks[],
    Task rr_tasks[],
    int task_count
);

int main() {
    srand(time(NULL)); // Seed the random generator with the current time

    /*Task tasks1[] = {
        {"C001", 2, 1, 1, 3, 0, 0, 0, -1},
        {"B001", 1, 2, 2, 1, 0, 0, 0, -1},
        {"A001", 0, 4, 4, 2, 0, 0, 0, -1}
    };
    int task_count1 = sizeof(tasks1) / sizeof(tasks1[0]);
    SortTasksByArrival(tasks1, task_count1);
    printf("\n========== WORKLOAD 1 ==========\n");
    RunWorkload(tasks1, task_count1); // Run the workload*/
    
    /*Task tasks2[] = {
        {"A001", 0, 8, 8, 3, 0, 0, 0, -1},
        {"D001", 0, 1, 1, 4, 0, 0, 0, -1},
        {"B001", 0, 2, 2, 1, 0, 0, 0, -1},
        {"C001", 0, 4, 4, 2, 0, 0, 0, -1}
    };
    int task_count2 = sizeof(tasks2) / sizeof(tasks2[0]);
    SortTasksByArrival(tasks2, task_count2);
    printf("\n========== WORKLOAD 2 ==========\n");
    RunWorkload(tasks2, task_count2); // Run the workload*/

    // Generate a random task set
    int random_count = 50;
    Task random_tasks[random_count];
    GenerateRandomWorkload(random_tasks, random_count);
    SortTasksByArrival(random_tasks, random_count);

    RunWorkload(random_tasks, random_count); // Run the workload
    
}

// Check whether a randomly generated task ID already exists
int IsDuplicateID(Task tasks[], int current_index, char id[]) {

    for (int j = 0; j < current_index; j++) {

        if (strcmp(tasks[j].id, id) == 0) {  // Compare the generated ID with existing task IDs
            return 1;                        // Duplicate found
        }
    }

    return 0;                                // No duplicate found
}
// Generate a random workload
void GenerateRandomWorkload(Task tasks[], int task_count){
    for (int i = 0; i < task_count; i++){
        
        // Generate a random task ID
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

        tasks[i].waiting_time = 0;
        tasks[i].turnaround_time = 0;
        tasks[i].completion_time = 0;
        tasks[i].response_time = -1;
    }
}

void SortTasksByArrival(Task tasks[], int task_count){
    for (int i = 0; i < task_count - 1; i++){
        for (int j = 0; j < task_count - 1 - i; j++){ // Larger arrival times are gradually moved to the end
            if (tasks[j].arrival_time > tasks[j+1].arrival_time){
                Task temp = tasks[j];
                tasks[j] = tasks[j+1];
                tasks[j+1] = temp;
            }
        }
    }
}


void RunWorkload(Task tasks[], int task_count) { // Run all scheduling algorithms

    Task fcfs_tasks[task_count];
    Task sjf_tasks[task_count];
    Task priority_tasks[task_count];
    Task rr_tasks[task_count];

    // Give each scheduler an independent copy of the same workload
    for (int i = 0; i < task_count; i++) {
        fcfs_tasks[i] = tasks[i];
        sjf_tasks[i] = tasks[i];
        priority_tasks[i] = tasks[i];
        rr_tasks[i] = tasks[i];
    }

    printf("\nInput Tasks:\n");

    for (int i = 0; i < task_count; i++) {
        printf(
            "Task %s: arrival=%d, burst=%d, priority=%d\n",
            tasks[i].id,
            tasks[i].arrival_time,
            tasks[i].burst_time,
            tasks[i].priority
        );
    }

    printf("\n========================================\n");
    // Run each scheduling algorithm
    FCFS(fcfs_tasks, task_count);
    SJF(sjf_tasks, task_count);
    Priority(priority_tasks, task_count);
    RoundRobin(rr_tasks, task_count);

    // Print the final comparison summary
    PrintBenchmarkSummary(
        fcfs_tasks,
        sjf_tasks,
        priority_tasks,
        rr_tasks,
        task_count
    );
    
}



//function1
void FCFS(Task fcfs_tasks[], int task_count){ 
    int current_time = 0; // Initial simulated time

    printf("\n1.FCFS Scheduling:\n");       // Run FCFS scheduling

    for (int i = 0; i < task_count; i++) {

        
        // Keep the CPU idle until the next task arrives
        while (current_time < fcfs_tasks[i].arrival_time) {
        /*printf("Time %d: CPU Idle\n", current_time);*/
        current_time++;
        }
        
        fcfs_tasks[i].waiting_time = current_time - fcfs_tasks[i].arrival_time; // Waiting time = start time - arrival time

        /*printf(
            "Time %d - %d: Running Task %s\n",
            current_time,
            current_time + fcfs_tasks[i].burst_time,
            fcfs_tasks[i].id
        );*/

        current_time += fcfs_tasks[i].burst_time;   // Advance simulated time after the task completes
        fcfs_tasks[i].turnaround_time = current_time - fcfs_tasks[i].arrival_time; // Turnaround time = completion time - arrival time
        fcfs_tasks[i].completion_time = current_time;
        fcfs_tasks[i].response_time = current_time - fcfs_tasks[i].burst_time - fcfs_tasks[i].arrival_time;
    }

    PrintResults(fcfs_tasks, task_count);
}

//function2
void SJF(Task sjf_tasks[], int task_count){
    int current_time = 0; // Initial simulated time
    int completed_count = 0;   // Number of completed tasks
    int completed[task_count];  // Tracks whether each task has completed

    for (int i = 0; i < task_count; i++) { // Initialize all tasks as not completed
        completed[i] = 0;
    }

    printf("\n2.SJF Scheduling:\n");

    while (completed_count < task_count) {

        int selected = -1;   // No task selected yet

        // Select the shortest task among those that have already arrived
        for (int i = 0; i < task_count; i++) {

            if (
                completed[i] == 0 &&
                sjf_tasks[i].arrival_time <= current_time
            ) {

                if (
                    selected == -1 ||
                    sjf_tasks[i].burst_time <
                    sjf_tasks[selected].burst_time
                ) {
                    selected = i;
                }
            }
        }

        // No task is currently ready to run
        if (selected == -1) {
            /*printf("Time %d: CPU Idle\n", current_time);*/
            current_time++;
            continue;
        }
        else{
            sjf_tasks[selected].waiting_time =
            current_time - sjf_tasks[selected].arrival_time;

            /*printf(
                "Time %d - %d: Running Task %s\n",
                current_time,
                current_time + sjf_tasks[selected].burst_time,
                sjf_tasks[selected].id
            );*/

            current_time += sjf_tasks[selected].burst_time;

            sjf_tasks[selected].turnaround_time =
                current_time - sjf_tasks[selected].arrival_time;
            sjf_tasks[selected].completion_time = current_time;
            sjf_tasks[selected].response_time = current_time - sjf_tasks[selected].burst_time - sjf_tasks[selected].arrival_time;

            completed[selected] = 1;
            completed_count++;
        }
    }
    PrintResults(sjf_tasks, task_count);

}

//function3
void Priority(Task priority_tasks[], int task_count){
    int current_time = 0; // Initial simulated time
    int completed_count = 0;   // Number of completed tasks
    int completed[task_count];  // Tracks whether each task has completed

    for (int i = 0; i < task_count; i++) { // Initialize all tasks as not completed
        completed[i] = 0;
    }

    printf("\n3.Priority Scheduling:\n");

    while (completed_count < task_count) {

        int selected = -1;   // No task selected yet

        // Select the highest-priority task among those that have already arrived
        for (int i = 0; i < task_count; i++) {

            if (
                completed[i] == 0 &&
                priority_tasks[i].arrival_time <= current_time
            ) {

                if (
                    selected == -1 ||
                    priority_tasks[i].priority <
                    priority_tasks[selected].priority
                ) {
                    selected = i;
                }
            }
        }

        // No task is currently ready to run
        if (selected == -1) {
            /*printf("Time %d: CPU Idle\n", current_time);*/
            current_time++;
            continue;
        }
        else{
            priority_tasks[selected].waiting_time =
            current_time - priority_tasks[selected].arrival_time;

            /*printf(
                "Time %d - %d: Running Task %s\n",
                current_time,
                current_time + priority_tasks[selected].burst_time,
                priority_tasks[selected].id
            );*/

            current_time += priority_tasks[selected].burst_time;

            priority_tasks[selected].turnaround_time =
                current_time - priority_tasks[selected].arrival_time;
            priority_tasks[selected].completion_time = current_time;
            priority_tasks[selected].response_time = current_time - priority_tasks[selected].burst_time - priority_tasks[selected].arrival_time;


            completed[selected] = 1;
            completed_count++;
        }
    }
    PrintResults(priority_tasks, task_count);

}

//function4
void RoundRobin(Task rr_tasks[], int task_count) {

    int quantum = 2;          // Each task may run for at most 2 time units per turn
    int current_time = 0;     // Current simulated time
    int completed_count = 0;  // Number of completed tasks

    printf("\n4.Round Robin Scheduling:\n");

    while (completed_count < task_count) {

        for (int i = 0; i < task_count; i++) {

            // The task has arrived and still has remaining work
            if (
                rr_tasks[i].arrival_time <= current_time &&
                rr_tasks[i].remaining_time > 0
            ) {
                if ( rr_tasks[i].response_time < 0){
                    rr_tasks[i].response_time = current_time - rr_tasks[i].arrival_time;
                }
                // If the remaining time fits within one quantum, finish the task
                if (rr_tasks[i].remaining_time <= quantum) {

                    int execution_time = rr_tasks[i].remaining_time;

                    /*printf(
                        "Time %d - %d: Running Task %s\n",
                        current_time,
                        current_time + execution_time,
                        rr_tasks[i].id
                    );*/

                    current_time += execution_time;
                    rr_tasks[i].remaining_time = 0;

                    // Task completed
                    rr_tasks[i].turnaround_time =
                        current_time - rr_tasks[i].arrival_time;
                    rr_tasks[i].completion_time = current_time;

                    completed_count++;
                }

                // Otherwise, execute only one quantum
                else {

                    /*printf(
                        "Time %d - %d: Running Task %s\n",
                        current_time,
                        current_time + quantum,
                        rr_tasks[i].id
                    );*/

                    rr_tasks[i].remaining_time -= quantum;
                    current_time += quantum;
                }
            }

            // The task has not arrived yet
            else if (
                rr_tasks[i].remaining_time > 0 &&
                rr_tasks[i].arrival_time > current_time
            ) {

                /*printf("Time %d: CPU Idle\n", current_time);*/

                current_time++;

                // Recheck the same task on the next iteration
                i--;
            }
        }
    }

    // For Round Robin, waiting time can be derived as:
    // waiting = turnaround - burst
    for (int i = 0; i < task_count; i++) {
        rr_tasks[i].waiting_time =
            rr_tasks[i].turnaround_time - rr_tasks[i].burst_time;
    }

    PrintResults(rr_tasks, task_count);
}

// Print single-CPU scheduling results
void PrintResults(Task tasks[], int task_count) {

    /*printf("\nWaiting Time:\n");

    for (int i = 0; i < task_count; i++) {
        printf(
            "Task %s: %d\n",
            tasks[i].id,
            tasks[i].waiting_time
        );
    }

    printf("\nTurnaround Time:\n");

    for (int i = 0; i < task_count; i++) {
        printf(
            "Task %s: %d\n",
            tasks[i].id,
            tasks[i].turnaround_time
        );
    }

    printf("\nCompletion Time:\n");

    for (int i = 0; i < task_count; i++) {
        printf(
            "Task %s: %d\n",
            tasks[i].id,
            tasks[i].completion_time
        );
    }

    printf("\nResponse Time:\n");

    for (int i = 0; i < task_count; i++) {
        printf(
            "Task %s: %d\n",
            tasks[i].id,
            tasks[i].response_time
        );
    }*/

    int total_waiting_time = 0;
    int total_turnaround_time = 0;
    int total_response_time = 0;

    for (int i = 0; i < task_count; i++) {
        total_waiting_time += tasks[i].waiting_time;
        total_turnaround_time += tasks[i].turnaround_time;
        total_response_time += tasks[i].response_time;
    }

    double average_waiting_time =
        total_waiting_time / (double)task_count;

    double average_turnaround_time =
        total_turnaround_time / (double)task_count;

    double average_response_time =
        total_response_time / (double)task_count;

    printf("\nAverage Waiting Time: %.2f\n", average_waiting_time);
    printf("Average Turnaround Time: %.2f\n", average_turnaround_time);
    printf("Average Response Time: %.2f\n", average_response_time);

    printf("\n______________________________\n");
}

// Summarize and compare single-CPU scheduling metrics
void PrintBenchmarkSummary(
    Task fcfs_tasks[],
    Task sjf_tasks[],
    Task priority_tasks[],
    Task rr_tasks[],
    int task_count
) {

    int fcfs_wait = 0, fcfs_turn = 0, fcfs_response = 0;
    int sjf_wait = 0, sjf_turn = 0, sjf_response = 0;
    int priority_wait = 0, priority_turn = 0, priority_response = 0;
    int rr_wait = 0, rr_turn = 0, rr_response = 0;

    for (int i = 0; i < task_count; i++) {

        fcfs_wait += fcfs_tasks[i].waiting_time;
        fcfs_turn += fcfs_tasks[i].turnaround_time;
        fcfs_response += fcfs_tasks[i].response_time;

        sjf_wait += sjf_tasks[i].waiting_time;
        sjf_turn += sjf_tasks[i].turnaround_time;
        sjf_response += sjf_tasks[i].response_time;

        priority_wait += priority_tasks[i].waiting_time;
        priority_turn += priority_tasks[i].turnaround_time;
        priority_response += priority_tasks[i].response_time;

        rr_wait += rr_tasks[i].waiting_time;
        rr_turn += rr_tasks[i].turnaround_time;
        rr_response += rr_tasks[i].response_time;
    }

    printf("\n========== BENCHMARK SUMMARY ==========\n");

    printf(
        "%-15s %-12s %-15s %-15s\n",
        "Algorithm",
        "Avg Wait",
        "Avg Turn",
        "Avg Response"
    );

    printf("-----------------------------------------------------------\n");

    printf(
        "%-15s %-12.2f %-15.2f %-15.2f\n",
        "FCFS",
        fcfs_wait / (double)task_count,
        fcfs_turn / (double)task_count,
        fcfs_response / (double)task_count
    );

    printf(
        "%-15s %-12.2f %-15.2f %-15.2f\n",
        "SJF",
        sjf_wait / (double)task_count,
        sjf_turn / (double)task_count,
        sjf_response / (double)task_count
    );

    printf(
        "%-15s %-12.2f %-15.2f %-15.2f\n",
        "Priority",
        priority_wait / (double)task_count,
        priority_turn / (double)task_count,
        priority_response / (double)task_count
    );

    printf(
        "%-15s %-12.2f %-15.2f %-15.2f\n",
        "Round Robin",
        rr_wait / (double)task_count,
        rr_turn / (double)task_count,
        rr_response / (double)task_count
    );

    printf("=======================================\n");
}