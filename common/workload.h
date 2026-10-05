#ifndef WORKLOAD_H
#define WORKLOAD_H


// Simple Task Structure
// Shared workload definition used by CPU and GPU

typedef struct {

    int id;                         // Task ID
    unsigned long long input;       // Initial input value

} SimpleTask;


// Workload Generator
// Generate the same deterministic workload for CPU and GPU


void GenerateSimpleWorkload(
    SimpleTask tasks[],
    int task_count
);


#endif