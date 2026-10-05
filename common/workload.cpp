#include "workload.h"


// Workload Generator
// Generate deterministic input values for all tasks

void GenerateSimpleWorkload(
    SimpleTask tasks[],
    int task_count
) {

    for (int i = 0; i < task_count; ++i) {

        tasks[i].id =
            i;

        tasks[i].input =
            123456789ULL
            + static_cast<unsigned long long>(i) * 1000003ULL;
    }
}